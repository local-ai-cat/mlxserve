import Foundation
import MLX
import MLXLMCommon

public final class LanguageModelBox: @unchecked Sendable {
    let model: any LanguageModel

    public init(_ model: any LanguageModel) {
        self.model = model
    }
}

public actor Scheduler {
    private let model: any LanguageModel
    private let parameters: GenerateParameters
    private let generator: ContinuousBatchGenerator
    private let maxConcurrentRequests: Int
    private let queueLimit: Int
    private let prefixStore: (any PrefixKVStore)?
    private let prefixCacheEnabled: Bool
    private let serializedDecode: Bool
    private let schedulerManagedTextPrefill: Bool
    private var waiting: [Request] = []
    private var running: [String: RunningRequest] = [:]
    private var admissionInProgress: AdmissionInProgress?
    private var pendingCancellation: Set<String> = []
    private let collector = OutputCollector()

    public init(
        modelBox: LanguageModelBox,
        parameters: GenerateParameters,
        maxConcurrentRequests: Int,
        prefixStore: (any PrefixKVStore)? = nil,
        cacheCapabilities: ModelCacheCapabilities = .default,
        serializedDecode: Bool = false,
        schedulerManagedTextPrefill: Bool = true
    ) {
        self.model = modelBox.model
        self.parameters = parameters
        self.generator = ContinuousBatchGenerator(model: modelBox.model, parameters: parameters)
        self.maxConcurrentRequests = maxConcurrentRequests
        self.queueLimit = max(maxConcurrentRequests * 4, 32)
        self.prefixStore = prefixStore
        self.prefixCacheEnabled = prefixStore != nil
            && !Self.usesWindowedKVCache(parameters: parameters, cacheCapabilities: cacheCapabilities)
        self.serializedDecode = serializedDecode
        self.schedulerManagedTextPrefill = schedulerManagedTextPrefill
    }

    public var isIdle: Bool {
        waiting.isEmpty && running.isEmpty && generator.isEmpty && admissionInProgress == nil
    }

    public var queueDepth: Int {
        waiting.count + running.count + (admissionInProgress == nil ? 0 : 1)
    }

    public func submit(_ request: Request) throws {
        if queueDepth >= queueLimit {
            throw SchedulerError.queueFull(retryAfterSteps: max(1, queueDepth / max(1, maxConcurrentRequests)))
        }
        if waiting.contains(where: { $0.uid == request.uid })
            || running[request.uid] != nil
            || admissionInProgress?.request.uid == request.uid
        {
            throw SchedulerError.duplicateRequest(request.uid)
        }
        waiting.append(request)
    }

    public func cancel(uid: String) {
        if let waitingIndex = waiting.firstIndex(where: { $0.uid == uid }) {
            waiting.remove(at: waitingIndex)
            collector.record(Response(uid: uid, token: -1, finishReason: .cancelled))
            return
        }
        if let admission = admissionInProgress, admission.request.uid == uid {
            if let hit = admission.prefixHit {
                prefixStore?.release(hit)
            }
            admissionInProgress = nil
            collector.record(Response(uid: uid, token: -1, finishReason: .cancelled))
            return
        }
        if running[uid] != nil {
            pendingCancellation.insert(uid)
        }
    }

    public func step() throws -> [Response] {
        var processed = applyPendingCancellation()
        processed.append(contentsOf: admitWaiting(allowPartialPrefill: !generator.isEmpty))

        guard !generator.isEmpty else { return processed }

        let rawResponses = generator.next()
        var finishedUIDs: [String] = []

        for response in rawResponses {
            guard var runningRequest = running[response.uid] else { continue }
            runningRequest.generatedTokenCount += 1
            if prefixCacheEnabled,
                prefixStore != nil,
                runningRequest.generatedTokenCount == 1,
                runningRequest.promptCacheSnapshot == nil,
                !runningRequest.promptTokens.isEmpty
            {
                runningRequest.promptCacheSnapshot = promptCacheSnapshot(uid: response.uid)
            }

            let finishReason: FinishReason?
            if runningRequest.request.eosTokenIds.contains(response.token) {
                finishReason = .stop
            } else if runningRequest.generatedTokenCount >= runningRequest.request.maxTokens {
                finishReason = .length
            } else {
                finishReason = nil
            }

            running[response.uid] = runningRequest
            let processedResponse = Response(
                uid: response.uid,
                token: response.token,
                finishReason: finishReason,
                logprobs: response.logprobs
            )
            processed.append(processedResponse)
            collector.record(processedResponse)

            if finishReason != nil {
                finishedUIDs.append(response.uid)
            }
        }

        if !finishedUIDs.isEmpty {
            Stream.gpu.synchronize()
            for uid in finishedUIDs {
                let finishedRequest = running[uid]
                generator.remove(uid: uid)
                if let hit = finishedRequest?.prefixHit {
                    prefixStore?.release(hit)
                }
                running.removeValue(forKey: uid)
                if let finishedRequest {
                    storePromptCache(from: finishedRequest)
                }
            }
        }

        if generator.isEmpty {
            processed.append(contentsOf: admitWaiting(allowPartialPrefill: false))
        }
        return processed
    }

    public func collectedTokens() -> [String: [Int]] {
        collector.allTokens
    }

    public func responses(for uid: String) -> [Response] {
        collector.responses(for: uid)
    }

    public func consumeTokens(for uid: String) -> [Int] {
        collector.consumeTokens(for: uid)
    }

    public func tokens(for uid: String) -> [Int] {
        collector.tokens(for: uid)
    }

    public func discardResponses(for uid: String) {
        collector.remove(uid: uid)
    }

    private func admitWaiting(allowPartialPrefill: Bool) -> [Response] {
        var admittedResponses: [Response] = []
        while running.count < maxConcurrentRequests {
            // Some model architectures still derive RoPE position ids or
            // shared-KV offsets from scalar cache.offset. Their loaders pass
            // serializedDecode=true so mixed-offset rows are not admitted into
            // the same decode batch.
            if serializedDecode, !running.isEmpty || !generator.isEmpty || !admittedResponses.isEmpty {
                return admittedResponses
            }

            if admissionInProgress == nil {
                guard !waiting.isEmpty else { return admittedResponses }
                let request = waiting.removeFirst()
                do {
                    var sampling = request.sampling
                    sampling.eosTokenIds.formUnion(request.eosTokenIds)
                    if !request.eosTokenIds.isEmpty {
                        sampling.xtcSpecialTokens = Array(Set(sampling.xtcSpecialTokens).union(request.eosTokenIds))
                    }

                    switch try prepareForInsert(request, sampling: sampling) {
                    case .ready(let row):
                        if let response = try completeAdmission(
                            row,
                            request: request,
                            sampling: sampling
                        ) {
                            admittedResponses.append(response)
                        }
                        continue
                    case .pending(let admission):
                        admissionInProgress = admission
                    }
                } catch {
                    let response = Response(
                        uid: request.uid,
                        token: -1,
                        finishReason: .failed(String(describing: error))
                    )
                    collector.record(response)
                    admittedResponses.append(response)
                    continue
                }
            }

            do {
                guard let row = try advanceAdmission(allowPartialPrefill: allowPartialPrefill) else {
                    return admittedResponses
                }
                guard let admission = admissionInProgress else { return admittedResponses }
                if let response = try completeAdmission(
                    row,
                    request: admission.request,
                    sampling: admission.sampling
                ) {
                    admittedResponses.append(response)
                }
                admissionInProgress = nil
            } catch {
                let request = admissionInProgress?.request
                if let hit = admissionInProgress?.prefixHit {
                    prefixStore?.release(hit)
                }
                admissionInProgress = nil
                if let request {
                    let response = Response(
                        uid: request.uid,
                        token: -1,
                        finishReason: .failed(String(describing: error))
                    )
                    collector.record(response)
                    admittedResponses.append(response)
                }
                continue
            }
        }
        return admittedResponses
    }

    private func applyPendingCancellation() -> [Response] {
        guard !pendingCancellation.isEmpty else { return [] }

        Stream.gpu.synchronize()
        var responses: [Response] = []
        for uid in pendingCancellation {
            guard let runningRequest = running[uid] else { continue }
            if let hit = runningRequest.prefixHit {
                prefixStore?.release(hit)
            }
            generator.remove(uid: uid)
            running.removeValue(forKey: uid)
            let response = Response(uid: uid, token: -1, finishReason: .cancelled)
            collector.record(response)
            responses.append(response)
        }
        pendingCancellation.removeAll()
        return responses
    }

    private func completeAdmission(
        _ row: PreparedBatchRow,
        request: Request,
        sampling: SamplingParameters
    ) throws -> Response? {
        let initialTokenID = row.initialGeneratedToken?.tokenID
        let generatedTokens = initialTokenID.map { [$0] } ?? []
        let generatedTokenCount = initialTokenID == nil ? 0 : 1
        let finishReason = initialTokenID.map { tokenID in
            self.finishReason(
                token: tokenID,
                generatedTokenCount: generatedTokenCount,
                request: request
            )
        } ?? nil

        if finishReason == nil {
            try generator.insert(
                uid: request.uid,
                cache: row.cache,
                lastToken: row.lastToken,
                sampling: sampling,
                generatedTokens: generatedTokens,
                thinkingBudgetState: row.initialGeneratedToken?.thinkingBudgetState
            )
            running[request.uid] = RunningRequest(
                request: request,
                promptTokens: row.promptTokens,
                prefixHit: row.prefixHit,
                generatedTokenCount: generatedTokenCount,
                promptCacheSnapshot: nil
            )
        }

        guard let initialTokenID else { return nil }
        let response = Response(
            uid: request.uid,
            token: initialTokenID,
            finishReason: finishReason
        )
        collector.record(response)
        return response
    }

    private func advanceAdmission(allowPartialPrefill: Bool) throws -> PreparedBatchRow? {
        guard var admission = admissionInProgress else { return nil }

        let stepBudget = allowPartialPrefill ? max(1, parameters.prefillStepSize) : Int.max
        var remainingBudget = stepBudget
        while admission.nextPrefillIndex < admission.prefillRange.upperBound, remainingBudget > 0 {
            let chunkLimit = allowPartialPrefill
                ? min(max(1, parameters.prefillStepSize), remainingBudget)
                : max(1, parameters.prefillStepSize)
            let end = min(
                admission.nextPrefillIndex + chunkLimit,
                admission.prefillRange.upperBound
            )
            let input = LMInput.Text(
                tokens: admission.promptTokensArray[admission.nextPrefillIndex ..< end]
            )
            let output = model(input[text: .newAxis], cache: admission.cache, state: admission.state)
            admission.state = output.state
            asyncEval(admission.cache)
            remainingBudget -= end - admission.nextPrefillIndex
            admission.nextPrefillIndex = end
        }

        if admission.nextPrefillIndex < admission.prefillRange.upperBound {
            admissionInProgress = admission
            return nil
        }

        eval(admission.cache)
        return PreparedBatchRow(
            cache: admission.cache,
            lastToken: admission.promptTokensArray[admission.promptTokens.count - 1],
            promptTokens: admission.storedPromptTokens,
            prefixHit: admission.prefixHit,
            initialGeneratedToken: nil
        )
    }

    private func prepareForInsert(_ request: Request, sampling: SamplingParameters) throws
        -> PreparedAdmission
    {
        let rowCache = model.newCache(parameters: parameters)
        let prefixCacheEligible = isPrefixCacheEligible(request.input)
        let canUseRawTextTokens = schedulerManagedTextPrefill
            && prefixCacheEligible
            && request.input.text.tokens.ndim == 1
            && request.input.text.mask == nil
        let promptText: LMInput.Text
        if canUseRawTextTokens {
            promptText = request.input.text
        } else {
            switch try model.prepare(request.input, cache: rowCache, windowSize: parameters.prefillStepSize) {
            case .tokens(let tokens):
                promptText = tokens
            case .logits(let output):
                let firstToken = sampledToken(from: output.logits, sampling: sampling)
                return .ready(PreparedBatchRow(
                    cache: rowCache,
                    lastToken: firstToken.token,
                    promptTokens: [],
                    prefixHit: nil,
                    initialGeneratedToken: firstToken
                ))
            }
        }

        let promptTokensArray = promptText.tokens
        let promptTokenCount = promptTokensArray.dim(0)
        guard promptTokenCount > 0 else {
            throw BatchGeneratorError.promptTooShortForExternalPrefill
        }
        let promptTokens = promptTokensArray.asArray(Int.self)

        if promptTokenCount == 1 {
            let output = model(promptText[text: .newAxis], cache: rowCache, state: nil)
            eval(rowCache)
            let firstToken = sampledToken(from: output.logits, sampling: sampling)
            return .ready(PreparedBatchRow(
                cache: rowCache,
                lastToken: firstToken.token,
                promptTokens: [],
                prefixHit: nil,
                initialGeneratedToken: firstToken
            ))
        }

        if prefixCacheEnabled,
            prefixCacheEligible,
            let prefixStore,
            let hit = prefixStore.fetch(tokens: promptTokens, sessionKey: request.cacheSession)
        {
            do {
                let serialized = try prefixStore.reconstructCache(from: hit)
                let reconstructedCache = try serialized.map {
                    try BlockAwarePrefixKVStore.cache(from: $0)
                }
                return try prepareHitRow(
                    request: request,
                    sampling: sampling,
                    promptTokens: promptTokens,
                    promptTokensArray: promptTokensArray,
                    hit: hit,
                    reconstructedCache: reconstructedCache
                )
            } catch {
                prefixStore.release(hit)
                logCacheFailure("prefix hit reconstruction failed; falling back to cache miss", error)
            }
        }

        return prefillMissRow(
            request: request,
            sampling: sampling,
            promptTokens: promptTokens,
            promptTokensArray: promptTokensArray,
            storedPromptTokens: prefixCacheEligible ? promptTokens : [],
            rowCache: rowCache
        )
    }

    private func prepareHitRow(
        request: Request,
        sampling: SamplingParameters,
        promptTokens: [Int],
        promptTokensArray: MLXArray,
        hit: PrefixKVStoreHit,
        reconstructedCache: [KVCacheSimple]
    ) throws -> PreparedAdmission {
        let matched = hit.matchedTokenCount
        guard matched <= promptTokens.count else {
            throw SchedulerError.invalidPrefixHit
        }

        if matched == promptTokens.count {
            for layerCache in reconstructedCache {
                _ = layerCache.trim(1)
            }
            return .ready(PreparedBatchRow(
                cache: reconstructedCache,
                lastToken: promptTokensArray[promptTokens.count - 1],
                promptTokens: promptTokens,
                prefixHit: hit,
                initialGeneratedToken: nil
            ))
        }

        return .pending(AdmissionInProgress(
            request: request,
            sampling: sampling,
            cache: reconstructedCache,
            promptTokens: promptTokens,
            promptTokensArray: promptTokensArray,
            storedPromptTokens: promptTokens,
            prefixHit: hit,
            prefillRange: matched ..< (promptTokens.count - 1),
            nextPrefillIndex: matched,
            state: nil
        ))
    }

    private func prefillMissRow(
        request: Request,
        sampling: SamplingParameters,
        promptTokens: [Int],
        promptTokensArray: MLXArray,
        storedPromptTokens: [Int],
        rowCache: [any KVCache]
    ) -> PreparedAdmission {
        .pending(AdmissionInProgress(
            request: request,
            sampling: sampling,
            cache: rowCache,
            promptTokens: promptTokens,
            promptTokensArray: promptTokensArray,
            storedPromptTokens: storedPromptTokens,
            prefixHit: nil,
            prefillRange: 0 ..< (promptTokens.count - 1),
            nextPrefillIndex: 0,
            state: nil
        ))
    }

    private func promptCacheSnapshot(uid: String) -> [SerializedKVLayer]? {
        guard let cache = generator.extractCache(uid: uid) else {
            return nil
        }
        return cache.map {
            SerializedKVLayer(
                state: $0.state,
                metaState: $0.metaState,
                className: "KVCacheSimple"
            )
        }
    }

    private func storePromptCache(from runningRequest: RunningRequest) {
        guard prefixCacheEnabled,
            let prefixStore,
            !runningRequest.promptTokens.isEmpty,
            let snapshot = runningRequest.promptCacheSnapshot
        else {
            return
        }
        do {
            try prefixStore.store(
                tokens: runningRequest.promptTokens,
                sessionKey: runningRequest.request.cacheSession,
                cache: snapshot
            )
        } catch {
            logCacheFailure("prompt cache store failed", error)
        }
    }

    private func logCacheFailure(_ message: String, _ error: Error) {
        let line = "MLXServe cache warning: \(message): \(error)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private func finishReason(
        token: Int,
        generatedTokenCount: Int,
        request: Request
    ) -> FinishReason? {
        if request.eosTokenIds.contains(token) {
            return .stop
        }
        if generatedTokenCount >= request.maxTokens {
            return .length
        }
        return nil
    }

    private func sampledToken(
        from logits: MLXArray,
        sampling: SamplingParameters
    ) -> PreparedGeneratedToken {
        let nextTokenLogits = logits[0..., -1, 0...]
        let matcher = sampling.jsonGrammar?.makeMatcher()
        let regexMatcher = sampling.regexGrammar?.makeMatcher()
        let gbnfMatcher = sampling.gbnfGrammar?.makeMatcher()
        var thinkingBudgetState = sampling.thinkingBudget.map(ThinkingBudgetState.init(configuration:))
        let token = TokenSampler.sample(
            logits: nextTokenLogits[0, 0...],
            parameters: sampling,
            generatedTokens: [],
            jsonGrammarMatcher: matcher,
            regexGrammarMatcher: regexMatcher,
            gbnfGrammarMatcher: gbnfMatcher,
            thinkingBudgetState: &thinkingBudgetState
        )
        let tokenID = token.item(Int.self)
        if matcher?.accepts(tokenID: tokenID) == true {
            matcher?.advance(tokenID: tokenID)
        }
        if regexMatcher?.accepts(tokenID: tokenID) == true {
            regexMatcher?.advance(tokenID: tokenID)
        }
        if gbnfMatcher?.accepts(tokenID: tokenID) == true {
            gbnfMatcher?.advance(tokenID: tokenID)
        }
        thinkingBudgetState?.advance(tokenID: tokenID)
        return PreparedGeneratedToken(
            token: token,
            tokenID: tokenID,
            thinkingBudgetState: thinkingBudgetState
        )
    }

    private func isPrefixCacheEligible(_ input: LMInput) -> Bool {
        input.image == nil && input.video == nil && input.audio == nil
    }

    private static func usesWindowedKVCache(
        parameters: GenerateParameters,
        cacheCapabilities: ModelCacheCapabilities
    ) -> Bool {
        parameters.maxKVSize != nil || cacheCapabilities.usesWindowedKVCache
    }
}

private enum PreparedAdmission {
    case ready(PreparedBatchRow)
    case pending(AdmissionInProgress)
}

private struct AdmissionInProgress {
    let request: Request
    let sampling: SamplingParameters
    let cache: [any KVCache]
    let promptTokens: [Int]
    let promptTokensArray: MLXArray
    let storedPromptTokens: [Int]
    let prefixHit: PrefixKVStoreHit?
    let prefillRange: Range<Int>
    var nextPrefillIndex: Int
    var state: LMOutput.State?
}

private struct PreparedBatchRow {
    let cache: [any KVCache]
    let lastToken: MLXArray
    let promptTokens: [Int]
    let prefixHit: PrefixKVStoreHit?
    let initialGeneratedToken: PreparedGeneratedToken?
}

private struct PreparedGeneratedToken {
    let token: MLXArray
    let tokenID: Int
    let thinkingBudgetState: ThinkingBudgetState?
}
