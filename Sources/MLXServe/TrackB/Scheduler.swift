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
    private var waiting: [Request] = []
    private var running: [String: RunningRequest] = [:]
    private var pendingCancellation: Set<String> = []
    private let collector = OutputCollector()

    public init(
        modelBox: LanguageModelBox,
        parameters: GenerateParameters,
        maxConcurrentRequests: Int,
        prefixStore: (any PrefixKVStore)? = nil
    ) {
        self.model = modelBox.model
        self.parameters = parameters
        self.generator = ContinuousBatchGenerator(model: modelBox.model, parameters: parameters)
        self.maxConcurrentRequests = maxConcurrentRequests
        self.queueLimit = max(maxConcurrentRequests * 4, 32)
        self.prefixStore = prefixStore
    }

    public var isIdle: Bool {
        waiting.isEmpty && running.isEmpty && generator.isEmpty
    }

    public var queueDepth: Int {
        waiting.count + running.count
    }

    public func submit(_ request: Request) throws {
        if queueDepth >= queueLimit {
            throw SchedulerError.queueFull(retryAfterSteps: max(1, queueDepth / max(1, maxConcurrentRequests)))
        }
        if waiting.contains(where: { $0.uid == request.uid }) || running[request.uid] != nil {
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
        if running[uid] != nil {
            pendingCancellation.insert(uid)
        }
    }

    public func step() throws -> [Response] {
        applyPendingCancellation()
        try admitWaiting()

        guard !generator.isEmpty else { return [] }

        let rawResponses = generator.next()
        var processed: [Response] = []
        var finishedUIDs: [String] = []

        for response in rawResponses {
            guard var runningRequest = running[response.uid] else { continue }
            runningRequest.generatedTokenCount += 1

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
                storeFinishedPromptCache(uid: uid)
                generator.remove(uid: uid)
                releasePrefixHit(uid: uid)
                running.removeValue(forKey: uid)
            }
        }

        try admitWaiting()
        return processed
    }

    public func collectedTokens() -> [String: [Int]] {
        collector.allTokens
    }

    public func responses(for uid: String) -> [Response] {
        collector.responses(for: uid)
    }

    private func admitWaiting() throws {
        while running.count < maxConcurrentRequests, !waiting.isEmpty {
            let request = waiting.removeFirst()
            let prepared = try prepareForInsert(request)
            generator.insert(
                uid: request.uid,
                cache: prepared.cache,
                lastToken: prepared.lastToken,
                sampling: request.sampling
            )
            running[request.uid] = RunningRequest(
                request: request,
                promptTokens: prepared.promptTokens,
                prefixHit: prepared.prefixHit,
                generatedTokenCount: 0
            )
        }
    }

    private func applyPendingCancellation() {
        guard !pendingCancellation.isEmpty else { return }

        Stream.gpu.synchronize()
        for uid in pendingCancellation {
            guard running[uid] != nil else { continue }
            clearPrefixEntry(uid: uid)
            generator.remove(uid: uid)
            running.removeValue(forKey: uid)
            collector.record(Response(uid: uid, token: -1, finishReason: .cancelled))
        }
        pendingCancellation.removeAll()
    }

    private func prepareForInsert(_ request: Request) throws -> PreparedBatchRow {
        let rowCache = model.newCache(parameters: parameters)
        let promptText: LMInput.Text
        switch try model.prepare(request.input, cache: rowCache, windowSize: parameters.prefillStepSize) {
        case .tokens(let tokens):
            promptText = tokens
        case .logits:
            throw BatchGeneratorError.unsupportedPreparedLogits
        }

        let promptTokensArray = promptText.tokens
        let promptTokenCount = promptTokensArray.dim(0)
        guard promptTokenCount > 1 else {
            throw BatchGeneratorError.promptTooShortForExternalPrefill
        }
        let promptTokens = promptTokensArray.asArray(Int.self)

        if let prefixStore, let hit = prefixStore.fetch(tokens: promptTokens) {
            do {
                try prefixStore.preload(hit)
                let serialized = try prefixStore.reconstructCache(from: hit)
                let reconstructedCache = try serialized.map {
                    try BlockAwarePrefixKVStore.cache(from: $0)
                }
                return try prepareHitRow(
                    request: request,
                    promptTokens: promptTokens,
                    promptTokensArray: promptTokensArray,
                    hit: hit,
                    reconstructedCache: reconstructedCache
                )
            } catch {
                prefixStore.release(hit)
                throw error
            }
        }

        return prefillMissRow(
            request: request,
            promptTokens: promptTokens,
            promptTokensArray: promptTokensArray,
            rowCache: rowCache
        )
    }

    private func prepareHitRow(
        request: Request,
        promptTokens: [Int],
        promptTokensArray: MLXArray,
        hit: PrefixKVStoreHit,
        reconstructedCache: [KVCacheSimple]
    ) throws -> PreparedBatchRow {
        let matched = hit.matchedTokenCount
        guard matched <= promptTokens.count else {
            throw SchedulerError.invalidPrefixHit
        }

        if matched == promptTokens.count {
            for layerCache in reconstructedCache {
                _ = layerCache.trim(1)
            }
            return PreparedBatchRow(
                cache: reconstructedCache,
                lastToken: promptTokensArray[promptTokens.count - 1],
                promptTokens: promptTokens,
                prefixHit: hit
            )
        }

        let remainingCount = promptTokens.count - matched
        if remainingCount > 1 {
            let suffixPrefix = LMInput.Text(
                tokens: promptTokensArray[matched ..< (promptTokens.count - 1)]
            )
            _ = model(suffixPrefix[text: .newAxis], cache: reconstructedCache, state: nil)
            eval(reconstructedCache)
        }

        return PreparedBatchRow(
            cache: reconstructedCache,
            lastToken: promptTokensArray[promptTokens.count - 1],
            promptTokens: promptTokens,
            prefixHit: hit
        )
    }

    private func prefillMissRow(
        request: Request,
        promptTokens: [Int],
        promptTokensArray: MLXArray,
        rowCache: [any KVCache]
    ) -> PreparedBatchRow {
        let prefixInput = LMInput.Text(tokens: promptTokensArray[..<(promptTokens.count - 1)])
        _ = model(prefixInput[text: .newAxis], cache: rowCache, state: nil)
        eval(rowCache)

        return PreparedBatchRow(
            cache: rowCache,
            lastToken: promptTokensArray[promptTokens.count - 1],
            promptTokens: promptTokens,
            prefixHit: nil
        )
    }

    private func storeFinishedPromptCache(uid: String) {
        guard let prefixStore, let runningRequest = running[uid],
            let cache = generator.extractCache(uid: uid)
        else {
            return
        }

        let serialized = cache.map {
            SerializedKVLayer(
                state: $0.state,
                metaState: $0.metaState,
                className: "KVCacheSimple"
            )
        }
        try? prefixStore.store(tokens: runningRequest.promptTokens, cache: serialized)
    }

    private func releasePrefixHit(uid: String) {
        guard let hit = running[uid]?.prefixHit else { return }
        prefixStore?.release(hit)
    }

    private func clearPrefixEntry(uid: String) {
        guard let hit = running[uid]?.prefixHit else { return }
        prefixStore?.clearEntry(hit)
    }
}

private struct PreparedBatchRow {
    let cache: [any KVCache]
    let lastToken: MLXArray
    let promptTokens: [Int]
    let prefixHit: PrefixKVStoreHit?
}
