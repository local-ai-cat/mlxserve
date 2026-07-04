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
        var processed = applyPendingCancellation()
        processed.append(contentsOf: admitWaiting())

        guard !generator.isEmpty else { return processed }

        let rawResponses = generator.next()
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

        processed.append(contentsOf: admitWaiting())
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

    private func admitWaiting() -> [Response] {
        var failedResponses: [Response] = []
        while running.count < maxConcurrentRequests, !waiting.isEmpty {
            let request = waiting[0]
            var prepared: PreparedBatchRow?
            do {
                let row = try prepareForInsert(request)
                prepared = row
                var sampling = request.sampling
                if !request.eosTokenIds.isEmpty {
                    sampling.xtcSpecialTokens = Array(Set(sampling.xtcSpecialTokens).union(request.eosTokenIds))
                }
                try generator.insert(
                    uid: request.uid,
                    cache: row.cache,
                    lastToken: row.lastToken,
                    sampling: sampling
                )
                waiting.removeFirst()
                running[request.uid] = RunningRequest(
                    request: request,
                    promptTokens: row.promptTokens,
                    prefixHit: row.prefixHit,
                    generatedTokenCount: 0
                )
            } catch {
                if let hit = prepared?.prefixHit {
                    prefixStore?.release(hit)
                }
                waiting.removeFirst()
                let response = Response(
                    uid: request.uid,
                    token: -1,
                    finishReason: .failed(String(describing: error))
                )
                collector.record(response)
                failedResponses.append(response)
                continue
            }
        }
        return failedResponses
    }

    private func applyPendingCancellation() -> [Response] {
        guard !pendingCancellation.isEmpty else { return [] }

        Stream.gpu.synchronize()
        var responses: [Response] = []
        for uid in pendingCancellation {
            guard running[uid] != nil else { continue }
            clearPrefixEntry(uid: uid)
            generator.remove(uid: uid)
            running.removeValue(forKey: uid)
            let response = Response(uid: uid, token: -1, finishReason: .cancelled)
            collector.record(response)
            responses.append(response)
        }
        pendingCancellation.removeAll()
        return responses
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
                logCacheFailure("prefix hit reconstruction failed; falling back to cache miss", error)
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
        do {
            try prefixStore.store(tokens: runningRequest.promptTokens, cache: serialized)
        } catch {
            logCacheFailure("prompt cache store failed", error)
        }
    }

    private func releasePrefixHit(uid: String) {
        guard let hit = running[uid]?.prefixHit else { return }
        prefixStore?.release(hit)
    }

    private func clearPrefixEntry(uid: String) {
        guard let hit = running[uid]?.prefixHit else { return }
        prefixStore?.clearEntry(hit)
    }

    private func logCacheFailure(_ message: String, _ error: Error) {
        let line = "MLXServe cache warning: \(message): \(error)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}

private struct PreparedBatchRow {
    let cache: [any KVCache]
    let lastToken: MLXArray
    let promptTokens: [Int]
    let prefixHit: PrefixKVStoreHit?
}
