import MLX
import MLXLMCommon

public struct BatchDecodeStep {
    public let tokenIds: [Int]
    public let logits: MLXArray
}

public enum FinishReason: Sendable, Equatable {
    case stop
    case length
    case cancelled
    case failed(String)
}

public struct Response: Sendable, Equatable {
    public let uid: String
    public let token: Int
    public let finishReason: FinishReason?
    public let logprobs: [Int: Float]?

    public init(
        uid: String,
        token: Int,
        finishReason: FinishReason? = nil,
        logprobs: [Int: Float]? = nil
    ) {
        self.uid = uid
        self.token = token
        self.finishReason = finishReason
        self.logprobs = logprobs
    }
}

public final class StaticBatchGenerator {
    private let model: any LanguageModel
    private var cache: [BatchLayerCache]
    private var currentTokens: MLXArray
    private var currentLogits: MLXArray
    private var state: LMOutput.State?

    public init(
        model: any LanguageModel,
        inputs: [LMInput],
        parameters: GenerateParameters
    ) throws {
        precondition(!inputs.isEmpty, "StaticBatchGenerator requires at least one input")

        self.model = model

        var rowCaches: [[any KVCache]] = []
        var firstTokens: [MLXArray] = []
        var firstLogits: [MLXArray] = []

        for input in inputs {
            let cache = model.newCache(parameters: parameters)
            let remaining: LMInput.Text
            switch try model.prepare(input, cache: cache, windowSize: parameters.prefillStepSize) {
            case .tokens(let tokens):
                remaining = tokens
            case .logits(let output):
                let logits = output.logits[0..., -1, 0...]
                firstLogits.append(logits)
                firstTokens.append(argMax(logits, axis: -1))
                rowCaches.append(cache)
                continue
            }

            let output = model(remaining[text: .newAxis], cache: cache, state: nil)
            let logits = output.logits[0..., -1, 0...]
            firstLogits.append(logits)
            firstTokens.append(argMax(logits, axis: -1))
            rowCaches.append(cache)
        }

        cache = try (0 ..< rowCaches[0].count).map { layer in
            try BatchLayerCache.merge(rowCaches.map { $0[layer] })
        }
        currentTokens = concatenated(firstTokens, axis: 0)
        currentLogits = concatenated(firstLogits, axis: 0)
        eval(currentTokens, currentLogits, cache.map(\.kvCache))
    }

    public func next() -> BatchDecodeStep {
        let returnedTokens = currentTokens
        let returnedLogits = currentLogits

        let output = model(
            LMInput.Text(tokens: currentTokens[0..., .newAxis]),
            cache: cache.map(\.kvCache),
            state: state
        )
        state = output.state
        currentLogits = output.logits[0..., -1, 0...]
        currentTokens = argMax(currentLogits, axis: -1)

        asyncEval(currentTokens)
        eval(returnedTokens, returnedLogits)

        return BatchDecodeStep(
            tokenIds: returnedTokens.asArray(Int.self),
            logits: returnedLogits
        )
    }
}

public final class ContinuousBatchGenerator {
    private let model: any LanguageModel
    private let parameters: GenerateParameters
    private var cache: [BatchLayerCache] = []
    private var currentTokens = MLXArray([Int32]())
    private var rowUIDs: [String] = []
    private var samplers: [SamplingParameters] = []
    private var generatedTokenHistory: [[Int]] = []
    private var state: LMOutput.State?

    public init(model: any LanguageModel, parameters: GenerateParameters) {
        self.model = model
        self.parameters = parameters
    }

    public var isEmpty: Bool {
        rowUIDs.isEmpty
    }

    public var count: Int {
        rowUIDs.count
    }

    public var uids: [String] {
        rowUIDs
    }

    public func insert(
        uid: String,
        input: LMInput,
        sampling: SamplingParameters
    ) throws {
        let prepared = try prefillWithLastTokenWithheld(input)
        try insert(uid: uid, cache: prepared.cache, lastToken: prepared.lastToken, sampling: sampling)
    }

    public func insert(
        uid: String,
        cache rowCache: [any KVCache],
        lastToken: MLXArray,
        sampling: SamplingParameters
    ) throws {
        precondition(!rowUIDs.contains(uid), "duplicate batch uid '\(uid)'")
        precondition(!rowCache.isEmpty, "continuous batching requires a non-empty KV cache")

        if let seed = sampling.seed {
            // MLXRandom.seed mutates global RNG state. In mixed batches, the last inserted
            // seeded request determines subsequent stochastic draws for all rows.
            MLXRandom.seed(UInt64(bitPattern: Int64(seed)))
        }

        if cache.isEmpty {
            let merged = try (0 ..< rowCache.count).map { layer in
                try BatchLayerCache.merge([rowCache[layer]])
            }
            cache = merged
            currentTokens = lastToken.reshaped([1])
        } else {
            guard cache.count == rowCache.count else {
                throw BatchGeneratorError.insertedCacheLayerCountChanged(
                    expected: cache.count,
                    actual: rowCache.count
                )
            }
            let rowLayers = try rowCache.map { try BatchLayerCache.merge([$0]) }
            cache = try zip(cache, rowLayers).map { currentLayer, rowLayer in
                let mergedLayer = currentLayer.copyLayer()
                try mergedLayer.extend(rowLayer)
                return mergedLayer
            }
            currentTokens = concatenated([currentTokens, lastToken.reshaped([1])], axis: 0)
        }

        rowUIDs.append(uid)
        samplers.append(sampling)
        generatedTokenHistory.append([])
        state = nil
        eval(currentTokens, cache.map(\.kvCache))
    }

    public func remove(uid: String) {
        guard let row = rowUIDs.firstIndex(of: uid) else { return }
        let keptRows = rowUIDs.indices.filter { $0 != row }
        filter(keeping: keptRows)
    }

    public func extractCache(uid: String) -> [KVCacheSimple]? {
        guard let row = rowUIDs.firstIndex(of: uid) else { return nil }
        return cache.map { $0.extract(row) }
    }

    public func filter(keeping rows: [Int]) {
        guard !rows.isEmpty else {
            cache.removeAll()
            rowUIDs.removeAll()
            samplers.removeAll()
            generatedTokenHistory.removeAll()
            currentTokens = MLXArray([Int32]())
            state = nil
            return
        }

        for layer in cache {
            layer.filter(keeping: rows)
        }
        let rowIndices = MLXArray(rows.map(Int32.init))
        currentTokens = currentTokens.take(rowIndices, axis: 0)
        rowUIDs = rows.map { rowUIDs[$0] }
        samplers = rows.map { samplers[$0] }
        generatedTokenHistory = rows.map { generatedTokenHistory[$0] }
        state = nil
        eval(currentTokens, cache.map(\.kvCache))
    }

    public func next() -> [Response] {
        guard !rowUIDs.isEmpty else { return [] }

        let output = model(
            LMInput.Text(tokens: currentTokens[0..., .newAxis]),
            cache: cache.map(\.kvCache),
            state: state
        )
        state = output.state

        let logits = output.logits[0..., -1, 0...]
        let sampledRows = (0 ..< rowUIDs.count).map { row in
            TokenSampler.sample(
                logprobs: logits[row, 0...],
                parameters: samplers[row],
                generatedTokens: generatedTokenHistory[row]
            )
        }
        let nextTokens = concatenated(sampledRows, axis: 0)

        asyncEval(nextTokens)
        eval(currentTokens, logits)
        currentTokens = nextTokens

        let tokenIds = nextTokens.asArray(Int.self)
        for row in generatedTokenHistory.indices {
            generatedTokenHistory[row].append(tokenIds[row])
        }
        return rowUIDs.enumerated().map { row, uid in
            Response(uid: uid, token: tokenIds[row])
        }
    }

    private func prefillWithLastTokenWithheld(_ input: LMInput) throws -> (
        cache: [any KVCache],
        lastToken: MLXArray
    ) {
        let rowCache = model.newCache(parameters: parameters)
        let remaining: LMInput.Text
        switch try model.prepare(input, cache: rowCache, windowSize: parameters.prefillStepSize) {
        case .tokens(let tokens):
            remaining = tokens
        case .logits:
            throw BatchGeneratorError.unsupportedPreparedLogits
        }

        let tokens = remaining.tokens
        let tokenCount = tokens.dim(0)
        guard tokenCount > 1 else {
            throw BatchGeneratorError.promptTooShortForExternalPrefill
        }

        let prefixTokens = tokens[..<(tokenCount - 1)]
        let prefixInput = LMInput.Text(tokens: prefixTokens)
        _ = model(prefixInput[text: .newAxis], cache: rowCache, state: nil)
        eval(rowCache)

        return (
            cache: rowCache,
            lastToken: tokens[tokenCount - 1]
        )
    }
}

public enum BatchGeneratorError: Error, Equatable {
    case unsupportedPreparedLogits
    case promptTooShortForExternalPrefill
    case insertedCacheLayerCountChanged(expected: Int, actual: Int)
}
