import MLX

public struct SamplingParameters: Sendable, Equatable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float
    public var presencePenalty: Float
    public var frequencyPenalty: Float
    public var xtcProbability: Float
    public var xtcThreshold: Float
    public var xtcSpecialTokens: [Int]
    public var seed: Int?
    public var logprobCount: Int?

    public init(
        temperature: Float = 0,
        topP: Float = 0,
        topK: Int = 0,
        minP: Float = 0,
        repetitionPenalty: Float = 1,
        presencePenalty: Float = 0,
        frequencyPenalty: Float = 0,
        xtcProbability: Float = 0,
        xtcThreshold: Float = 0.1,
        xtcSpecialTokens: [Int] = [],
        seed: Int? = nil,
        logprobCount: Int? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.xtcProbability = xtcProbability
        self.xtcThreshold = xtcThreshold
        self.xtcSpecialTokens = xtcSpecialTokens
        self.seed = seed
        self.logprobCount = logprobCount
    }

    var hasSamplingFiltersOrPenalties: Bool {
        (topP > 0 && topP < 1)
            || topK > 0
            || minP > 0
            || repetitionPenalty != 1
            || presencePenalty != 0
            || frequencyPenalty != 0
            || xtcProbability > 0
    }
}

public enum TokenSampler {
    public static func sample(
        logprobs logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokens: [Int] = []
    ) -> MLXArray {
        if parameters.temperature == 0 && !parameters.hasSamplingFiltersOrPenalties {
            return argMax(logits, axis: -1).reshaped([1])
        }

        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }
        logits = applyPenalties(logits, parameters: parameters, generatedTokens: generatedTokens)

        if parameters.temperature == 0 {
            return argMax(logits, axis: -1).reshaped([1])
        }

        var logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
        if parameters.topP > 0 && parameters.topP < 1 {
            logprobs = applyTopP(logprobs, topP: parameters.topP)
        }
        if parameters.minP > 0 {
            logprobs = applyMinP(logprobs, minP: parameters.minP)
        }
        if parameters.xtcProbability > 0 {
            logprobs = applyXTC(
                logprobs,
                probability: parameters.xtcProbability,
                threshold: parameters.xtcThreshold,
                specialTokens: parameters.xtcSpecialTokens
            )
        }
        if parameters.topK > 0 {
            logprobs = applyTopK(logprobs, topK: parameters.topK)
        }

        let scaled = logprobs / MLXArray(parameters.temperature)
        return MLXRandom.categorical(scaled, axis: -1).asType(.int32).reshaped([1])
    }

    private static func applyPenalties(
        _ logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokens: [Int]
    ) -> MLXArray {
        guard !generatedTokens.isEmpty else { return logits }

        var result = logits
        let counts = tokenCounts(generatedTokens)
        let uniqueTokens = Array(counts.keys).sorted()
        let indices = MLXArray(uniqueTokens.map(Int32.init)).asType(.uint32)

        if parameters.repetitionPenalty != 1 {
            var selectedLogits = result[indices]
            selectedLogits = MLX.where(
                selectedLogits .< 0,
                selectedLogits * parameters.repetitionPenalty,
                selectedLogits / parameters.repetitionPenalty
            )
            result = putAlong(result, indices, values: selectedLogits, axis: -1)
        }

        if parameters.presencePenalty != 0 {
            result = putAlong(
                result,
                indices,
                values: result[indices] - parameters.presencePenalty,
                axis: -1
            )
        }

        if parameters.frequencyPenalty != 0 {
            let frequencyPenalties = uniqueTokens.map { token in
                Float(counts[token] ?? 0) * parameters.frequencyPenalty
            }
            result = putAlong(
                result,
                indices,
                values: result[indices] - MLXArray(frequencyPenalties),
                axis: -1
            )
        }

        return result
    }

    private static func tokenCounts(_ tokens: [Int]) -> [Int: Int] {
        var counts: [Int: Int] = [:]
        for token in tokens {
            counts[token, default: 0] += 1
        }
        return counts
    }

    private static func applyTopP(_ logprobs: MLXArray, topP: Float) -> MLXArray {
        let topP = MLXArray(topP)
        let negInf = MLXArray(-Float.infinity)
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let cumulativeProbs = cumsum(exp(sortedLogprobs), axis: -1)
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    private static func applyMinP(_ logprobs: MLXArray, minP: Float) -> MLXArray {
        let threshold = logprobs.max(axis: -1, keepDims: true) + log(MLXArray(minP))
        return MLX.where(logprobs .>= threshold, logprobs, MLXArray(-Float.infinity))
    }

    private static func applyXTC(
        _ logprobs: MLXArray,
        probability: Float,
        threshold: Float,
        specialTokens: [Int]
    ) -> MLXArray {
        let threshold = max(0, min(threshold, 0.5))
        let probability = max(0, min(probability, 1))
        let probs = softmax(logprobs, axis: -1)
        let aboveThreshold = MLX.where(probs .> threshold, probs, MLXArray(Float.infinity))
        let cutoff = aboveThreshold.min(axis: -1, keepDims: true)
        var mask = probs .> cutoff
        let validSpecialTokens = specialTokens.filter { $0 >= 0 && $0 < logprobs.dim(-1) }
        if !validSpecialTokens.isEmpty {
            let indices = MLXArray(validSpecialTokens.map(Int32.init)).asType(.uint32)
            let values = MLXArray(Array(repeating: false, count: validSpecialTokens.count))
            mask = putAlong(mask, indices, values: values, axis: -1)
        }
        let shouldApply = MLXRandom.uniform(0 ..< 1) .<= probability
        return MLX.where(shouldApply, MLX.where(mask, MLXArray(-Float.infinity), logprobs), logprobs)
    }

    private static func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK > 0 && topK < vocabularySize else {
            return logprobs
        }

        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[topK...]
        return putAlong(logprobs, maskIndices, values: MLXArray(-Float.infinity), axis: -1)
    }
}
