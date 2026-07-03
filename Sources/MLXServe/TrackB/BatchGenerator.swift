import MLX
import MLXLMCommon

public struct BatchDecodeStep {
    public let tokenIds: [Int]
    public let logits: MLXArray
}

public final class StaticBatchGenerator {
    private let model: any LanguageModel
    private var cache: [BatchKVCache]
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

        cache = (0 ..< rowCaches[0].count).map { layer in
            BatchKVCache.merge(rowCaches.map { $0[layer] })
        }
        currentTokens = concatenated(firstTokens, axis: 0)
        currentLogits = concatenated(firstLogits, axis: 0)
        eval(currentTokens, currentLogits, cache)
    }

    public func next() -> BatchDecodeStep {
        let returnedTokens = currentTokens
        let returnedLogits = currentLogits

        let output = model(
            LMInput.Text(tokens: currentTokens[0..., .newAxis]),
            cache: cache,
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
