import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import Tokenizers
import XCTest

private struct DecodeTraceStep {
    let tokenId: Int
    let logits: MLXArray
}

private struct BatchGateResult: Sendable {
    let batchSize: Int
    let maxLogitError: Float
    let maxErrorStep: Int
    let maxErrorRow: Int
    let maxErrorPrompt: String
    let maxErrorMargin: Float
    let maxErrorSerialToken: Int
    let maxErrorBatchToken: Int
    let checkedTokenCount: Int
    let mismatchedCheckedTokens: Int
}

final class BatchInvarianceTests: XCTestCase {
    private static let prompts = [
        "The capital of France is",
        "Write a haiku about GPU kernels.",
        "In Swift, an actor protects",
        "List three colors:",
        "What is 2 + 2?",
        "Translate hello to Spanish:",
        "The largest planet is",
        "Complete: once upon a",
    ]

    func testStaticBatchedDecodeMatchesSerialWithinMarginGate() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M1 batch-invariance.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M1 batch-invariance fixture is pinned to Qwen3-0.6B-4bit.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let prompts = Self.prompts
        let results = try await container.perform { context in
            try await Self.evaluateBatchGate(context: context, prompts: prompts)
        }

        for result in results {
            if ProcessInfo.processInfo.environment["MLXSERVE_DEBUG_BATCH_GATE"] == "1" {
                print(
                    "M1 batch \(result.batchSize): maxLogitError=\(result.maxLogitError), checkedTokens=\(result.checkedTokenCount), mismatches=\(result.mismatchedCheckedTokens), maxErrorStep=\(result.maxErrorStep), maxErrorRow=\(result.maxErrorRow)"
                )
            }
            XCTAssertLessThan(
                result.maxLogitError,
                1.25,
                "batch size \(result.batchSize) exceeded logit tolerance at step \(result.maxErrorStep), row \(result.maxErrorRow), prompt '\(result.maxErrorPrompt)', margin \(result.maxErrorMargin), serial token \(result.maxErrorSerialToken), batch token \(result.maxErrorBatchToken)"
            )
            XCTAssertGreaterThan(
                result.checkedTokenCount,
                0,
                "batch size \(result.batchSize) had no wide-margin token checks"
            )
            XCTAssertEqual(
                result.mismatchedCheckedTokens,
                0,
                "batch size \(result.batchSize) mismatched wide-margin tokens"
            )
        }
    }

    private static func evaluateBatchGate(
        context: ModelContext,
        prompts: [String]
    ) async throws -> [BatchGateResult] {
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        var results: [BatchGateResult] = []

        for batchSize in [2, 4, 8] {
            var inputs: [LMInput] = []
            for prompt in prompts.prefix(batchSize) {
                inputs.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
            }

            let serial = try inputs.map {
                try Self.serialTrace(
                    model: context.model,
                    input: $0,
                    parameters: parameters,
                    steps: parameters.maxTokens ?? 4
                )
            }
            let batchGenerator = try StaticBatchGenerator(
                model: context.model,
                inputs: inputs,
                parameters: parameters
            )

            var maxLogitError: Float = 0
            var maxErrorStep = -1
            var maxErrorRow = -1
            var maxErrorPrompt = ""
            var maxErrorMargin: Float = 0
            var maxErrorSerialToken = -1
            var maxErrorBatchToken = -1
            var checkedTokenCount = 0
            var mismatchedCheckedTokens = 0

            for stepIndex in 0 ..< (parameters.maxTokens ?? 4) {
                let batchStep = batchGenerator.next()
                if stepIndex == 0 {
                    continue
                }
                for row in 0 ..< batchSize {
                    let serialStep = serial[row][stepIndex]
                    let batchLogits = batchStep.logits[row, 0...]
                    let logitError = Self.maxAbsoluteDifference(batchLogits, serialStep.logits)
                    let margin = Self.topOneTopTwoMargin(serialStep.logits)
                    if logitError > maxLogitError {
                        maxLogitError = logitError
                        maxErrorStep = stepIndex
                        maxErrorRow = row
                        maxErrorPrompt = prompts[row]
                        maxErrorMargin = margin
                        maxErrorSerialToken = serialStep.tokenId
                        maxErrorBatchToken = batchStep.tokenIds[row]
                    }

                    if margin > logitError * 4 + 1e-3 {
                        checkedTokenCount += 1
                        if batchStep.tokenIds[row] != serialStep.tokenId {
                            mismatchedCheckedTokens += 1
                        }
                    }
                }
            }

            results.append(
                BatchGateResult(
                    batchSize: batchSize,
                    maxLogitError: maxLogitError,
                    maxErrorStep: maxErrorStep,
                    maxErrorRow: maxErrorRow,
                    maxErrorPrompt: maxErrorPrompt,
                    maxErrorMargin: maxErrorMargin,
                    maxErrorSerialToken: maxErrorSerialToken,
                    maxErrorBatchToken: maxErrorBatchToken,
                    checkedTokenCount: checkedTokenCount,
                    mismatchedCheckedTokens: mismatchedCheckedTokens
                )
            )
        }

        return results
    }

    private static func serialTrace(
        model: any LanguageModel,
        input: LMInput,
        parameters: GenerateParameters,
        steps: Int
    ) throws -> [DecodeTraceStep] {
        let cache = model.newCache(parameters: parameters)
        var state: LMOutput.State?

        var currentLogits: MLXArray
        var currentToken: MLXArray

        switch try model.prepare(input, cache: cache, windowSize: parameters.prefillStepSize) {
        case .tokens(let tokens):
            let output = model(tokens[text: .newAxis], cache: cache, state: state)
            state = output.state
            currentLogits = output.logits[0..., -1, 0...]
            currentToken = argMax(currentLogits, axis: -1)
        case .logits(let output):
            state = output.state
            currentLogits = output.logits[0..., -1, 0...]
            currentToken = argMax(currentLogits, axis: -1)
        }
        eval(currentToken, currentLogits, cache)

        var trace: [DecodeTraceStep] = []
        for _ in 0 ..< steps {
            eval(currentToken, currentLogits)
            trace.append(
                DecodeTraceStep(
                    tokenId: currentToken.item(Int.self),
                    logits: currentLogits
                )
            )

            let output = model(
                LMInput.Text(tokens: currentToken[.newAxis, 0...]),
                cache: cache,
                state: state
            )
            state = output.state
            currentLogits = output.logits[0..., -1, 0...]
            currentToken = argMax(currentLogits, axis: -1)
            asyncEval(currentToken, currentLogits)
        }

        eval(currentToken, currentLogits)
        return trace
    }

    private static func maxAbsoluteDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let difference = abs(lhs.asType(.float32) - rhs.asType(.float32))
        return difference.max().item(Float.self)
    }

    private static func topOneTopTwoMargin(_ logits: MLXArray) -> Float {
        let topValues = top(logits.asType(.float32), k: 2, axis: -1).asArray(Float.self)
        let sorted = topValues.sorted(by: >)
        return sorted[0] - sorted[1]
    }
}
