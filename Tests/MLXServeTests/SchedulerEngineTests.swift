import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import Tokenizers
import XCTest

private struct EngineTraceStep {
    let tokenId: Int
    let logits: MLXArray
}

private struct EngineGateResult: Sendable {
    let batchCheckedTokens: Int
    let batchMismatches: Int
    let cancelCheckedTokens: Int
    let cancelMismatches: Int
    let cancelledResponses: Int
    let queueFullRejected: Bool
}

final class SchedulerEngineTests: XCTestCase {
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

    func testEngineConcurrentGenerationCancellationAndBackpressure() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M2 engine gate.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M2 engine fixture is pinned to Qwen3-0.6B-4bit.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let prompts = Self.prompts
        let result = try await container.perform { context in
            try await Self.evaluateEngineGate(context: context, prompts: prompts)
        }

        if ProcessInfo.processInfo.environment["MLXSERVE_DEBUG_ENGINE_GATE"] == "1" {
            print(
                "M2 engine: batchChecked=\(result.batchCheckedTokens), batchMismatches=\(result.batchMismatches), cancelChecked=\(result.cancelCheckedTokens), cancelMismatches=\(result.cancelMismatches), cancelledResponses=\(result.cancelledResponses), queueFullRejected=\(result.queueFullRejected)"
            )
        }

        XCTAssertGreaterThan(result.batchCheckedTokens, 0)
        XCTAssertEqual(result.batchMismatches, 0)
        XCTAssertGreaterThan(result.cancelCheckedTokens, 0)
        XCTAssertEqual(result.cancelMismatches, 0)
        XCTAssertEqual(result.cancelledResponses, 1)
        XCTAssertTrue(result.queueFullRejected)
    }

    private static func evaluateEngineGate(
        context: ModelContext,
        prompts: [String]
    ) async throws -> EngineGateResult {
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        var inputs: [LMInput] = []
        for prompt in prompts {
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

        let engine = MLXServeEngine(
            model: context.model,
            parameters: parameters,
            maxConcurrentRequests: 8
        )

        var batchCheckedTokens = 0
        var batchMismatches = 0
        for batchSize in [1, 4, 8] {
            let requests = (0 ..< batchSize).map { row in
                Request(
                    uid: "batch-\(batchSize)-\(row)",
                    input: inputs[row],
                    maxTokens: parameters.maxTokens ?? 4,
                    sampling: SamplingParameters(temperature: 0)
                )
            }
            let outputs = try await engine.generate(requests)
            for row in 0 ..< batchSize {
                let uid = "batch-\(batchSize)-\(row)"
                let comparison = compareWideMarginTokens(
                    generated: outputs[uid, default: []],
                    serial: serial[row]
                )
                batchCheckedTokens += comparison.checked
                batchMismatches += comparison.mismatches
            }
        }

        let cancelRows = [0, 1, 2]
        for row in cancelRows {
            try await engine.submit(
                Request(
                    uid: "cancel-\(row)",
                    input: inputs[row],
                    maxTokens: parameters.maxTokens ?? 4,
                    sampling: SamplingParameters(temperature: 0)
                )
            )
        }

        _ = try await engine.step()
        await engine.cancel(uid: "cancel-1")
        while await !engine.isIdle {
            _ = try await engine.step()
        }

        var cancelCheckedTokens = 0
        var cancelMismatches = 0
        for row in [0, 2] {
            let comparison = await compareWideMarginTokens(
                generated: engine.tokens(for: "cancel-\(row)"),
                serial: serial[row]
            )
            cancelCheckedTokens += comparison.checked
            cancelMismatches += comparison.mismatches
        }
        let cancelledResponses = await engine.responses(for: "cancel-1")
            .filter { $0.finishReason == .cancelled }
            .count

        var queueFullRejected = false
        for index in 0 ..< 32 {
            try await engine.submit(
                Request(
                    uid: "queue-\(index)",
                    input: inputs[0],
                    maxTokens: 1,
                    sampling: SamplingParameters(temperature: 0)
                )
            )
        }
        do {
            try await engine.submit(
                Request(
                    uid: "queue-overflow",
                    input: inputs[0],
                    maxTokens: 1,
                    sampling: SamplingParameters(temperature: 0)
                )
            )
        } catch SchedulerError.queueFull {
            queueFullRejected = true
        }

        return EngineGateResult(
            batchCheckedTokens: batchCheckedTokens,
            batchMismatches: batchMismatches,
            cancelCheckedTokens: cancelCheckedTokens,
            cancelMismatches: cancelMismatches,
            cancelledResponses: cancelledResponses,
            queueFullRejected: queueFullRejected
        )
    }

    private static func compareWideMarginTokens(
        generated: [Int],
        serial: [EngineTraceStep]
    ) -> (checked: Int, mismatches: Int) {
        var checked = 0
        var mismatches = 0
        for (step, token) in generated.enumerated() where step < serial.count {
            let serialStep = serial[step]
            let margin = topOneTopTwoMargin(serialStep.logits)
            if margin > 1.25 * 4 + 1e-3 {
                checked += 1
                if token != serialStep.tokenId {
                    mismatches += 1
                }
            }
        }
        return (checked, mismatches)
    }

    private static func serialTrace(
        model: any LanguageModel,
        input: LMInput,
        parameters: GenerateParameters,
        steps: Int
    ) throws -> [EngineTraceStep] {
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

        var trace: [EngineTraceStep] = []
        for _ in 0 ..< steps {
            eval(currentToken, currentLogits)
            trace.append(EngineTraceStep(tokenId: currentToken.item(Int.self), logits: currentLogits))

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

    private static func topOneTopTwoMargin(_ logits: MLXArray) -> Float {
        let topValues = top(logits.asType(.float32), k: 2, axis: -1).asArray(Float.self)
        let sorted = topValues.sorted(by: >)
        return sorted[0] - sorted[1]
    }
}
