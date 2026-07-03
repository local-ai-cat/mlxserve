import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import Tokenizers
import XCTest

private struct PrefixCacheGateResult: Sendable {
    let maxLogitError: Float
    let checkedTokens: Int
    let mismatches: Int
    let sharedBlockCount: Int
    let allocatedBaseline: Int
    let allocatedAfterRequests: Int
    let totalRefBaseline: Int
    let totalRefAfterRequests: Int
}

final class TrackAPrefixCacheTests: XCTestCase {
    func testHotPrefixCacheReconstructsAndDoesNotLeakBlocks() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M3 prefix-cache gate.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M3 prefix-cache fixture is pinned to Qwen3-0.6B-4bit.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let result = try await container.perform { context in
            try await Self.evaluatePrefixCacheGate(context: context)
        }

        if ProcessInfo.processInfo.environment["MLXSERVE_DEBUG_PREFIX_GATE"] == "1" {
            print(
                "M3 prefix: maxLogitError=\(result.maxLogitError), checkedTokens=\(result.checkedTokens), mismatches=\(result.mismatches), sharedBlocks=\(result.sharedBlockCount), allocatedBaseline=\(result.allocatedBaseline), allocatedAfter=\(result.allocatedAfterRequests), totalRefBaseline=\(result.totalRefBaseline), totalRefAfter=\(result.totalRefAfterRequests)"
            )
        }

        XCTAssertLessThan(result.maxLogitError, 1.25)
        XCTAssertGreaterThan(result.checkedTokens, 0)
        XCTAssertEqual(result.mismatches, 0)
        XCTAssertEqual(result.sharedBlockCount, 1)
        XCTAssertEqual(result.allocatedAfterRequests, result.allocatedBaseline)
        XCTAssertEqual(result.totalRefAfterRequests, result.totalRefBaseline)
    }

    private static func evaluatePrefixCacheGate(
        context: ModelContext
    ) async throws -> PrefixCacheGateResult {
        let model = context.model
        let blockSize = 256
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
        let seedText = String(
            repeating: "The capital of France is Paris. Swift concurrency protects shared state. GPU kernels execute matrix operations quickly. ",
            count: 128
        )
        let seedTokens = try await tokenIDs(for: seedText, context: context, parameters: parameters)
        var prefixTokens: [Int] = []
        while prefixTokens.count < blockSize {
            prefixTokens.append(contentsOf: seedTokens)
        }
        prefixTokens = Array(prefixTokens.prefix(blockSize))
        let candidateSuffixes = try await [
            " The answer is",
            " Therefore",
            " In summary",
            " Paris",
            " Swift",
            " The next",
            " GPU",
            " because",
            " once",
            " list",
        ].mapAsync { text in
            try await tokenIDs(for: text, context: context, parameters: parameters)
        }

        let prefix = prefill(model: model, tokens: prefixTokens, parameters: parameters)
        let prefixCache = BlockAwarePrefixCache(modelName: "Qwen3-0.6B-4bit", blockSize: blockSize)
        let storedTable = prefixCache.storeCache(tokens: prefixTokens, cache: prefix.cache)
        XCTAssertEqual(storedTable.count, 1)

        let allocatedBaseline = prefixCache.manager.allocatedBlocks
        let totalRefBaseline = prefixCache.manager.totalRefCount

        var selectedBranches: [(suffix: [Int], logitError: Float, checkedTokens: Int, mismatches: Int)] = []
        for suffix in candidateSuffixes {
            let branch = try compareBranch(
                model: model,
                prefixCache: prefixCache,
                prefixTokens: prefixTokens,
                suffixTokens: suffix,
                parameters: parameters
            )
            if branch.checkedTokens > 0 {
                selectedBranches.append((suffix, branch.logitError, branch.checkedTokens, branch.mismatches))
            }
            if selectedBranches.count == 2 {
                break
            }
        }
        guard selectedBranches.count == 2 else {
            XCTFail("expected at least two wide-margin suffixes for M3 prefix-cache gate")
            return PrefixCacheGateResult(
                maxLogitError: Float.infinity,
                checkedTokens: 0,
                mismatches: 1,
                sharedBlockCount: 0,
                allocatedBaseline: allocatedBaseline,
                allocatedAfterRequests: prefixCache.manager.allocatedBlocks,
                totalRefBaseline: totalRefBaseline,
                totalRefAfterRequests: prefixCache.manager.totalRefCount
            )
        }
        let suffixX = selectedBranches[0].suffix
        let suffixY = selectedBranches[1].suffix

        var maxLogitError = selectedBranches.map(\.logitError).max() ?? 0
        var checkedTokens = selectedBranches.reduce(0) { $0 + $1.checkedTokens }
        var mismatches = selectedBranches.reduce(0) { $0 + $1.mismatches }

        guard let hitX = prefixCache.fetchCache(tokens: prefixTokens + suffixX),
            let hitY = prefixCache.fetchCache(tokens: prefixTokens + suffixY)
        else {
            XCTFail("expected divergent branches to share the stored prefix block")
            return PrefixCacheGateResult(
                maxLogitError: Float.infinity,
                checkedTokens: 0,
                mismatches: 1,
                sharedBlockCount: 0,
                allocatedBaseline: allocatedBaseline,
                allocatedAfterRequests: prefixCache.manager.allocatedBlocks,
                totalRefBaseline: totalRefBaseline,
                totalRefAfterRequests: prefixCache.manager.totalRefCount
            )
        }
        let sharedBlockCount = zip(hitX.table.blockIDs, hitY.table.blockIDs)
            .filter { $0 == $1 }
            .count
        prefixCache.release(hitX)
        prefixCache.release(hitY)

        for index in 0 ..< 8 {
            let suffix = index.isMultiple(of: 2) ? suffixX : suffixY
            let branch = try compareBranch(
                model: model,
                prefixCache: prefixCache,
                prefixTokens: prefixTokens,
                suffixTokens: suffix,
                parameters: parameters
            )
            maxLogitError = max(maxLogitError, branch.logitError)
            checkedTokens += branch.checkedTokens
            mismatches += branch.mismatches
        }

        return PrefixCacheGateResult(
            maxLogitError: maxLogitError,
            checkedTokens: checkedTokens,
            mismatches: mismatches,
            sharedBlockCount: sharedBlockCount,
            allocatedBaseline: allocatedBaseline,
            allocatedAfterRequests: prefixCache.manager.allocatedBlocks,
            totalRefBaseline: totalRefBaseline,
            totalRefAfterRequests: prefixCache.manager.totalRefCount
        )
    }

    private static func compareBranch(
        model: any LanguageModel,
        prefixCache: BlockAwarePrefixCache,
        prefixTokens: [Int],
        suffixTokens: [Int],
        parameters: GenerateParameters
    ) throws -> (logitError: Float, checkedTokens: Int, mismatches: Int) {
        guard let hit = prefixCache.fetchCache(tokens: prefixTokens + suffixTokens) else {
            throw PrefixCacheTestError.missingPrefixHit
        }
        defer { prefixCache.release(hit) }

        let reconstructedCache = prefixCache.reconstructCache(from: hit)
        let reconstructedOutput = model(
            LMInput.Text(tokens: MLXArray(suffixTokens.map(Int32.init)))[text: .newAxis],
            cache: reconstructedCache,
            state: nil
        )
        let reconstructedLogits = reconstructedOutput.logits[0..., -1, 0...]
        eval(reconstructedLogits, reconstructedCache)

        let fresh = prefill(
            model: model,
            tokens: prefixTokens + suffixTokens,
            parameters: parameters
        )

        let logitError = maxAbsoluteDifference(reconstructedLogits, fresh.logits)
        let margin = topOneTopTwoMargin(fresh.logits)
        let reconstructedToken = argMax(reconstructedLogits, axis: -1).item(Int.self)
        let freshToken = argMax(fresh.logits, axis: -1).item(Int.self)

        if margin > logitError * 4 + 1e-3 {
            return (logitError, 1, reconstructedToken == freshToken ? 0 : 1)
        }
        return (logitError, 0, 0)
    }

    private static func prefill(
        model: any LanguageModel,
        tokens: [Int],
        parameters: GenerateParameters
    ) -> (cache: [any KVCache], logits: MLXArray) {
        let cache = model.newCache(parameters: parameters)
        let input = LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)))
        let output = model(input[text: .newAxis], cache: cache, state: nil)
        let logits = output.logits[0..., -1, 0...]
        eval(logits, cache)
        return (cache, logits)
    }

    private static func maxAbsoluteDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
    }

    private static func topOneTopTwoMargin(_ logits: MLXArray) -> Float {
        let topValues = top(logits.asType(.float32), k: 2, axis: -1).asArray(Float.self)
        let sorted = topValues.sorted(by: >)
        return sorted[0] - sorted[1]
    }

    private static func tokenIDs(
        for text: String,
        context: ModelContext,
        parameters: GenerateParameters
    ) async throws -> [Int] {
        let input = try await context.processor.prepare(input: UserInput(prompt: text))
        let cache = context.model.newCache(parameters: parameters)
        switch try context.model.prepare(input, cache: cache, windowSize: parameters.prefillStepSize) {
        case .tokens(let tokens):
            return tokens.tokens.asArray(Int.self)
        case .logits:
            throw PrefixCacheTestError.missingPrefixHit
        }
    }
}

private enum PrefixCacheTestError: Error {
    case missingPrefixHit
}

private extension Array {
    func mapAsync<T>(_ transform: (Element) async throws -> T) async throws -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        for element in self {
            result.append(try await transform(element))
        }
        return result
    }
}
