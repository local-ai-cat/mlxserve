import MLX
import MLXLMCommon
import MLXNN
import MLXServe
@testable import MLXServeNative
import XCTest

final class ModelCacheCapabilitiesTests: XCTestCase {
    func testExplicitWindowedKVCapabilityDisablesPrefixCache() async throws {
        let store = SessionPrefixKVStore()
        let engine = MLXServeEngine(
            model: FixedLogitPrefixModel(cacheFactory: Self.simpleCache),
            parameters: GenerateParameters(maxTokens: 1, temperature: 0),
            maxConcurrentRequests: 1,
            prefixStore: store,
            cacheCapabilities: ModelCacheCapabilities(usesWindowedKVCache: true)
        )

        _ = try await engine.generate([Self.request(uid: "windowed")])

        XCTAssertEqual(store.stats.storeCount, 0)
    }

    func testPrefixCacheDoesNotDisableFromCacheTypeName() async throws {
        let store = SessionPrefixKVStore()
        let engine = MLXServeEngine(
            model: FixedLogitPrefixModel(cacheFactory: { RotatingNameOnlyCache() }),
            parameters: GenerateParameters(maxTokens: 1, temperature: 0),
            maxConcurrentRequests: 1,
            prefixStore: store
        )

        _ = try await engine.generate([Self.request(uid: "name-only")])

        XCTAssertEqual(store.stats.storeCount, 1)
    }

    func testNativeLoaderDerivesWindowedKVCapabilityFromModelConfig() throws {
        let loader = NativeModelLoader(maxConcurrentRequests: 1)

        XCTAssertTrue(
            try loader.cacheCapabilities(in: Self.modelDirectory(configJSON: #"{"model_type":"gemma4"}"#))
                .usesWindowedKVCache
        )
        XCTAssertTrue(
            try loader.cacheCapabilities(in: Self.modelDirectory(configJSON: #"{"model_type":"gpt_oss","sliding_window":128}"#))
                .usesWindowedKVCache
        )
        XCTAssertFalse(
            try loader.cacheCapabilities(
                in: Self.modelDirectory(
                    configJSON: #"{"model_type":"qwen3","sliding_window":4096,"use_sliding_window":false}"#
                )
            ).usesWindowedKVCache
        )
    }

    private static func request(uid: String) -> Request {
        Request(
            uid: uid,
            input: LMInput(text: LMInput.Text(tokens: MLXArray([Int32(1), 2, 3]))),
            maxTokens: 1,
            sampling: SamplingParameters(temperature: 0)
        )
    }

    private static func simpleCache() -> KVCacheSimple {
        let cache = KVCacheSimple()
        cache.state = [
            MLXArray.zeros([1, 1, 1, 1]),
            MLXArray.zeros([1, 1, 1, 1]),
        ]
        return cache
    }

    private static func modelDirectory(configJSON: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "mlxserve-model-capabilities-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(configJSON.utf8).write(to: directory.appending(component: "config.json"))
        return directory
    }
}

private final class FixedLogitPrefixModel: Module, LanguageModel {
    private let cacheFactory: () -> any KVCache

    init(cacheFactory: @escaping () -> any KVCache) {
        self.cacheFactory = cacheFactory
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        LMOutput(logits: MLXArray.zeros([input.tokens.dim(0), 1, 8]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [cacheFactory()]
    }
}

private final class RotatingNameOnlyCache: KVCache {
    var state: [MLXArray] = [
        MLXArray.zeros([1, 1, 1, 1]),
        MLXArray.zeros([1, 1, 1, 1]),
    ]
    var metaState: [String] = []
    var offset: Int { 0 }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    func innerState() -> [MLXArray] {
        state
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        state = [newKeys, newValues]
        return (newKeys, newValues)
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        0
    }

    func copy() -> any KVCache {
        let copy = RotatingNameOnlyCache()
        copy.state = state.map { $0[.ellipsis] }
        copy.metaState = metaState
        return copy
    }
}
