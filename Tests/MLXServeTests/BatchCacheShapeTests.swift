import MLX
import MLXLMCommon
@testable import MLXServe
import XCTest

final class BatchCacheShapeTests: XCTestCase {
    func testSequenceMergeUsesInferredAxisForRankThreeKV() throws {
        let short = KVCacheSimple()
        short.state = [
            MLXArray.zeros([1, 1, 8], dtype: .float32),
            MLXArray.zeros([1, 1, 8], dtype: .float32),
        ]

        let long = KVCacheSimple()
        long.state = [
            MLXArray.zeros([1, 3, 8], dtype: .float32),
            MLXArray.zeros([1, 3, 8], dtype: .float32),
        ]

        let batch = try BatchKVCache.merge([short, long])

        XCTAssertEqual(batch.state[0].shape, [2, 3, 8])
        XCTAssertEqual(batch.state[1].shape, [2, 3, 8])
        XCTAssertEqual(batch.extract(0).state[0].shape, [1, 1, 8])
        XCTAssertEqual(batch.extract(1).state[0].shape, [1, 3, 8])

        switch (batch as any KVCache).ropeOffset {
        case .batch(let offset):
            XCTAssertEqual(offset.shape, [2])
        default:
            XCTFail("BatchKVCache must dispatch per-row RoPE offsets")
        }
    }

    func testFixedStateMergePreservesConcreteCacheType() throws {
        let first = FixedStateCache(
            state: [
                MLXArray.zeros([1, 3, 16], dtype: .bfloat16),
                MLXArray.zeros([1, 4, 8, 8], dtype: .float32),
            ]
        )
        let second = FixedStateCache(
            state: [
                MLXArray.zeros([1, 3, 16], dtype: .bfloat16),
                MLXArray.zeros([1, 4, 8, 8], dtype: .float32),
            ]
        )

        let batch = try BatchLayerCache.merge([first, second])

        XCTAssertTrue(batch.kvCache is FixedStateCache)
        XCTAssertEqual(batch.kvCache.state[0].shape, [2, 3, 16])
        XCTAssertEqual(batch.kvCache.state[1].shape, [2, 4, 8, 8])

        batch.filter(keeping: [1])

        XCTAssertTrue(batch.kvCache is FixedStateCache)
        XCTAssertEqual(batch.kvCache.state[0].shape, [1, 3, 16])
        XCTAssertEqual(batch.kvCache.state[1].shape, [1, 4, 8, 8])

        let extracted = batch.extract(0)
        XCTAssertEqual(extracted.state[0].shape, [1, 3, 16])
        XCTAssertEqual(extracted.state[1].shape, [1, 4, 8, 8])
    }

    func testMergeRejectsKnownRotatingCacheTypes() throws {
        let rotating = RotatingTestCache()
        rotating.state = [
            MLXArray.zeros([1, 1, 2, 8], dtype: .float32),
            MLXArray.zeros([1, 1, 2, 8], dtype: .float32),
        ]

        XCTAssertThrowsError(try BatchKVCache.merge([rotating])) { error in
            XCTAssertEqual(
                error as? BatchKVCacheError,
                .rotatingCacheUnsupported(cacheType: "RotatingTestCache")
            )
        }
    }
}

private final class FixedStateCache: KVCache {
    var state: [MLXArray]
    var metaState: [String]
    var offset: Int { 0 }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    init(state: [MLXArray], metaState: [String] = ["2", "0,1"]) {
        self.state = state
        self.metaState = metaState
    }

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
        FixedStateCache(state: state.map { $0[.ellipsis] }, metaState: metaState)
    }
}

private final class RotatingTestCache: KVCache {
    var state: [MLXArray] = []
    var metaState: [String] = []
    var offset: Int { 2 }
    var maxSize: Int? { 4 }
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
        let copy = RotatingTestCache()
        copy.state = state.map { $0[.ellipsis] }
        copy.metaState = metaState
        return copy
    }
}
