import MLX
import MLXLMCommon

final class BatchLayerCache {
    var kvCache: any KVCache

    private enum Layout: Equatable {
        case sequence
        case batchState
    }

    private let layout: Layout
    private var rowMetaStates: [[String]]

    private init(kvCache: any KVCache, layout: Layout, rowMetaStates: [[String]]) {
        self.kvCache = kvCache
        self.layout = layout
        self.rowMetaStates = rowMetaStates
    }

    var batchSize: Int {
        kvCache.state.first?.dim(0) ?? 0
    }

    static func merge(_ caches: [any KVCache]) throws -> BatchLayerCache {
        guard let firstCache = caches.first else {
            throw BatchKVCacheError.emptyMerge
        }
        let firstState = firstCache.state
        guard !firstState.isEmpty else {
            throw BatchKVCacheError.emptyState(cacheType: String(describing: type(of: firstCache)))
        }

        if isSequenceState(firstState) {
            let batchCache = try BatchKVCache.merge(caches)
            return BatchLayerCache(
                kvCache: batchCache,
                layout: .sequence,
                rowMetaStates: caches.map(\.metaState)
            )
        }

        return try mergeBatchState(caches)
    }

    func insert(_ cache: any KVCache) throws {
        try extend(Self.merge([cache]))
    }

    func extend(_ other: BatchLayerCache) throws {
        guard layout == other.layout else {
            throw BatchKVCacheError.incompatibleLayout(
                expected: String(describing: layout),
                actual: String(describing: other.layout)
            )
        }

        switch layout {
        case .sequence:
            guard let cache = kvCache as? BatchKVCache,
                let otherCache = other.kvCache as? BatchKVCache
            else {
                throw BatchKVCacheError.incompatibleLayout(expected: "BatchKVCache", actual: String(describing: type(of: other.kvCache)))
            }
            try cache.extend(otherCache)
        case .batchState:
            let currentState = kvCache.state
            let otherState = other.kvCache.state
            guard currentState.count == otherState.count else {
                throw BatchKVCacheError.inconsistentStateCount(
                    expected: currentState.count,
                    actual: otherState.count,
                    cacheType: String(describing: type(of: other.kvCache))
                )
            }
            kvCache.state = zip(currentState, otherState).map { concatenated([$0, $1], axis: 0) }
        }

        rowMetaStates.append(contentsOf: other.rowMetaStates)
    }

    func filter(keeping rows: [Int]) {
        guard !rows.isEmpty else {
            kvCache.state = []
            rowMetaStates.removeAll()
            return
        }

        switch layout {
        case .sequence:
            (kvCache as? BatchKVCache)?.filter(keeping: rows)
        case .batchState:
            let rowIndices = MLXArray(rows.map(Int32.init))
            kvCache.state = kvCache.state.map { $0.take(rowIndices, axis: 0) }
        }
        rowMetaStates = rows.compactMap { row in
            row < rowMetaStates.count ? rowMetaStates[row] : nil
        }
    }

    func extract(_ row: Int) -> KVCacheSimple {
        if layout == .sequence, let cache = kvCache as? BatchKVCache {
            return cache.extract(row)
        }

        let cache = KVCacheSimple()
        guard row >= 0, row < batchSize else { return cache }
        cache.state = kvCache.state.map { Self.sliceRow($0, row: row) }
        if row < rowMetaStates.count {
            cache.metaState = rowMetaStates[row]
        }
        return cache
    }

    private static func mergeBatchState(_ caches: [any KVCache]) throws -> BatchLayerCache {
        guard let firstCache = caches.first else {
            throw BatchKVCacheError.emptyMerge
        }
        let states = caches.map(\.state)
        let cacheTypes = caches.map { String(describing: type(of: $0)) }
        let firstState = states[0]

        for (state, cacheType) in zip(states, cacheTypes) where state.count != firstState.count {
            throw BatchKVCacheError.inconsistentStateCount(
                expected: firstState.count,
                actual: state.count,
                cacheType: cacheType
            )
        }

        var batchState: [MLXArray] = []
        for stateIndex in firstState.indices {
            let firstShape = firstState[stateIndex].shape
            for row in states.indices {
                try validateBatchStateShape(
                    cacheType: cacheTypes[row],
                    stateIndex: stateIndex,
                    expected: firstShape,
                    actual: states[row][stateIndex].shape
                )
            }
            batchState.append(concatenated(states.map { $0[stateIndex] }, axis: 0))
        }

        var cache = firstCache.copy()
        cache.state = batchState
        return BatchLayerCache(
            kvCache: cache,
            layout: .batchState,
            rowMetaStates: caches.map(\.metaState)
        )
    }

    private static func isSequenceState(_ state: [MLXArray]) -> Bool {
        guard state.count == 2 else { return false }
        let ranks = state.map { $0.shape.count }
        return ranks[0] == ranks[1] && ranks[0] >= 3
    }

    private static func validateBatchStateShape(
        cacheType: String,
        stateIndex: Int,
        expected: [Int],
        actual: [Int]
    ) throws {
        guard expected.count == actual.count else {
            throw BatchKVCacheError.incompatibleStateShape(
                cacheType: cacheType,
                stateIndex: stateIndex,
                expected: expected,
                actual: actual
            )
        }
        for axis in expected.indices where axis != 0 && expected[axis] != actual[axis] {
            throw BatchKVCacheError.incompatibleStateShape(
                cacheType: cacheType,
                stateIndex: stateIndex,
                expected: expected,
                actual: actual
            )
        }
    }

    private static func sliceRow(_ array: MLXArray, row: Int) -> MLXArray {
        var indices: [any MLXArrayIndex] = Array(repeating: 0..., count: array.shape.count)
        indices[0] = row ..< row + 1
        return array[indices]
    }
}
