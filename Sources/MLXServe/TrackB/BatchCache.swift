import MLX
import MLXLMCommon

public final class BatchKVCache: KVCache, BatchPositionedKVCache {
    public private(set) var leftPadding: MLXArray
    public private(set) var idx: Int
    public var offset: Int { idx }
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { true }

    private var keys: MLXArray?
    private var values: MLXArray?
    private let step: Int

    public var batchSize: Int {
        leftPadding.dim(0)
    }

    public init(leftPadding: [Int], idx: Int = 0, step: Int = 256) {
        self.leftPadding = MLXArray(leftPadding.map(Int32.init))
        self.idx = idx
        self.step = step
    }

    public var batchOffset: MLXArray {
        MLXArray(Int32(idx)) - leftPadding
    }

    public func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public static func merge(_ caches: [any KVCache]) -> BatchKVCache {
        precondition(!caches.isEmpty, "BatchKVCache.merge requires at least one cache")

        let states = caches.map(\.state)
        let lengths = states.map { state in
            precondition(state.count == 2, "BatchKVCache.merge supports KVCache state only")
            return state[0].dim(2)
        }
        let maxLength = lengths.max() ?? 0
        let padding = lengths.map { maxLength - $0 }
        let firstState = states[0]

        let batchSize = caches.count
        let keyShape = firstState[0].shape
        let valueShape = firstState[1].shape
        let batchKeys = MLXArray.zeros(
            [batchSize, keyShape[1], maxLength, keyShape[3]],
            dtype: firstState[0].dtype
        )
        let batchValues = MLXArray.zeros(
            [batchSize, valueShape[1], maxLength, valueShape[3]],
            dtype: firstState[1].dtype
        )

        for (row, state) in states.enumerated() {
            let length = lengths[row]
            let start = padding[row]
            batchKeys[row ..< row + 1, 0..., start ..< start + length, 0...] = state[0]
            batchValues[row ..< row + 1, 0..., start ..< start + length, 0...] = state[1]
        }

        let cache = BatchKVCache(leftPadding: padding, idx: maxLength)
        cache.keys = batchKeys
        cache.values = batchValues
        return cache
    }

    public func extract(_ row: Int) -> KVCacheSimple {
        let padding = leftPadding.asArray(Int.self)[row]
        let cache = KVCacheSimple()
        guard let keys, let values else { return cache }
        cache.state = [
            keys[row ..< row + 1, 0..., padding ..< idx, 0...],
            values[row ..< row + 1, 0..., padding ..< idx, 0...],
        ]
        return cache
    }

    public func filter(keeping rows: [Int]) {
        guard !rows.isEmpty else {
            keys = nil
            values = nil
            leftPadding = MLXArray([Int32]())
            idx = 0
            return
        }

        let rowIndices = MLXArray(rows.map(Int32.init))
        keys = keys?.take(rowIndices, axis: 0)
        values = values?.take(rowIndices, axis: 0)
        leftPadding = leftPadding.take(rowIndices, axis: 0)

        let minLeftPadding = leftPadding.min().item(Int.self)
        guard minLeftPadding > 0 else { return }

        if let currentKeys = keys, let currentValues = values {
            keys = currentKeys[0..., 0..., minLeftPadding ..< idx, 0...]
            values = currentValues[0..., 0..., minLeftPadding ..< idx, 0...]
        }
        idx -= minLeftPadding
        leftPadding = leftPadding - MLXArray(Int32(minLeftPadding))
    }

    public func insert(_ cache: any KVCache) {
        extend([cache])
    }

    public func extend(_ caches: [any KVCache]) {
        guard !caches.isEmpty else { return }
        extend(BatchKVCache.merge(caches))
    }

    public func extend(_ other: BatchKVCache) {
        guard other.batchSize > 0 else { return }
        guard batchSize > 0 else {
            state = other.state
            leftPadding = other.leftPadding
            idx = other.idx
            return
        }
        guard let currentKeys = keys, let currentValues = values else {
            state = other.state
            leftPadding = other.leftPadding
            idx = other.idx
            return
        }
        guard let otherKeys = other.keys, let otherValues = other.values else { return }

        let targetIdx = max(idx, other.idx)
        let normalizedCurrent = normalize(
            keys: currentKeys[0..., 0..., ..<idx, 0...],
            values: currentValues[0..., 0..., ..<idx, 0...],
            leftPadding: leftPadding,
            from: idx,
            to: targetIdx
        )
        let normalizedOther = normalize(
            keys: otherKeys[0..., 0..., ..<other.idx, 0...],
            values: otherValues[0..., 0..., ..<other.idx, 0...],
            leftPadding: other.leftPadding,
            from: other.idx,
            to: targetIdx
        )

        keys = concatenated([normalizedCurrent.keys, normalizedOther.keys], axis: 0)
        values = concatenated([normalizedCurrent.values, normalizedOther.values], axis: 0)
        leftPadding = concatenated([normalizedCurrent.leftPadding, normalizedOther.leftPadding], axis: 0)
        idx = targetIdx
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let previous = idx
        ensureCapacity(keys: newKeys, values: newValues)
        idx += newKeys.dim(2)

        keys![0..., 0..., previous ..< idx, 0...] = newKeys
        values![0..., 0..., previous ..< idx, 0...] = newValues

        return (
            keys![0..., 0..., ..<idx, 0...],
            values![0..., 0..., ..<idx, 0...]
        )
    }

    public var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            return [
                keys[0..., 0..., ..<idx, 0...],
                values[0..., 0..., ..<idx, 0...],
            ]
        }
        set {
            precondition(newValue.count == 2, "BatchKVCache state must contain keys and values")
            keys = newValue[0]
            values = newValue[1]
            idx = newValue[0].dim(2)
        }
    }

    public var metaState: [String] {
        get {
            [
                String(idx),
                leftPadding.asArray(Int.self).map(String.init).joined(separator: ","),
            ]
        }
        set {
            precondition(newValue.count == 2, "BatchKVCache metaState must contain idx and leftPadding")
            idx = Int(newValue[0]) ?? 0
            let padding = newValue[1].split(separator: ",").compactMap { Int($0) }
            leftPadding = MLXArray(padding.map(Int32.init))
        }
    }

    public func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard let mask = CausalMask.create(
            n: n,
            offset: idx,
            leftPadding: leftPadding,
            windowSize: windowSize
        ) else {
            return .none
        }
        var additiveMask = MLX.where(mask, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        if let keys {
            additiveMask = additiveMask.asType(keys.dtype)
        }
        return .array(additiveMask)
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = min(idx, n)
        idx -= trimmed
        return trimmed
    }

    public func copy() -> any KVCache {
        let copy = BatchKVCache(leftPadding: leftPadding.asArray(Int.self), idx: idx, step: step)
        let currentState = state
        if !currentState.isEmpty {
            copy.state = currentState.map { $0[.ellipsis] }
        }
        return copy
    }

    private func ensureCapacity(keys newKeys: MLXArray, values newValues: MLXArray) {
        guard let currentKeys = keys, let currentValues = values else {
            let nSteps = (step + newKeys.dim(2) - 1) / step
            keys = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), nSteps * step, newKeys.dim(3)],
                dtype: newKeys.dtype
            )
            values = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), nSteps * step, newValues.dim(3)],
                dtype: newValues.dtype
            )
            return
        }

        guard idx + newKeys.dim(2) > currentKeys.dim(2) else { return }

        let nSteps = (step + newKeys.dim(2) - 1) / step
        let keyExtension = MLXArray.zeros(
            [newKeys.dim(0), newKeys.dim(1), nSteps * step, newKeys.dim(3)],
            dtype: newKeys.dtype
        )
        let valueExtension = MLXArray.zeros(
            [newValues.dim(0), newValues.dim(1), nSteps * step, newValues.dim(3)],
            dtype: newValues.dtype
        )
        keys = concatenated([currentKeys, keyExtension], axis: 2)
        values = concatenated([currentValues, valueExtension], axis: 2)
    }

    private func normalize(
        keys inputKeys: MLXArray,
        values inputValues: MLXArray,
        leftPadding inputLeftPadding: MLXArray,
        from currentIdx: Int,
        to targetIdx: Int
    ) -> (keys: MLXArray, values: MLXArray, leftPadding: MLXArray) {
        let delta = targetIdx - currentIdx
        guard delta > 0 else {
            return (inputKeys, inputValues, inputLeftPadding)
        }

        let keyPadding = MLXArray.zeros(
            [inputKeys.dim(0), inputKeys.dim(1), delta, inputKeys.dim(3)],
            dtype: inputKeys.dtype
        )
        let valuePadding = MLXArray.zeros(
            [inputValues.dim(0), inputValues.dim(1), delta, inputValues.dim(3)],
            dtype: inputValues.dtype
        )
        return (
            concatenated([keyPadding, inputKeys], axis: 2),
            concatenated([valuePadding, inputValues], axis: 2),
            inputLeftPadding + MLXArray(Int32(delta))
        )
    }
}
