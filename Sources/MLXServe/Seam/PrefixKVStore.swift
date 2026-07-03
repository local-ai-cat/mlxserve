import Foundation
import MLX
import MLXLMCommon

public struct SerializedKVLayer: @unchecked Sendable {
    public let state: [MLXArray]
    public let metaState: [String]
    public let className: String

    public init(state: [MLXArray], metaState: [String], className: String) {
        self.state = state
        self.metaState = metaState
        self.className = className
    }
}

public final class PrefixKVStoreHit: @unchecked Sendable {
    public let matchedTokenCount: Int
    public let blockCount: Int
    fileprivate let storage: Any

    init(matchedTokenCount: Int, blockCount: Int, storage: Any) {
        self.matchedTokenCount = matchedTokenCount
        self.blockCount = blockCount
        self.storage = storage
    }
}

public protocol PrefixKVStore: AnyObject, Sendable {
    func fetch(tokens: [Int]) -> PrefixKVStoreHit?
    func preload(_ hit: PrefixKVStoreHit) throws
    func reconstructCache(from hit: PrefixKVStoreHit) throws -> [SerializedKVLayer]
    func store(tokens: [Int], cache: [SerializedKVLayer]) throws
    func release(_ hit: PrefixKVStoreHit)
    func clearEntry(_ hit: PrefixKVStoreHit)
}

public final class BlockAwarePrefixKVStore: PrefixKVStore, @unchecked Sendable {
    public let prefixCache: BlockAwarePrefixCache
    private let lock = NSRecursiveLock()
    private var _fetchHitCount = 0
    private var _storeCount = 0
    private var _releaseCount = 0
    private var _clearCount = 0

    public var fetchHitCount: Int {
        withLock { _fetchHitCount }
    }

    public var storeCount: Int {
        withLock { _storeCount }
    }

    public var releaseCount: Int {
        withLock { _releaseCount }
    }

    public var clearCount: Int {
        withLock { _clearCount }
    }

    public init(prefixCache: BlockAwarePrefixCache) {
        self.prefixCache = prefixCache
    }

    public func fetch(tokens: [Int]) -> PrefixKVStoreHit? {
        withLock {
            guard let hit = prefixCache.fetchCache(tokens: tokens) else { return nil }
            _fetchHitCount += 1
            return PrefixKVStoreHit(
                matchedTokenCount: hit.matchedTokenCount,
                blockCount: hit.blockCount,
                storage: hit
            )
        }
    }

    public func preload(_ hit: PrefixKVStoreHit) throws {
        try withLock {
            _ = try reconstructCache(from: hit)
        }
    }

    public func reconstructCache(from hit: PrefixKVStoreHit) throws -> [SerializedKVLayer] {
        try withLock {
            guard let rawHit = hit.storage as? PrefixCacheHit else {
                throw PrefixKVStoreError.invalidHit
            }

            return try prefixCache.reconstructCache(from: rawHit).map { layerCache in
                SerializedKVLayer(
                    state: layerCache.state,
                    metaState: layerCache.metaState,
                    className: "KVCacheSimple"
                )
            }
        }
    }

    public func store(tokens: [Int], cache: [SerializedKVLayer]) throws {
        try withLock {
            let layerCaches = try cache.map { layer in
                try Self.cache(from: layer)
            }
            try prefixCache.storeCache(tokens: tokens, cache: layerCaches)
            _storeCount += 1
        }
    }

    public func release(_ hit: PrefixKVStoreHit) {
        withLock {
            guard let rawHit = hit.storage as? PrefixCacheHit else { return }
            prefixCache.release(rawHit)
            _releaseCount += 1
        }
    }

    public func clearEntry(_ hit: PrefixKVStoreHit) {
        withLock {
            release(hit)
            _clearCount += 1
        }
    }

    public static func cache(from layer: SerializedKVLayer) throws -> KVCacheSimple {
        guard layer.state.count == 2 else {
            throw PrefixKVStoreError.unsupportedLayerState
        }

        let cache = KVCacheSimple()
        cache.state = layer.state
        return cache
    }

    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }
}

public enum PrefixKVStoreError: Error, Equatable {
    case invalidHit
    case unsupportedLayerState
}
