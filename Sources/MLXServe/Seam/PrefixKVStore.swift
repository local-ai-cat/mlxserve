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
    public private(set) var fetchHitCount = 0
    public private(set) var storeCount = 0
    public private(set) var releaseCount = 0
    public private(set) var clearCount = 0

    public init(prefixCache: BlockAwarePrefixCache) {
        self.prefixCache = prefixCache
    }

    public func fetch(tokens: [Int]) -> PrefixKVStoreHit? {
        guard let hit = prefixCache.fetchCache(tokens: tokens) else { return nil }
        fetchHitCount += 1
        return PrefixKVStoreHit(
            matchedTokenCount: hit.matchedTokenCount,
            blockCount: hit.blockCount,
            storage: hit
        )
    }

    public func preload(_ hit: PrefixKVStoreHit) throws {
        _ = try reconstructCache(from: hit)
    }

    public func reconstructCache(from hit: PrefixKVStoreHit) throws -> [SerializedKVLayer] {
        guard let rawHit = hit.storage as? PrefixCacheHit else {
            throw PrefixKVStoreError.invalidHit
        }

        return prefixCache.reconstructCache(from: rawHit).map { layerCache in
            SerializedKVLayer(
                state: layerCache.state,
                metaState: layerCache.metaState,
                className: "KVCacheSimple"
            )
        }
    }

    public func store(tokens: [Int], cache: [SerializedKVLayer]) throws {
        let layerCaches = try cache.map { layer in
            try Self.cache(from: layer)
        }
        prefixCache.storeCache(tokens: tokens, cache: layerCaches)
        storeCount += 1
    }

    public func release(_ hit: PrefixKVStoreHit) {
        guard let rawHit = hit.storage as? PrefixCacheHit else { return }
        prefixCache.release(rawHit)
        releaseCount += 1
    }

    public func clearEntry(_ hit: PrefixKVStoreHit) {
        release(hit)
        clearCount += 1
    }

    public static func cache(from layer: SerializedKVLayer) throws -> KVCacheSimple {
        guard layer.state.count == 2 else {
            throw PrefixKVStoreError.unsupportedLayerState
        }

        let cache = KVCacheSimple()
        cache.state = layer.state
        return cache
    }
}

public enum PrefixKVStoreError: Error, Equatable {
    case invalidHit
    case unsupportedLayerState
}
