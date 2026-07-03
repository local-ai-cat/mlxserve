import Foundation
import MLX
import MLXLMCommon

public struct PrefixCacheHit: Sendable, Equatable {
    public let matchedTokenCount: Int
    public let table: BlockTable
    public let blockHashes: [Data]

    public var blockCount: Int {
        table.count
    }
}

public final class BlockAwarePrefixCache: @unchecked Sendable {
    public let modelName: String
    public let blockSize: Int
    public let manager: PagedCacheManager

    public init(
        modelName: String,
        blockSize: Int = 256,
        manager: PagedCacheManager? = nil
    ) {
        self.modelName = modelName
        self.blockSize = blockSize
        self.manager = manager ?? PagedCacheManager(blockSize: blockSize)
    }

    public func fetchCache(tokens: [Int]) -> PrefixCacheHit? {
        let candidateHashes = BlockHashing.chainHashes(
            modelName: modelName,
            tokens: tokens,
            blockSize: blockSize
        )
        guard !candidateHashes.isEmpty else { return nil }

        var retainedBlockIDs: [Int] = []
        var retainedHashes: [Data] = []
        for hash in candidateHashes {
            guard let blockID = manager.retain(hash: hash) else { break }
            retainedBlockIDs.append(blockID)
            retainedHashes.append(hash)
        }

        guard !retainedBlockIDs.isEmpty else { return nil }
        return PrefixCacheHit(
            matchedTokenCount: retainedBlockIDs.count * blockSize,
            table: BlockTable(blockIDs: retainedBlockIDs),
            blockHashes: retainedHashes
        )
    }

    public func reconstructCache(from hit: PrefixCacheHit) -> [KVCacheSimple] {
        let payloads = hit.table.blockIDs.compactMap { manager.payload(for: $0) }
        guard let firstPayload = payloads.first else { return [] }

        return (0 ..< firstPayload.layers.count).map { layerIndex in
            let keys = concatenated(payloads.map { $0.layers[layerIndex].keys }, axis: 2)
            let values = concatenated(payloads.map { $0.layers[layerIndex].values }, axis: 2)
            let cache = KVCacheSimple()
            cache.state = [keys, values]
            return cache
        }
    }

    @discardableResult
    public func storeCache(tokens: [Int], cache: [any KVCache]) -> BlockTable {
        let fullBlockCount = tokens.count / blockSize
        guard fullBlockCount > 0 else { return BlockTable() }

        let hashes = BlockHashing.chainHashes(
            modelName: modelName,
            tokens: tokens,
            blockSize: blockSize
        )
        precondition(hashes.count == fullBlockCount)

        var blockIDs: [Int] = []
        for blockIndex in 0 ..< fullBlockCount {
            let hash = hashes[blockIndex]
            if manager.contains(hash: hash), let blockID = manager.retain(hash: hash) {
                manager.release(BlockTable(blockIDs: [blockID]))
                blockIDs.append(blockID)
                continue
            }

            let payload = payloadForBlock(
                cache: cache,
                tokenStart: blockIndex * blockSize,
                tokenEnd: (blockIndex + 1) * blockSize
            )
            let blockID = manager.storeBlock(
                hash: hash,
                tokenCount: blockSize,
                payload: payload
            )
            blockIDs.append(blockID)
        }

        return BlockTable(blockIDs: blockIDs)
    }

    public func release(_ hit: PrefixCacheHit) {
        manager.release(hit.table)
    }

    public func release(_ table: BlockTable) {
        manager.release(table)
    }

    private func payloadForBlock(
        cache: [any KVCache],
        tokenStart: Int,
        tokenEnd: Int
    ) -> KVCacheBlockPayload {
        let layers = cache.enumerated().map { _, layerCache in
            let state = layerCache.state
            precondition(state.count == 2, "M3 supports KVCache family only")
            let keys = state[0][0..., 0..., tokenStart ..< tokenEnd, 0...]
            let values = state[1][0..., 0..., tokenStart ..< tokenEnd, 0...]
            return CacheLayerBlockPayload(
                keys: keys,
                values: values,
                metaState: [
                    String(blockSize),
                    CacheTypeHandlers.encodeBool(false),
                ]
            )
        }
        return KVCacheBlockPayload(layers: layers)
    }
}
