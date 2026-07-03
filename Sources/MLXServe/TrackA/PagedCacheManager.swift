import Foundation

public final class PagedCacheManager: @unchecked Sendable {
    public let blockSize: Int

    private var blocks: [KVCacheBlock]
    private var freeBlocks = FreeBlockQueue()
    private var hashIndex = BlockHashIndex()
    private var payloadsByHash: [Data: KVCacheBlockPayload] = [:]
    private var tick: UInt64 = 0

    public init(blockSize: Int = 256, initialCapacity: Int = 0) {
        self.blockSize = blockSize
        self.blocks = [KVCacheBlock(blockID: 0)]
        if initialCapacity > 0 {
            grow(by: initialCapacity)
        }
    }

    public var allocatedBlocks: Int {
        blocks.filter(\.isAllocated).count
    }

    public var totalRefCount: Int {
        blocks.reduce(0) { $0 + $1.refCount }
    }

    public func contains(hash: Data) -> Bool {
        hashIndex.blockID(for: hash) != nil && payloadsByHash[hash] != nil
    }

    public func blockHash(for blockID: Int) -> Data? {
        guard blocks.indices.contains(blockID) else { return nil }
        return blocks[blockID].blockHash
    }

    public func blockHashes(for table: BlockTable) -> [Data] {
        table.blockIDs.compactMap { blockHash(for: $0) }
    }

    public func payload(for hash: Data) -> KVCacheBlockPayload? {
        payloadsByHash[hash]
    }

    public func payload(for blockID: Int) -> KVCacheBlockPayload? {
        guard let hash = blockHash(for: blockID) else { return nil }
        return payload(for: hash)
    }

    @discardableResult
    public func storeBlock(
        hash: Data,
        tokenCount: Int,
        payload: KVCacheBlockPayload
    ) -> Int {
        if let blockID = hashIndex.blockID(for: hash) {
            payloadsByHash[hash] = payloadsByHash[hash] ?? payload
            touch(blockID: blockID)
            return blockID
        }

        let blockID = allocateBlockID()
        let block = blocks[blockID]
        block.refCount = 1
        block.blockHash = hash
        block.tokenCount = tokenCount
        block.lastAccessTick = nextTick()
        hashIndex.insert(blockID: blockID, hash: hash)
        payloadsByHash[hash] = payload
        return blockID
    }

    public func retain(hash: Data) -> Int? {
        guard let blockID = hashIndex.blockID(for: hash) else { return nil }
        retain(blockID: blockID)
        return blockID
    }

    public func retain(blockID: Int) {
        guard blocks.indices.contains(blockID), blocks[blockID].isAllocated else { return }
        blocks[blockID].refCount += 1
        touch(blockID: blockID)
    }

    public func release(_ table: BlockTable) {
        for blockID in table.blockIDs {
            release(blockID: blockID)
        }
    }

    public func forkBlockTable(_ table: BlockTable) -> BlockTable {
        for blockID in table.blockIDs {
            retain(blockID: blockID)
        }
        return table
    }

    public func getBlocksForGeneration(_ table: inout BlockTable) {
        for (index, blockID) in table.blockIDs.enumerated() {
            guard blocks.indices.contains(blockID), blocks[blockID].refCount > 1,
                let hash = blocks[blockID].blockHash,
                let payload = payloadsByHash[hash]
            else {
                continue
            }

            blocks[blockID].refCount -= 1
            let cloneID = allocateBlockID()
            let clone = blocks[cloneID]
            clone.refCount = 1
            clone.blockHash = hash
            clone.tokenCount = blocks[blockID].tokenCount
            clone.lastAccessTick = nextTick()
            hashIndex.insert(blockID: cloneID, hash: hash)
            payloadsByHash[hash] = payload
            table.blockIDs[index] = cloneID
        }
    }

    public func clearAll() {
        blocks = [KVCacheBlock(blockID: 0)]
        freeBlocks = FreeBlockQueue()
        hashIndex = BlockHashIndex()
        payloadsByHash.removeAll()
        tick = 0
    }

    private func release(blockID: Int) {
        guard blocks.indices.contains(blockID), blocks[blockID].isAllocated else { return }
        blocks[blockID].refCount -= 1
        guard blocks[blockID].refCount <= 0 else { return }

        if let hash = blocks[blockID].blockHash {
            hashIndex.remove(blockID: blockID, hash: hash)
            if hashIndex.blockIDs(for: hash).isEmpty {
                payloadsByHash.removeValue(forKey: hash)
            }
        }

        blocks[blockID].refCount = 0
        blocks[blockID].blockHash = nil
        blocks[blockID].tokenCount = 0
        blocks[blockID].lastAccessTick = 0
        freeBlocks.push(blockID)
    }

    private func touch(blockID: Int) {
        guard blocks.indices.contains(blockID) else { return }
        blocks[blockID].lastAccessTick = nextTick()
    }

    private func allocateBlockID() -> Int {
        if let blockID = freeBlocks.pop() {
            return blockID
        }

        let blockID = blocks.count
        blocks.append(KVCacheBlock(blockID: blockID))
        return blockID
    }

    private func grow(by count: Int) {
        guard count > 0 else { return }
        for _ in 0 ..< count {
            let blockID = blocks.count
            blocks.append(KVCacheBlock(blockID: blockID))
            freeBlocks.push(blockID)
        }
    }

    private func nextTick() -> UInt64 {
        tick += 1
        return tick
    }
}
