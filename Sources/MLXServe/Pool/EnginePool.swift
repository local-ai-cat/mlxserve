import Foundation

public protocol EnginePoolModelLoader: Sendable {
    associatedtype Engine: Sendable

    func loadModel(id: String, modelURL: URL) async throws -> Engine
    func unloadModel(_ engine: Engine, id: String) async
}

public struct EnginePoolLease<Engine: Sendable>: Sendable {
    public let id: UUID
    public let modelID: String
    public let engine: Engine

    fileprivate init(id: UUID, modelID: String, engine: Engine) {
        self.id = id
        self.modelID = modelID
        self.engine = engine
    }
}

public struct EnginePoolQueueTicket: Sendable {
    public let id: UUID

    fileprivate init(id: UUID) {
        self.id = id
    }
}

public struct EnginePoolModelStatus: Equatable, Sendable {
    public let id: String
    public let modelPath: String
    public let loaded: Bool
    public let isLoading: Bool
    public let estimatedSize: Int64
    public let actualSize: Int64?
    public let pinned: Bool
    public let lastAccess: TimeInterval?
    public let inUse: Int
}

public struct EnginePoolStatus: Equatable, Sendable {
    public let finalCeiling: Int64
    public let currentModelMemory: Int64
    public let modelCount: Int
    public let loadedCount: Int
    public let models: [EnginePoolModelStatus]
}

private struct EnginePoolEntry<Engine: Sendable>: Sendable {
    let modelID: String
    let modelURL: URL
    let estimatedSize: Int64
    var actualSize: Int64?
    var engine: Engine?
    var lastAccess: Date?
    var isLoading: Bool
    var isPinned: Bool
    var activeLeaseIDs: Set<UUID>

    var inUse: Int {
        activeLeaseIDs.count
    }
}

public actor EnginePool<Loader: EnginePoolModelLoader> {
    private var entries: [String: EnginePoolEntry<Loader.Engine>]
    private let loader: Loader
    private let finalCeilingBytes: Int64
    private let idleTimeout: TimeInterval?
    private let maxWaitingQueueDepth: Int
    private var currentModelMemory: Int64
    private var loadInProgress: Bool
    private var activeQueueTickets: Set<UUID>

    public init(
        models: [String: DiscoveredModel],
        loader: Loader,
        finalCeiling: Int64 = 0,
        idleTimeout: TimeInterval? = nil,
        maxWaitingQueueDepth: Int = 0
    ) {
        var entries: [String: EnginePoolEntry<Loader.Engine>] = [:]
        for (id, model) in models {
            entries[id] = EnginePoolEntry(
                modelID: id,
                modelURL: model.modelURL,
                estimatedSize: model.estimatedSize,
                actualSize: nil,
                engine: nil,
                lastAccess: nil,
                isLoading: false,
                isPinned: false,
                activeLeaseIDs: []
            )
        }

        self.entries = entries
        self.loader = loader
        self.finalCeilingBytes = max(0, finalCeiling)
        self.idleTimeout = idleTimeout
        self.maxWaitingQueueDepth = max(0, maxWaitingQueueDepth)
        self.currentModelMemory = 0
        self.loadInProgress = false
        self.activeQueueTickets = []
    }

    public func availableModelIDs() -> [String] {
        entries.keys.sorted()
    }

    public func setPinned(_ isPinned: Bool, for modelID: String) throws {
        guard var entry = entries[modelID] else {
            throw EnginePoolError.modelNotFound(id: modelID, available: availableModelIDs())
        }
        entry.isPinned = isPinned
        entries[modelID] = entry
    }

    public func acquire(_ modelID: String, now: Date = Date()) async throws -> EnginePoolLease<Loader.Engine> {
        let engine = try await getOrLoadEngine(modelID, takeLease: true, now: now)
        guard let entry = entries[modelID], let loadedEngine = entry.engine else {
            throw EnginePoolError.modelLoading(id: modelID)
        }
        return EnginePoolLease(id: engine.leaseID, modelID: modelID, engine: loadedEngine)
    }

    public func release(_ lease: EnginePoolLease<Loader.Engine>) async {
        guard var entry = entries[lease.modelID] else { return }
        guard entry.activeLeaseIDs.remove(lease.id) != nil else { return }
        entries[lease.modelID] = entry
    }

    public func load(_ modelID: String, now: Date = Date()) async throws {
        _ = try await getOrLoadEngine(modelID, takeLease: false, now: now)
    }

    public func unload(_ modelID: String) async throws {
        guard let entry = entries[modelID] else {
            throw EnginePoolError.modelNotFound(id: modelID, available: availableModelIDs())
        }
        guard entry.engine != nil else {
            throw EnginePoolError.modelNotLoaded(id: modelID)
        }
        guard !entry.isLoading, entry.inUse == 0 else {
            throw EnginePoolError.modelBusy(id: modelID, operation: "unload")
        }

        await unloadLoadedModel(modelID)
    }

    public func sweepIdleModels(now: Date = Date()) async -> [String] {
        guard let idleTimeout, idleTimeout > 0 else {
            return []
        }

        var unloaded: [String] = []
        let candidates = entries.values
            .filter { entry in
                guard let lastAccess = entry.lastAccess else { return false }
                return entry.engine != nil
                    && !entry.isPinned
                    && !entry.isLoading
                    && entry.inUse == 0
                    && now.timeIntervalSince(lastAccess) >= idleTimeout
            }
            .sorted { left, right in
                (left.lastAccess ?? .distantPast) < (right.lastAccess ?? .distantPast)
            }

        for candidate in candidates {
            await unloadLoadedModel(candidate.modelID)
            unloaded.append(candidate.modelID)
        }
        return unloaded
    }

    public func status() -> EnginePoolStatus {
        let modelStatuses = entries.values
            .sorted { $0.modelID < $1.modelID }
            .map { entry in
                EnginePoolModelStatus(
                    id: entry.modelID,
                    modelPath: entry.modelURL.path,
                    loaded: entry.engine != nil,
                    isLoading: entry.isLoading,
                    estimatedSize: entry.estimatedSize,
                    actualSize: entry.actualSize,
                    pinned: entry.isPinned,
                    lastAccess: entry.lastAccess?.timeIntervalSince1970,
                    inUse: entry.inUse
                )
            }

        return EnginePoolStatus(
            finalCeiling: finalCeilingBytes,
            currentModelMemory: currentModelMemory,
            modelCount: entries.count,
            loadedCount: modelStatuses.filter(\.loaded).count,
            models: modelStatuses
        )
    }

    public func admitWaitingRequest() throws -> EnginePoolQueueTicket {
        let ticket = EnginePoolQueueTicket(id: UUID())
        guard maxWaitingQueueDepth > 0 else {
            return ticket
        }
        let currentDepth = activeQueueTickets.count
        guard currentDepth < maxWaitingQueueDepth else {
            throw EnginePoolError.schedulerQueueFull(current: currentDepth, max: maxWaitingQueueDepth)
        }
        activeQueueTickets.insert(ticket.id)
        return ticket
    }

    public func finishWaitingRequest(_ ticket: EnginePoolQueueTicket) {
        activeQueueTickets.remove(ticket.id)
    }

    private func getOrLoadEngine(
        _ modelID: String,
        takeLease: Bool,
        now: Date
    ) async throws -> (engine: Loader.Engine, leaseID: UUID) {
        guard var entry = entries[modelID] else {
            throw EnginePoolError.modelNotFound(id: modelID, available: availableModelIDs())
        }

        if let engine = entry.engine {
            let leaseID = UUID()
            entry.lastAccess = now
            if takeLease {
                entry.activeLeaseIDs.insert(leaseID)
            }
            entries[modelID] = entry
            return (engine, leaseID)
        }

        guard !entry.isLoading, !loadInProgress else {
            throw EnginePoolError.modelLoading(id: modelID)
        }

        loadInProgress = true

        do {
            try await admitModelForLoading(entry)

            var loadingEntry = entries[modelID] ?? entry
            loadingEntry.isLoading = true
            loadingEntry.lastAccess = now
            entries[modelID] = loadingEntry

            let engine = try await loader.loadModel(id: modelID, modelURL: loadingEntry.modelURL)
            var loaded = entries[modelID] ?? loadingEntry
            let leaseID = UUID()
            loaded.engine = engine
            loaded.isLoading = false
            loaded.lastAccess = now
            if takeLease {
                loaded.activeLeaseIDs.insert(leaseID)
            }
            entries[modelID] = loaded
            currentModelMemory += loaded.actualSize ?? loaded.estimatedSize
            loadInProgress = false
            return (engine, leaseID)
        } catch {
            var failed = entries[modelID] ?? entry
            failed.isLoading = false
            entries[modelID] = failed
            loadInProgress = false
            throw error
        }
    }

    private func admitModelForLoading(_ entry: EnginePoolEntry<Loader.Engine>) async throws {
        guard finalCeilingBytes > 0 else {
            return
        }

        if entry.estimatedSize > finalCeilingBytes {
            throw EnginePoolError.modelTooLarge(
                id: entry.modelID,
                size: entry.estimatedSize,
                ceiling: finalCeilingBytes
            )
        }

        while currentModelMemory + entry.estimatedSize > finalCeilingBytes {
            guard let victim = findLRUVictim() else {
                throw EnginePoolError.insufficientMemory(
                    id: entry.modelID,
                    required: entry.estimatedSize,
                    current: currentModelMemory,
                    ceiling: finalCeilingBytes
                )
            }
            await unloadLoadedModel(victim)
        }
    }

    private func findLRUVictim() -> String? {
        entries.values
            .filter { entry in
                entry.engine != nil
                    && !entry.isPinned
                    && !entry.isLoading
                    && entry.inUse == 0
            }
            .sorted { left, right in
                (left.lastAccess ?? .distantPast) < (right.lastAccess ?? .distantPast)
            }
            .first?
            .modelID
    }

    private func unloadLoadedModel(_ modelID: String) async {
        guard var entry = entries[modelID], let engine = entry.engine else {
            return
        }

        let modelMemory = entry.actualSize ?? entry.estimatedSize
        entry.engine = nil
        entry.actualSize = nil
        entry.activeLeaseIDs.removeAll()
        currentModelMemory = max(0, currentModelMemory - modelMemory)
        entries[modelID] = entry

        await loader.unloadModel(engine, id: modelID)
    }
}
