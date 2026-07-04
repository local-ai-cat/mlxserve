import Foundation
@testable import MLXServe
import XCTest

private struct FakeEngine: Equatable, Sendable {
    let id: String
}

private actor FakeLoaderState {
    private(set) var loaded: [String] = []
    private(set) var unloaded: [String] = []

    func recordLoad(_ id: String) {
        loaded.append(id)
    }

    func recordUnload(_ id: String) {
        unloaded.append(id)
    }

    func unloadedIDs() -> [String] {
        unloaded
    }
}

private struct FakeLoader: EnginePoolModelLoader {
    let state: FakeLoaderState

    func loadModel(id: String, modelURL: URL) async throws -> FakeEngine {
        await state.recordLoad(id)
        return FakeEngine(id: id)
    }

    func unloadModel(_ engine: FakeEngine, id: String) async {
        await state.recordUnload(id)
    }
}

final class EnginePoolTests: XCTestCase {
    func testOnDemandLoadTakesAndReleasesLease() async throws {
        let pool = makePool(["a": 10])

        let lease = try await pool.acquire("a")
        var status = await pool.status()
        XCTAssertEqual(status.loadedCount, 1)
        XCTAssertEqual(status.currentModelMemory, 10)
        XCTAssertEqual(status.models.first?.inUse, 1)
        XCTAssertEqual(lease.engine.id, "a")

        await pool.release(lease)
        status = await pool.status()
        XCTAssertEqual(status.models.first?.inUse, 0)
    }

    func testLRUEvictionUsesOldestUnpinnedIdleModel() async throws {
        let state = FakeLoaderState()
        let pool = makePool(["a": 60, "b": 60, "c": 60], finalCeiling: 120, state: state)
        let start = Date(timeIntervalSince1970: 1_000)

        try await pool.load("a", now: start)
        try await pool.load("b", now: start.addingTimeInterval(10))
        try await pool.load("c", now: start.addingTimeInterval(20))

        let status = await pool.status()
        XCTAssertEqual(status.currentModelMemory, 120)
        XCTAssertEqual(loadedModelIDs(status), ["b", "c"])
        let unloaded = await state.unloadedIDs()
        XCTAssertEqual(unloaded, ["a"])
    }

    func testPinnedModelPreventsEviction() async throws {
        let pool = makePool(["a": 60, "b": 60], finalCeiling: 60)
        try await pool.load("a")
        try await pool.setPinned(true, for: "a")

        await XCTAssertThrowsEnginePoolError(try await pool.load("b")) { error in
            XCTAssertEqual(error.httpStatus, 507)
            if case .insufficientMemory(let id, _, _, _) = error {
                XCTAssertEqual(id, "b")
            } else {
                XCTFail("expected insufficient memory")
            }
        }

        let status = await pool.status()
        XCTAssertEqual(loadedModelIDs(status), ["a"])
    }

    func testLeasePreventsEvictionUntilReleased() async throws {
        let state = FakeLoaderState()
        let pool = makePool(["a": 60, "b": 60], finalCeiling: 60, state: state)
        let lease = try await pool.acquire("a")

        await XCTAssertThrowsEnginePoolError(try await pool.load("b")) { error in
            XCTAssertEqual(error.httpStatus, 507)
        }
        var unloaded = await state.unloadedIDs()
        XCTAssertEqual(unloaded, [])

        await pool.release(lease)
        try await pool.load("b")

        let status = await pool.status()
        XCTAssertEqual(loadedModelIDs(status), ["b"])
        unloaded = await state.unloadedIDs()
        XCTAssertEqual(unloaded, ["a"])
    }

    func testModelTooLargeUses507() async throws {
        let pool = makePool(["huge": 100], finalCeiling: 50)

        await XCTAssertThrowsEnginePoolError(try await pool.load("huge")) { error in
            XCTAssertEqual(error.httpStatus, 507)
            if case .modelTooLarge(let id, let size, let ceiling) = error {
                XCTAssertEqual(id, "huge")
                XCTAssertEqual(size, 100)
                XCTAssertEqual(ceiling, 50)
            } else {
                XCTFail("expected model too large")
            }
        }
    }

    func testIdleTimeoutUnloadsOnlyUnpinnedIdleModels() async throws {
        let pool = makePool(["a": 10, "b": 10], idleTimeout: 30)
        let start = Date(timeIntervalSince1970: 1_000)

        try await pool.load("a", now: start)
        try await pool.load("b", now: start.addingTimeInterval(1))
        try await pool.setPinned(true, for: "b")

        let unloaded = await pool.sweepIdleModels(now: start.addingTimeInterval(40))

        XCTAssertEqual(unloaded, ["a"])
        let status = await pool.status()
        XCTAssertEqual(loadedModelIDs(status), ["b"])
    }

    func testUnknownModelErrorListsAvailableModels() async throws {
        let pool = makePool(["a": 10, "b": 10])

        await XCTAssertThrowsEnginePoolError(try await pool.load("missing")) { error in
            XCTAssertEqual(error.httpStatus, 404)
            XCTAssertEqual(error.message, "Model 'missing' not found. Available models: a, b")
        }
    }

    func testSchedulerQueueFullHasRetryAfter() async throws {
        let pool = makePool(["a": 10], maxWaitingQueueDepth: 1)
        _ = try await pool.admitWaitingRequest()

        await XCTAssertThrowsEnginePoolError(try await pool.admitWaitingRequest()) { error in
            XCTAssertEqual(error.httpStatus, 503)
            XCTAssertEqual(error.retryAfterSeconds, 1)
            XCTAssertEqual(error.message, "Scheduler waiting queue full (1/1). Try again shortly.")
        }
    }

    func testMemoryGuardCeilingArithmetic() {
        let gib = MemoryGuard.gibibyte

        XCTAssertEqual(
            MemoryGuard.finalCeiling(
                recommendedWorkingSetBytes: 32 * gib,
                physicalMemoryBytes: 64 * gib,
                tier: .safe
            ),
            24 * gib
        )
        XCTAssertEqual(
            MemoryGuard.finalCeiling(
                recommendedWorkingSetBytes: 16 * gib,
                physicalMemoryBytes: 16 * gib,
                tier: .balanced
            ),
            12 * gib
        )
        XCTAssertEqual(
            MemoryGuard.finalCeiling(recommendedWorkingSetBytes: 16 * gib, tier: nil),
            0
        )
    }

    private func makePool(
        _ sizes: [String: Int64],
        finalCeiling: Int64 = 0,
        idleTimeout: TimeInterval? = nil,
        maxWaitingQueueDepth: Int = 0,
        state: FakeLoaderState = FakeLoaderState()
    ) -> EnginePool<FakeLoader> {
        var models: [String: DiscoveredModel] = [:]
        for (id, size) in sizes {
            models[id] = DiscoveredModel(
                id: id,
                modelURL: URL(fileURLWithPath: "/tmp/\(id)", isDirectory: true),
                estimatedSize: size
            )
        }
        return EnginePool(
            models: models,
            loader: FakeLoader(state: state),
            finalCeiling: finalCeiling,
            idleTimeout: idleTimeout,
            maxWaitingQueueDepth: maxWaitingQueueDepth
        )
    }

    private func loadedModelIDs(_ status: EnginePoolStatus) -> [String] {
        status.models.filter(\.loaded).map(\.id).sorted()
    }
}

private func XCTAssertThrowsEnginePoolError(
    _ expression: @autoclosure () async throws -> some Sendable,
    file: StaticString = #filePath,
    line: UInt = #line,
    _ assertion: (EnginePoolError) -> Void
) async {
    do {
        _ = try await expression()
        XCTFail("expected EnginePoolError", file: file, line: line)
    } catch let error as EnginePoolError {
        assertion(error)
    } catch {
        XCTFail("expected EnginePoolError, got \(error)", file: file, line: line)
    }
}
