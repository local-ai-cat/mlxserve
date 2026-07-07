import Foundation

protocol GrammarMaskSnapshot: Sendable {
    func allowedTokenIDs() -> [Int]
}

final class AsyncGrammarMask<Snapshot: GrammarMaskSnapshot>: @unchecked Sendable {
    private let lock = NSLock()
    private var stateID = 0
    private var readyStateID: Int?
    private var readyAllowedTokenIDs: [Int]?
    private var computedMaskCount = 0

    var readyTokenIDs: [Int]? {
        lock.lock()
        defer { lock.unlock() }
        guard readyStateID == stateID else {
            return nil
        }
        return readyAllowedTokenIDs
    }

    var completedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return computedMaskCount
    }

    func prepareCurrentState(from snapshot: Snapshot) {
        let targetStateID: Int
        lock.lock()
        targetStateID = stateID
        lock.unlock()
        schedule(snapshot: snapshot, stateID: targetStateID)
    }

    func prepareAdvancedState(from snapshot: Snapshot) {
        let targetStateID: Int
        lock.lock()
        stateID += 1
        targetStateID = stateID
        readyStateID = nil
        readyAllowedTokenIDs = nil
        lock.unlock()
        schedule(snapshot: snapshot, stateID: targetStateID)
    }

    private func schedule(snapshot: Snapshot, stateID targetStateID: Int) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self, snapshot] in
            let allowedTokenIDs = snapshot.allowedTokenIDs()
            guard let self else { return }
            self.lock.lock()
            defer { self.lock.unlock() }
            self.computedMaskCount += 1
            guard self.stateID == targetStateID else {
                return
            }
            self.readyStateID = targetStateID
            self.readyAllowedTokenIDs = allowedTokenIDs
        }
    }
}

struct PrecomputedGrammarMasks {
    var jsonAllowedTokenIDs: [Int]?
    var regexAllowedTokenIDs: [Int]?
    var gbnfAllowedTokenIDs: [Int]?
}
