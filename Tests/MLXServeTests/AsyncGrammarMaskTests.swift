import Foundation
@testable import MLXServe
import XCTest

final class AsyncGrammarMaskTests: XCTestCase {
    func testPreparedMaskBecomesReadyOffThread() {
        let mask = AsyncGrammarMask<TestMaskSnapshot>()

        mask.prepareCurrentState(from: TestMaskSnapshot(tokenIDs: [1, 2, 3]))

        XCTAssertEqual(waitForReadyMask(mask), [1, 2, 3])
        XCTAssertEqual(mask.completedCount, 1)
    }

    func testStalePreparedMaskIsDiscardedAfterStateAdvance() {
        let mask = AsyncGrammarMask<TestMaskSnapshot>()

        mask.prepareCurrentState(from: TestMaskSnapshot(tokenIDs: [1], delay: 0.05))
        mask.prepareAdvancedState(from: TestMaskSnapshot(tokenIDs: [2]))

        XCTAssertEqual(waitForReadyMask(mask), [2])
        XCTAssertTrue(waitForCompletedCount(mask, atLeast: 2))
        XCTAssertEqual(mask.readyTokenIDs, [2])
    }

    private func waitForReadyMask(
        _ mask: AsyncGrammarMask<TestMaskSnapshot>,
        timeout: TimeInterval = 1
    ) -> [Int]? {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let tokenIDs = mask.readyTokenIDs {
                return tokenIDs
            }
            Thread.sleep(forTimeInterval: 0.005)
        }
        return mask.readyTokenIDs
    }

    private func waitForCompletedCount(
        _ mask: AsyncGrammarMask<TestMaskSnapshot>,
        atLeast expectedCount: Int,
        timeout: TimeInterval = 1
    ) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if mask.completedCount >= expectedCount {
                return true
            }
            Thread.sleep(forTimeInterval: 0.005)
        }
        return mask.completedCount >= expectedCount
    }
}

private struct TestMaskSnapshot: GrammarMaskSnapshot {
    let tokenIDs: [Int]
    var delay: TimeInterval = 0

    func allowedTokenIDs() -> [Int] {
        if delay > 0 {
            Thread.sleep(forTimeInterval: delay)
        }
        return tokenIDs
    }
}
