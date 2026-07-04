@testable import MLXServeHTTP
import XCTest

final class ThinkingParserTests: XCTestCase {
    func testExtractThinkingSplitsCompleteBlock() {
        let result = extractThinking("<think>reasoning</think>answer")

        XCTAssertEqual(result.reasoning, "reasoning")
        XCTAssertEqual(result.content, "answer")
    }

    func testExtractThinkingPassesThroughTextWithoutThinkTags() {
        let result = extractThinking("just answer")

        XCTAssertEqual(result.reasoning, "")
        XCTAssertEqual(result.content, "just answer")
    }

    func testExtractThinkingRecoversTailWithoutOpenTag() {
        let result = extractThinking("reasoning</think>answer")

        XCTAssertEqual(result.reasoning, "reasoning")
        XCTAssertEqual(result.content, "answer")
    }

    func testExtractThinkingTreatsMalformedOpenAsContent() {
        let result = extractThinking("<think>everything")

        XCTAssertEqual(result.reasoning, "")
        XCTAssertEqual(result.content, "everything")
    }

    func testStreamingParserSplitsAcrossChunks() {
        var parser = ThinkingParser()

        let first = parser.feed("<think>Let me")
        let second = parser.feed(" think</think>Answer")
        let final = parser.finish()

        XCTAssertEqual(first.reasoning, "Let me")
        XCTAssertEqual(first.content, "")
        XCTAssertEqual(second.reasoning, " think")
        XCTAssertEqual(second.content, "Answer")
        XCTAssertEqual(final.reasoning, "")
        XCTAssertEqual(final.content, "")
    }

    func testStreamingParserBuffersPartialTagsAcrossChunks() {
        var parser = ThinkingParser()

        let first = parser.feed("<thi")
        let second = parser.feed("nk>reason")
        let third = parser.feed("</thi")
        let fourth = parser.feed("nk>answer")

        XCTAssertEqual(first.reasoning, "")
        XCTAssertEqual(first.content, "")
        XCTAssertEqual(second.reasoning, "reason")
        XCTAssertEqual(second.content, "")
        XCTAssertEqual(third.reasoning, "")
        XCTAssertEqual(third.content, "")
        XCTAssertEqual(fourth.reasoning, "")
        XCTAssertEqual(fourth.content, "answer")
    }

    func testStreamingParserRecoversMalformedOpenOnFinish() {
        var parser = ThinkingParser()

        let first = parser.feed("<think>unfinished")
        let final = parser.finish()

        XCTAssertEqual(first.reasoning, "unfinished")
        XCTAssertEqual(first.content, "")
        XCTAssertEqual(final.reasoning, "")
        XCTAssertEqual(final.content, "unfinished")
    }
}
