import Foundation
@testable import MLXServeHTTP
import XCTest

final class OpenAIServerTests: XCTestCase {
    func testNegativeContentLengthIsRejected() throws {
        let rawRequest = Data(
            """
            POST /v1/chat/completions HTTP/1.1\r
            Host: localhost\r
            Content-Length: -1\r
            \r
            {}
            """.utf8
        )

        XCTAssertThrowsError(try HTTPRequest.parseComplete(rawRequest)) { error in
            XCTAssertEqual(error as? OpenAIServerError, .invalidContentLength)
        }
    }
}
