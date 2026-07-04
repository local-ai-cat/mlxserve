import Foundation
@testable import MLXServeHTTP
import XCTest

final class StructuredOutputTests: XCTestCase {
    func testChatResponseFormatJSONObjectsParseToStructuredOutputSpec() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "response_format": {"type": "json_object"}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .jsonObject)
        XCTAssertNotNil(request.structuredOutputWarning)
    }

    func testChatResponseFormatJSONSchemaParsesToStructuredOutputSpec() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                      "name": "answer",
                      "schema": {
                        "type": "object",
                        "properties": {
                          "answer": {"type": "string"}
                        },
                        "required": ["answer"]
                      }
                    }
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(
            request.structuredOutput,
            .jsonSchema(
                name: "answer",
                schema: [
                    "type": .string("object"),
                    "properties": .object([
                        "answer": .object(["type": .string("string")])
                    ]),
                    "required": .array([.string("answer")]),
                ]
            )
        )
        XCTAssertNotNil(request.structuredOutputWarning)
    }

    func testChatResponseFormatTextIsNoOp() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "response_format": {"type": "text"}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .none)
        XCTAssertNil(request.structuredOutputWarning)
    }

    func testStructuredOutputsChoiceParses() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "structured_outputs": {
                    "type": "choice",
                    "choices": ["yes", "no"]
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .choice(["yes", "no"]))
        XCTAssertNil(request.structuredOutputWarning)
    }

    func testStructuredOutputsPrecedeResponseFormat() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "response_format": {"type": "json_object"},
                  "structured_outputs": {
                    "type": "choice",
                    "choices": ["yes", "no"]
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .choice(["yes", "no"]))
        XCTAssertNil(request.structuredOutputWarning)
    }

    func testStructuredOutputsJSONMapsToJSONObjectFallback() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "structured_outputs": {"type": "json"}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .jsonObject)
        XCTAssertNotNil(request.structuredOutputWarning)
    }

    func testStructuredOutputsEmptyChoiceThrowsBadRequest() {
        XCTAssertThrowsError(
            try OpenAIChatRequest.parse(
                Data(
                    """
                    {
                      "model": "test-model",
                      "messages": [{"role": "user", "content": "hello"}],
                      "structured_outputs": {
                        "type": "choice",
                        "choices": []
                      }
                    }
                    """.utf8
                )
            )
        ) { error in
            XCTAssertEqual((error as? OpenAIServerError)?.httpStatus, 400)
        }
    }

    func testGuidedGrammarThrowsBadRequest() {
        XCTAssertThrowsError(
            try OpenAIChatRequest.parse(
                Data(
                    """
                    {
                      "model": "test-model",
                      "messages": [{"role": "user", "content": "hello"}],
                      "guided_grammar": "root ::= \\"yes\\""
                    }
                    """.utf8
                )
            )
        ) { error in
            XCTAssertEqual((error as? OpenAIServerError)?.httpStatus, 400)
        }
    }

    func testStructuredOutputsGrammarAndRegexThrowBadRequest() {
        for type in ["grammar", "regex"] {
            XCTAssertThrowsError(
                try OpenAIChatRequest.parse(
                    Data(
                        """
                        {
                          "model": "test-model",
                          "messages": [{"role": "user", "content": "hello"}],
                          "structured_outputs": {"type": "\(type)"}
                        }
                        """.utf8
                    )
                )
            ) { error in
                XCTAssertEqual((error as? OpenAIServerError)?.httpStatus, 400)
            }
        }
    }

    func testCompletionRequestParsesStructuredOutputsChoice() throws {
        let request = try OpenAICompletionRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "prompt": "hello",
                  "structured_outputs": {
                    "type": "choice",
                    "choices": ["A", "B"]
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .choice(["A", "B"]))
    }
}
