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

    func testOpenAIErrorBodyShapeIncludesTypeParamAndCode() throws {
        let body = openAIErrorBody(message: "not found", status: 404)
        let error = try XCTUnwrap(body["error"] as? [String: Any])

        XCTAssertEqual(error["message"] as? String, "not found")
        XCTAssertEqual(error["type"] as? String, "not_found_error")
        XCTAssertTrue(error["param"] is NSNull)
        XCTAssertTrue(error["code"] is NSNull)
    }

    func testOpenAIErrorTypeMapping() {
        XCTAssertEqual(openAIErrorType(status: 400), "invalid_request_error")
        XCTAssertEqual(openAIErrorType(status: 422), "invalid_request_error")
        XCTAssertEqual(openAIErrorType(status: 404), "not_found_error")
        XCTAssertEqual(openAIErrorType(status: 500), "server_error")
        XCTAssertEqual(openAIErrorType(status: 503), "server_error")
    }

    func testChatRequestParsePopulatesSamplingAndOpenAIFields() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [
                    {
                      "role": "assistant",
                      "content": "hello",
                      "reasoning_content": "prior reasoning"
                    }
                  ],
                  "max_completion_tokens": 32,
                  "temperature": 0.7,
                  "top_p": 0.8,
                  "top_k": 42,
                  "repetition_penalty": 1.2,
                  "min_p": 0.05,
                  "xtc_probability": 0.3,
                  "xtc_threshold": 0.4,
                  "presence_penalty": 0.5,
                  "frequency_penalty": 0.6,
                  "stop": ["END", "STOP"],
                  "seed": 1234,
                  "stream": true,
                  "stream_options": { "include_usage": true },
                  "enable_thinking": false,
                  "chat_template_kwargs": {
                    "reasoning_effort": "low",
                    "budget": 16,
                    "nested": { "enabled": true }
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.messages[0].reasoningContent, "prior reasoning")
        XCTAssertEqual(request.maxTokens, 32)
        XCTAssertEqual(request.temperature, 0.7, accuracy: 0.0001)
        XCTAssertEqual(request.topP, 0.8, accuracy: 0.0001)
        XCTAssertEqual(request.topK, 42)
        XCTAssertEqual(request.repetitionPenalty, 1.2, accuracy: 0.0001)
        XCTAssertEqual(request.minP, 0.05, accuracy: 0.0001)
        XCTAssertEqual(request.xtcProbability, 0.3, accuracy: 0.0001)
        XCTAssertEqual(request.xtcThreshold, 0.4, accuracy: 0.0001)
        XCTAssertEqual(request.presencePenalty, 0.5, accuracy: 0.0001)
        XCTAssertEqual(request.frequencyPenalty, 0.6, accuracy: 0.0001)
        XCTAssertEqual(request.stop, ["END", "STOP"])
        XCTAssertEqual(request.seed, 1234)
        XCTAssertTrue(request.stream)
        XCTAssertTrue(request.includeUsage)
        XCTAssertEqual(request.enableThinking, false)
        XCTAssertEqual(request.chatTemplateKwargs?["reasoning_effort"], .string("low"))
        XCTAssertEqual(request.chatTemplateKwargs?["budget"], .number(16))
        XCTAssertEqual(request.chatTemplateKwargs?["nested"], .object(["enabled": .bool(true)]))
    }

    func testChatRequestParseCoercesSingleStopString() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "stop": "END"
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.stop, ["END"])
    }

    func testChatRequestParseUsesNeutralDefaults() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.maxTokens, 16)
        XCTAssertEqual(request.temperature, 0)
        XCTAssertEqual(request.topP, 0)
        XCTAssertEqual(request.topK, 0)
        XCTAssertEqual(request.repetitionPenalty, 1)
        XCTAssertEqual(request.minP, 0)
        XCTAssertEqual(request.xtcProbability, 0)
        XCTAssertEqual(request.xtcThreshold, 0.1, accuracy: 0.0001)
        XCTAssertEqual(request.presencePenalty, 0)
        XCTAssertEqual(request.frequencyPenalty, 0)
        XCTAssertEqual(request.stop, [])
        XCTAssertNil(request.seed)
        XCTAssertFalse(request.stream)
        XCTAssertFalse(request.includeUsage)
        XCTAssertNil(request.enableThinking)
        XCTAssertNil(request.chatTemplateKwargs)
    }

    func testChatRequestParseIncludeUsageDefaultsFalse() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "stream": true
                }
                """.utf8
            )
        )

        XCTAssertFalse(request.includeUsage)
    }
}
