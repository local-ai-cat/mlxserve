import Foundation
@testable import MLXServeHTTP
import XCTest

final class AnthropicDialectTests: XCTestCase {
    func testMessagesRequestRequiresMaxTokens() {
        let body = Data(
            """
            {
              "model": "test-model",
              "messages": [{"role": "user", "content": "hello"}]
            }
            """.utf8
        )

        XCTAssertThrowsError(try AnthropicMessagesRequest.parse(body)) { error in
            XCTAssertEqual(error as? OpenAIServerError, .invalidJSON)
        }
    }

    func testMessagesRequestParsesSystemStringAndTranslationFields() throws {
        let request = try AnthropicMessagesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "max_tokens": 64,
                  "system": "You are concise.",
                  "messages": [{"role": "user", "content": "hello"}],
                  "stop_sequences": ["END"],
                  "stream": true,
                  "temperature": 0.4,
                  "top_p": 0.8,
                  "top_k": 20,
                  "thinking": {"type": "enabled", "budget_tokens": 16},
                  "tools": [{"name": "lookup", "description": "search"}],
                  "tool_choice": {"type": "tool", "name": "lookup"}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.maxTokens, 64)
        XCTAssertEqual(request.messages.map(\.role), ["system", "user"])
        XCTAssertEqual(request.messages[0].content, "You are concise.")
        XCTAssertEqual(request.stopSequences, ["END"])
        XCTAssertTrue(request.stream)
        XCTAssertEqual(request.temperature, 0.4, accuracy: 0.0001)
        XCTAssertEqual(request.topP, 0.8, accuracy: 0.0001)
        XCTAssertEqual(request.topK, 20)
        XCTAssertEqual(request.enableThinking, true)
        XCTAssertEqual(request.chatTemplateKwargs?["tools"], .array([.object(["name": .string("lookup"), "description": .string("search")])]))
        XCTAssertEqual(request.chatTemplateKwargs?["tool_choice"], .object(["type": .string("tool"), "name": .string("lookup")]))

        let chatRequest = request.openAIRequest()
        XCTAssertEqual(chatRequest.model, "test-model")
        XCTAssertEqual(chatRequest.maxTokens, 64)
        XCTAssertEqual(chatRequest.stop, ["END"])
        XCTAssertEqual(chatRequest.enableThinking, true)
    }

    func testMessagesRequestParsesSystemArrayAndContentBlocks() throws {
        let request = try AnthropicMessagesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "max_tokens": 16,
                  "system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "image", "source": {"type": "base64", "data": "..."}},
                        {"type": "tool_result", "content": [{"type": "text", "text": "result"}]}
                      ]
                    }
                  ]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.messages.count, 2)
        XCTAssertEqual(request.messages[0].content, "AB")
        XCTAssertEqual(request.messages[1].content, "hello result")
    }

    func testBuildAnthropicMessageResponseSplitsThinkingAndText() throws {
        let request = try AnthropicMessagesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hello"}]
                }
                """.utf8
            )
        )
        let completion = AnthropicBufferedCompletion(
            text: "<think>reasoning</think>answer",
            completionTokens: 3,
            finishReason: "stop",
            stoppedByTextStop: false,
            stopSequence: nil
        )

        let response = buildAnthropicMessageResponse(
            request: request,
            completion: completion,
            promptTokens: 5,
            id: "msg_test"
        )

        XCTAssertEqual(response["id"] as? String, "msg_test")
        XCTAssertEqual(response["type"] as? String, "message")
        XCTAssertEqual(response["role"] as? String, "assistant")
        XCTAssertEqual(response["model"] as? String, "test-model")
        XCTAssertEqual(response["stop_reason"] as? String, "end_turn")
        let content = try XCTUnwrap(response["content"] as? [[String: Any]])
        XCTAssertEqual(content.count, 2)
        XCTAssertEqual(content[0]["type"] as? String, "thinking")
        XCTAssertEqual(content[0]["thinking"] as? String, "reasoning")
        XCTAssertEqual(content[1]["type"] as? String, "text")
        XCTAssertEqual(content[1]["text"] as? String, "answer")
        let usage = try XCTUnwrap(response["usage"] as? [String: Int])
        XCTAssertEqual(usage["input_tokens"], 5)
        XCTAssertEqual(usage["output_tokens"], 3)
    }

    func testStreamingFormatterEventOrderForThinkingThenText() {
        var formatter = AnthropicStreamFormatter(
            id: "msg_test",
            model: "test-model",
            promptTokens: 5,
            stopSequences: []
        )

        var events = formatter.startEvents()
        events.append(contentsOf: formatter.feed(OpenAIChatChunk(text: "<think>reason", tokenID: 1)))
        events.append(contentsOf: formatter.feed(OpenAIChatChunk(text: "ing</think>answer", tokenID: 2, finishReason: "stop")))
        events.append(contentsOf: formatter.finishEvents())

        XCTAssertEqual(
            events.map(\.name),
            [
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        )

        let firstBlock = events[2].payload["content_block"] as? [String: Any]
        XCTAssertEqual(firstBlock?["type"] as? String, "thinking")
        let secondBlock = events[6].payload["content_block"] as? [String: Any]
        XCTAssertEqual(secondBlock?["type"] as? String, "text")
    }

    func testStreamingFormatterReportsStopSequence() throws {
        var formatter = AnthropicStreamFormatter(
            id: "msg_test",
            model: "test-model",
            promptTokens: 5,
            stopSequences: ["END"]
        )

        var events = formatter.startEvents()
        events.append(contentsOf: formatter.feed(OpenAIChatChunk(text: "answer EN", tokenID: 1)))
        events.append(contentsOf: formatter.feed(OpenAIChatChunk(text: "D hidden", tokenID: 2)))
        events.append(contentsOf: formatter.finishEvents())

        let delta = try XCTUnwrap(events.first { $0.name == "message_delta" }?.payload["delta"] as? [String: Any])
        XCTAssertEqual(delta["stop_reason"] as? String, "stop_sequence")
        XCTAssertEqual(delta["stop_sequence"] as? String, "END")
    }
}
