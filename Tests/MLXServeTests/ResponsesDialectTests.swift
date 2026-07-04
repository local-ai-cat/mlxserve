import Foundation
@testable import MLXServeHTTP
import XCTest

final class ResponsesDialectTests: XCTestCase {
    func testResponsesRequestParsesStringInputWithDefaults() throws {
        let request = try ResponsesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "input": "hello"
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.inputMessages.count, 1)
        XCTAssertEqual(request.inputMessages[0].role, "user")
        XCTAssertEqual(request.inputMessages[0].content, "hello")
        XCTAssertEqual(request.maxOutputTokens, 16)
        XCTAssertFalse(request.stream)
        XCTAssertTrue(request.store)
        XCTAssertNil(request.previousResponseID)
        XCTAssertEqual(request.textPayload["format"] as? [String: String], ["type": "text"])
    }

    func testResponsesRequestParsesArrayInputAndPreviousResponseID() throws {
        let request = try ResponsesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "instructions": "Be direct.",
                  "input": [
                    {
                      "type": "message",
                      "role": "user",
                      "content": [
                        {"type": "input_text", "text": "question "},
                        {"type": "output_text", "text": "context"}
                      ]
                    },
                    {"type": "function_call", "name": "lookup", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "result"}
                  ],
                  "temperature": 0.5,
                  "top_p": 0.7,
                  "max_output_tokens": 32,
                  "stream": true,
                  "previous_response_id": "resp_prev",
                  "store": false,
                  "metadata": {"trace": "abc"},
                  "reasoning": {"effort": "low"},
                  "seed": 123
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.inputMessages.count, 4)
        XCTAssertEqual(request.inputMessages.map(\.role), ["system", "user", "assistant", "tool"])
        XCTAssertEqual(request.inputMessages[0].content, "Be direct.")
        XCTAssertEqual(request.inputMessages[1].content, "question context")
        XCTAssertEqual(request.inputMessages[3].content, "result")
        XCTAssertEqual(request.temperature, 0.5, accuracy: 0.0001)
        XCTAssertEqual(request.topP, 0.7, accuracy: 0.0001)
        XCTAssertEqual(request.maxOutputTokens, 32)
        XCTAssertTrue(request.stream)
        XCTAssertEqual(request.previousResponseID, "resp_prev")
        XCTAssertFalse(request.store)
        XCTAssertEqual(request.metadata["trace"] as? String, "abc")
        XCTAssertEqual(request.seed, 123)
        XCTAssertEqual(request.chatTemplateKwargs?["reasoning"], .object(["effort": .string("low")]))
    }

    func testResponsesRequestRejectsMissingInput() {
        let body = Data(
            """
            {
              "model": "test-model"
            }
            """.utf8
        )

        XCTAssertThrowsError(try ResponsesRequest.parse(body)) { error in
            XCTAssertEqual(error as? OpenAIServerError, .invalidJSON)
        }
    }

    func testBuildResponsesObjectSplitsReasoningAndCountsReasoningTokens() throws {
        let request = try ResponsesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "input": "hello",
                  "previous_response_id": "resp_prev",
                  "metadata": {"trace": "abc"},
                  "text": {"format": {"type": "json_object"}}
                }
                """.utf8
            )
        )
        let response = buildResponsesObject(
            request: request,
            id: "resp_test",
            createdAt: 123,
            promptTokens: 5,
            completion: ResponsesBufferedCompletion(
                text: "<think>step one</think>answer",
                completionTokens: 4
            )
        )

        XCTAssertEqual(response["id"] as? String, "resp_test")
        XCTAssertEqual(response["object"] as? String, "response")
        XCTAssertEqual(response["created_at"] as? Int, 123)
        XCTAssertEqual(response["status"] as? String, "completed")
        XCTAssertEqual(response["previous_response_id"] as? String, "resp_prev")
        XCTAssertEqual((response["metadata"] as? [String: Any])?["trace"] as? String, "abc")

        let output = try XCTUnwrap(response["output"] as? [[String: Any]])
        XCTAssertEqual(output.count, 2)
        XCTAssertEqual(output[0]["type"] as? String, "reasoning")
        let reasoningContent = try XCTUnwrap(output[0]["content"] as? [[String: String]])
        XCTAssertEqual(reasoningContent[0]["type"], "reasoning_text")
        XCTAssertEqual(reasoningContent[0]["text"], "step one")
        XCTAssertEqual(output[1]["type"] as? String, "message")
        let messageContent = try XCTUnwrap(output[1]["content"] as? [[String: Any]])
        XCTAssertEqual(messageContent[0]["type"] as? String, "output_text")
        XCTAssertEqual(messageContent[0]["text"] as? String, "answer")

        let usage = try XCTUnwrap(response["usage"] as? [String: Any])
        XCTAssertEqual(usage["input_tokens"] as? Int, 5)
        XCTAssertEqual(usage["output_tokens"] as? Int, 4)
        XCTAssertEqual(usage["total_tokens"] as? Int, 9)
        let details = try XCTUnwrap(usage["output_tokens_details"] as? [String: Int])
        XCTAssertEqual(details["reasoning_tokens"], 2)
    }

    func testStreamingFormatterEventOrderWithoutReasoning() throws {
        let request = try ResponsesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "input": "hello",
                  "stream": true
                }
                """.utf8
            )
        )
        var formatter = ResponsesStreamFormatter(
            id: "resp_test",
            model: "test-model",
            createdAt: 123,
            promptTokens: 5,
            request: request
        )

        var events = formatter.startEvents()
        events.append(contentsOf: formatter.feed(OpenAIChatChunk(text: "answer", tokenID: 1)))
        events.append(contentsOf: formatter.finishEvents())

        XCTAssertEqual(
            events.map(\.name),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        )
        XCTAssertEqual(events.compactMap { $0.payload["sequence_number"] as? Int }, Array(0..<events.count))
    }

    func testStreamingFormatterReasoningEventsPrecedeMessageEvents() throws {
        let request = try ResponsesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "input": "hello",
                  "stream": true
                }
                """.utf8
            )
        )
        var formatter = ResponsesStreamFormatter(
            id: "resp_test",
            model: "test-model",
            createdAt: 123,
            promptTokens: 5,
            request: request
        )

        var events = formatter.startEvents()
        events.append(contentsOf: formatter.feed(OpenAIChatChunk(text: "<think>reason</think>answer", tokenID: 1)))
        events.append(contentsOf: formatter.finishEvents())

        XCTAssertEqual(
            events.map(\.name),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        )
        XCTAssertEqual(events.compactMap { $0.payload["sequence_number"] as? Int }, Array(0..<events.count))

        let addedItems = events.filter { $0.name == "response.output_item.added" }
        let firstItem = try XCTUnwrap(addedItems[0].payload["item"] as? [String: Any])
        let secondItem = try XCTUnwrap(addedItems[1].payload["item"] as? [String: Any])
        XCTAssertEqual(firstItem["type"] as? String, "reasoning")
        XCTAssertEqual(secondItem["type"] as? String, "message")
    }

    func testResponsesStoreRoundTrip() async throws {
        let store = ResponsesStore()
        let response: [String: Any] = [
            "id": "resp_test",
            "object": "response",
            "output": [],
        ]
        let data = try responsesJSONData(response)
        let context = [OpenAIChatMessage(role: "assistant", content: "answer")]

        await store.put(id: "resp_test", responseData: data, contextMessages: context)
        let maybeStoredData = await store.responseData(id: "resp_test")
        let storedData = try XCTUnwrap(maybeStoredData)
        let storedObject = try XCTUnwrap(JSONSerialization.jsonObject(with: storedData) as? [String: Any])
        XCTAssertEqual(storedObject["id"] as? String, "resp_test")
        let storedContext = await store.contextMessages(id: "resp_test")
        XCTAssertEqual(storedContext?.first?.content, "answer")

        let firstDelete = await store.delete(id: "resp_test")
        let dataAfterDelete = await store.responseData(id: "resp_test")
        let secondDelete = await store.delete(id: "resp_test")
        XCTAssertTrue(firstDelete)
        XCTAssertNil(dataAfterDelete)
        XCTAssertFalse(secondDelete)
    }
}
