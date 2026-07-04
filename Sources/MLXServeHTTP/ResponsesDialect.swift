import Foundation

public struct ResponsesRequest {
    public let model: String
    public let inputMessages: [OpenAIChatMessage]
    public let temperature: Float
    public let topP: Float
    public let maxOutputTokens: Int
    public let stream: Bool
    public let text: [String: Any]?
    public let previousResponseID: String?
    public let store: Bool
    public let metadata: [String: Any]
    public let seed: Int?
    public let chatTemplateKwargs: [String: OpenAIJSONValue]?

    public static func parse(_ body: Data) throws -> ResponsesRequest {
        guard
            let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
            let model = object["model"] as? String,
            let input = object["input"]
        else {
            throw OpenAIServerError.invalidJSON
        }

        var messages: [OpenAIChatMessage] = []
        if let instructions = object["instructions"] as? String, !instructions.isEmpty {
            messages.append(OpenAIChatMessage(role: "system", content: instructions))
        }
        messages.append(contentsOf: try responsesInputMessages(input))
        guard !messages.isEmpty else { throw OpenAIServerError.invalidJSON }

        return ResponsesRequest(
            model: model,
            inputMessages: messages,
            temperature: responsesFloatValue(object["temperature"]) ?? 0,
            topP: responsesFloatValue(object["top_p"]) ?? 0,
            maxOutputTokens: responsesIntValue(object["max_output_tokens"]) ?? 16,
            stream: object["stream"] as? Bool ?? false,
            text: object["text"] as? [String: Any],
            previousResponseID: object["previous_response_id"] as? String,
            store: object["store"] as? Bool ?? true,
            metadata: object["metadata"] as? [String: Any] ?? [:],
            seed: responsesIntValue(object["seed"]),
            chatTemplateKwargs: try responsesChatTemplateKwargs(from: object)
        )
    }

    public func openAIRequest(previousMessages: [OpenAIChatMessage] = [], stream: Bool? = nil) -> OpenAIChatRequest {
        OpenAIChatRequest(
            model: model,
            messages: previousMessages + inputMessages,
            maxTokens: maxOutputTokens,
            temperature: temperature,
            topP: topP,
            seed: seed,
            stream: stream ?? self.stream,
            chatTemplateKwargs: chatTemplateKwargs
        )
    }

    public var textPayload: [String: Any] {
        text ?? ["format": ["type": "text"]]
    }
}

public struct ResponsesBufferedCompletion {
    public let text: String
    public let completionTokens: Int

    public init(text: String, completionTokens: Int) {
        self.text = text
        self.completionTokens = completionTokens
    }
}

public struct ResponseSSEEvent {
    public let name: String
    public let payload: [String: Any]

    public init(name: String, payload: [String: Any]) {
        self.name = name
        self.payload = payload
    }
}

public actor ResponsesStore {
    public struct Record: Sendable {
        public let responseData: Data
        public let contextMessages: [OpenAIChatMessage]

        public init(responseData: Data, contextMessages: [OpenAIChatMessage]) {
            self.responseData = responseData
            self.contextMessages = contextMessages
        }
    }

    private var records: [String: Record] = [:]

    public init() {}

    public func put(id: String, responseData: Data, contextMessages: [OpenAIChatMessage]) {
        records[id] = Record(responseData: responseData, contextMessages: contextMessages)
    }

    public func responseData(id: String) -> Data? {
        records[id]?.responseData
    }

    public func contextMessages(id: String) -> [OpenAIChatMessage]? {
        records[id]?.contextMessages
    }

    public func delete(id: String) -> Bool {
        records.removeValue(forKey: id) != nil
    }
}

public struct ResponsesStreamFormatter {
    private let id: String
    private let model: String
    private let createdAt: Int
    private let promptTokens: Int
    private let request: ResponsesRequest
    private var parser = ThinkingParser()
    private var sequenceNumber = 0
    private var completionTokens = 0
    private var reasoningText = ""
    private var outputText = ""
    private var reasoningStarted = false
    private var reasoningDone = false
    private var messageStarted = false

    public init(
        id: String,
        model: String,
        createdAt: Int,
        promptTokens: Int,
        request: ResponsesRequest
    ) {
        self.id = id
        self.model = model
        self.createdAt = createdAt
        self.promptTokens = promptTokens
        self.request = request
    }

    public mutating func startEvents() -> [ResponseSSEEvent] {
        [
            event(
                "response.created",
                [
                    "type": "response.created",
                    "response": baseResponse(status: "in_progress", output: [], usage: NSNull()),
                ]
            ),
            event(
                "response.in_progress",
                [
                    "type": "response.in_progress",
                    "response": baseResponse(status: "in_progress", output: [], usage: NSNull()),
                ]
            ),
        ]
    }

    public mutating func feed(_ chunk: OpenAIChatChunk) -> [ResponseSSEEvent] {
        completionTokens += 1
        return parserEvents(for: parser.feed(chunk.text))
    }

    public mutating func finishEvents() -> [ResponseSSEEvent] {
        var events = parserEvents(for: parser.finish())
        if reasoningStarted && !reasoningDone {
            events.append(reasoningDoneEvent())
        }
        if !messageStarted {
            events.append(contentsOf: startMessageEvents())
        }
        events.append(outputTextDoneEvent())
        events.append(contentPartDoneEvent())
        events.append(outputItemDoneEvent())
        events.append(
            event(
                "response.completed",
                [
                    "type": "response.completed",
                    "response": completedResponseObject(id: id, createdAt: createdAt),
                ]
            )
        )
        return events
    }

    public func completedResponseObject(id: String? = nil, createdAt: Int? = nil) -> [String: Any] {
        buildResponsesObject(
            request: request,
            id: id ?? self.id,
            createdAt: createdAt ?? self.createdAt,
            promptTokens: promptTokens,
            completion: ResponsesBufferedCompletion(text: fullText, completionTokens: completionTokens)
        )
    }

    public var completedContextMessages: [OpenAIChatMessage] {
        request.inputMessages + [OpenAIChatMessage(role: "assistant", content: outputText, reasoningContent: reasoningText.isEmpty ? nil : reasoningText)]
    }

    private var fullText: String {
        reasoningText.isEmpty ? outputText : "<think>\(reasoningText)</think>\(outputText)"
    }

    private mutating func parserEvents(for delta: (reasoning: String, content: String)) -> [ResponseSSEEvent] {
        var events: [ResponseSSEEvent] = []
        if !delta.reasoning.isEmpty {
            events.append(contentsOf: reasoningDeltaEvents(delta.reasoning))
        }
        if !delta.content.isEmpty {
            if reasoningStarted && !reasoningDone {
                events.append(reasoningDoneEvent())
            }
            if !messageStarted {
                events.append(contentsOf: startMessageEvents())
            }
            outputText += delta.content
            events.append(
                event(
                    "response.output_text.delta",
                    [
                        "type": "response.output_text.delta",
                        "item_id": messageItemID,
                        "output_index": outputIndex,
                        "content_index": 0,
                        "delta": delta.content,
                    ]
                )
            )
        }
        return events
    }

    private mutating func reasoningDeltaEvents(_ text: String) -> [ResponseSSEEvent] {
        var events: [ResponseSSEEvent] = []
        if !reasoningStarted {
            reasoningStarted = true
            events.append(
                event(
                    "response.output_item.added",
                    [
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": reasoningItem(status: "in_progress", text: ""),
                    ]
                )
            )
        }
        reasoningText += text
        events.append(
            event(
                "response.reasoning_text.delta",
                [
                    "type": "response.reasoning_text.delta",
                    "item_id": reasoningItemID,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                ]
            )
        )
        return events
    }

    private mutating func reasoningDoneEvent() -> ResponseSSEEvent {
        reasoningDone = true
        return event(
            "response.output_item.done",
            [
                "type": "response.output_item.done",
                "output_index": 0,
                "item": reasoningItem(status: "completed", text: reasoningText),
            ]
        )
    }

    private mutating func startMessageEvents() -> [ResponseSSEEvent] {
        messageStarted = true
        return [
            event(
                "response.output_item.added",
                [
                    "type": "response.output_item.added",
                    "output_index": outputIndex,
                    "item": messageItem(status: "in_progress", text: nil),
                ]
            ),
            event(
                "response.content_part.added",
                [
                    "type": "response.content_part.added",
                    "item_id": messageItemID,
                    "output_index": outputIndex,
                    "content_index": 0,
                    "content_part": ["type": "output_text", "text": ""],
                ]
            ),
        ]
    }

    private mutating func outputTextDoneEvent() -> ResponseSSEEvent {
        event(
            "response.output_text.done",
            [
                "type": "response.output_text.done",
                "item_id": messageItemID,
                "output_index": outputIndex,
                "content_index": 0,
                "text": outputText,
            ]
        )
    }

    private mutating func contentPartDoneEvent() -> ResponseSSEEvent {
        event(
            "response.content_part.done",
            [
                "type": "response.content_part.done",
                "item_id": messageItemID,
                "output_index": outputIndex,
                "content_index": 0,
                "content_part": ["type": "output_text", "text": outputText, "annotations": []],
            ]
        )
    }

    private mutating func outputItemDoneEvent() -> ResponseSSEEvent {
        event(
            "response.output_item.done",
            [
                "type": "response.output_item.done",
                "output_index": outputIndex,
                "item": messageItem(status: "completed", text: outputText),
            ]
        )
    }

    private mutating func event(_ name: String, _ payload: [String: Any]) -> ResponseSSEEvent {
        var payload = payload
        payload["sequence_number"] = sequenceNumber
        sequenceNumber += 1
        return ResponseSSEEvent(name: name, payload: payload)
    }

    private var outputIndex: Int {
        reasoningText.isEmpty && !reasoningStarted ? 0 : 1
    }

    private var messageItemID: String {
        "\(id)_msg"
    }

    private var reasoningItemID: String {
        "\(id)_reasoning"
    }

    private func baseResponse(status: String, output: [[String: Any]], usage: Any) -> [String: Any] {
        [
            "id": id,
            "object": "response",
            "created_at": createdAt,
            "model": model,
            "status": status,
            "output": output,
            "usage": usage,
            "text": request.textPayload,
            "previous_response_id": request.previousResponseID as Any? ?? NSNull(),
            "metadata": request.metadata,
        ]
    }

    private func reasoningItem(status: String, text: String) -> [String: Any] {
        [
            "type": "reasoning",
            "id": reasoningItemID,
            "status": status,
            "summary": [],
            "content": text.isEmpty ? [] : [["type": "reasoning_text", "text": text]],
        ]
    }

    private func messageItem(status: String, text: String?) -> [String: Any] {
        var item: [String: Any] = [
            "type": "message",
            "id": messageItemID,
            "role": "assistant",
            "status": status,
        ]
        if let text {
            item["content"] = [["type": "output_text", "text": text, "annotations": []]]
        } else {
            item["content"] = []
        }
        return item
    }
}

public func buildResponsesObject(
    request: ResponsesRequest,
    id: String = "resp_\(UUID().uuidString.prefix(8))",
    createdAt: Int = Int(Date().timeIntervalSince1970),
    promptTokens: Int,
    completion: ResponsesBufferedCompletion
) -> [String: Any] {
    let extracted = extractThinking(completion.text)
    let reasoningTokens = responsesEstimatedTokenCount(extracted.reasoning)
    var output: [[String: Any]] = []
    if !extracted.reasoning.isEmpty {
        output.append(
            [
                "type": "reasoning",
                "id": "\(id)_reasoning",
                "summary": [],
                "content": [["type": "reasoning_text", "text": extracted.reasoning]],
            ]
        )
    }
    output.append(
        [
            "type": "message",
            "id": "\(id)_msg",
            "role": "assistant",
            "status": "completed",
            "content": [["type": "output_text", "text": extracted.content, "annotations": []]],
        ]
    )

    return [
        "id": id,
        "object": "response",
        "created_at": createdAt,
        "model": request.model,
        "status": "completed",
        "output": output,
        "usage": [
            "input_tokens": promptTokens,
            "output_tokens": completion.completionTokens,
            "total_tokens": promptTokens + completion.completionTokens,
            "output_tokens_details": ["reasoning_tokens": reasoningTokens],
        ],
        "text": request.textPayload,
        "previous_response_id": request.previousResponseID as Any? ?? NSNull(),
        "metadata": request.metadata,
    ]
}

public func responsesEstimatedTokenCount(_ text: String) -> Int {
    guard !text.isEmpty else { return 0 }
    let wordLikeTokens = text.split { $0.isWhitespace || $0.isPunctuation }.count
    return max(1, wordLikeTokens)
}

public func responsesJSONData(_ object: [String: Any]) throws -> Data {
    try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
}

private func responsesInputMessages(_ input: Any) throws -> [OpenAIChatMessage] {
    if let string = input as? String {
        return [OpenAIChatMessage(role: "user", content: string)]
    }
    guard let items = input as? [[String: Any]] else {
        throw OpenAIServerError.invalidJSON
    }

    var messages: [OpenAIChatMessage] = []
    for item in items {
        guard let type = item["type"] as? String else {
            throw OpenAIServerError.invalidJSON
        }
        switch type {
        case "message":
            guard let role = item["role"] as? String else {
                throw OpenAIServerError.invalidJSON
            }
            messages.append(OpenAIChatMessage(role: role, content: try responsesMessageContent(item["content"])))
        case "function_call":
            messages.append(OpenAIChatMessage(role: "assistant", content: try responsesCanonicalJSONString(item)))
        case "function_call_output":
            let output = item["output"] as? String ?? (try? responsesCanonicalJSONString(item)) ?? ""
            messages.append(OpenAIChatMessage(role: "tool", content: output))
        default:
            continue
        }
    }
    return messages
}

private func responsesMessageContent(_ value: Any?) throws -> String {
    if let string = value as? String {
        return string
    }
    guard let parts = value as? [[String: Any]] else {
        throw OpenAIServerError.invalidJSON
    }
    return parts.compactMap { part in
        let type = part["type"] as? String
        switch type {
        case "input_text", "output_text":
            return part["text"] as? String
        default:
            return nil
        }
    }.joined()
}

private func responsesChatTemplateKwargs(from object: [String: Any]) throws -> [String: OpenAIJSONValue]? {
    var kwargs: [String: OpenAIJSONValue] = [:]
    for key in ["tools", "tool_choice", "reasoning", "text"] {
        guard let value = object[key] else { continue }
        guard let jsonValue = OpenAIJSONValue(value) else {
            throw OpenAIServerError.invalidJSON
        }
        kwargs[key] = jsonValue
    }
    return kwargs.isEmpty ? nil : kwargs
}

private func responsesCanonicalJSONString(_ object: [String: Any]) throws -> String {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    return String(decoding: data, as: UTF8.self)
}

private func responsesFloatValue(_ value: Any?) -> Float? {
    switch value {
    case let value as Double:
        return Float(value)
    case let value as Float:
        return value
    case let value as Int:
        return Float(value)
    case let value as NSNumber:
        return value.floatValue
    default:
        return nil
    }
}

private func responsesIntValue(_ value: Any?) -> Int? {
    switch value {
    case let value as Int:
        return value
    case let value as NSNumber:
        return value.intValue
    default:
        return nil
    }
}
