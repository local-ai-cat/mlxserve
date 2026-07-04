import Foundation

public struct AnthropicMessagesRequest {
    public let model: String
    public let maxTokens: Int
    public let messages: [OpenAIChatMessage]
    public let stopSequences: [String]
    public let stream: Bool
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let enableThinking: Bool?
    public let chatTemplateKwargs: [String: OpenAIJSONValue]?

    public static func parse(_ body: Data) throws -> AnthropicMessagesRequest {
        let parsed = try anthropicParsedBase(body)
        let object = parsed.object
        guard let maxTokens = anthropicIntValue(object["max_tokens"]) else {
            throw OpenAIServerError.invalidJSON
        }

        let thinking = object["thinking"] as? [String: Any]
        let thinkingType = thinking?["type"] as? String
        let enableThinking: Bool?
        switch thinkingType {
        case "enabled":
            enableThinking = true
        case "disabled":
            enableThinking = false
        default:
            enableThinking = nil
        }

        return AnthropicMessagesRequest(
            model: parsed.model,
            maxTokens: maxTokens,
            messages: parsed.messages,
            stopSequences: try anthropicStringArray(object["stop_sequences"]),
            stream: object["stream"] as? Bool ?? false,
            temperature: anthropicFloatValue(object["temperature"]) ?? 0,
            topP: anthropicFloatValue(object["top_p"]) ?? 0,
            topK: anthropicIntValue(object["top_k"]) ?? 0,
            enableThinking: enableThinking,
            chatTemplateKwargs: try anthropicChatTemplateKwargs(from: object)
        )
    }

    public func openAIRequest(stream: Bool? = nil) -> OpenAIChatRequest {
        OpenAIChatRequest(
            model: model,
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            stop: stopSequences,
            stream: stream ?? self.stream,
            enableThinking: enableThinking,
            chatTemplateKwargs: chatTemplateKwargs
        )
    }

    public func estimatedInputTokens() -> Int {
        let text = messages.map { "\($0.role): \($0.content)" }.joined(separator: "\n")
        return anthropicEstimatedTokenCount(text)
    }
}

public struct AnthropicCountTokensRequest {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let chatTemplateKwargs: [String: OpenAIJSONValue]?

    public static func parse(_ body: Data) throws -> AnthropicCountTokensRequest {
        let parsed = try anthropicParsedBase(body)
        return AnthropicCountTokensRequest(
            model: parsed.model,
            messages: parsed.messages,
            chatTemplateKwargs: try anthropicChatTemplateKwargs(from: parsed.object)
        )
    }

    public func estimatedInputTokens() -> Int {
        let text = messages.map { "\($0.role): \($0.content)" }.joined(separator: "\n")
        return anthropicEstimatedTokenCount(text)
    }
}

public func buildAnthropicCountTokensResponse(request: AnthropicCountTokensRequest) -> [String: Int] {
    ["input_tokens": request.estimatedInputTokens()]
}

public struct AnthropicBufferedCompletion {
    public let text: String
    public let completionTokens: Int
    public let finishReason: String
    public let stoppedByTextStop: Bool
    public let stopSequence: String?

    public init(
        text: String,
        completionTokens: Int,
        finishReason: String,
        stoppedByTextStop: Bool,
        stopSequence: String?
    ) {
        self.text = text
        self.completionTokens = completionTokens
        self.finishReason = finishReason
        self.stoppedByTextStop = stoppedByTextStop
        self.stopSequence = stopSequence
    }
}

public struct AnthropicSSEEvent {
    public let name: String
    public let payload: [String: Any]

    public init(name: String, payload: [String: Any]) {
        self.name = name
        self.payload = payload
    }
}

struct AnthropicStopMatch {
    let text: String
    let stopped: Bool
    let stopSequence: String?
}

struct AnthropicStopSequenceMatcher {
    private let stopSequences: [String]
    private let maxTailCount: Int
    private var pending = ""
    private var matcher: StreamingStopSequenceMatcher

    init(stopSequences: [String]) {
        let stopSequences = stopSequences.filter { !$0.isEmpty }
        self.stopSequences = stopSequences
        self.maxTailCount = max((stopSequences.map(\.count).max() ?? 0) - 1, 0)
        self.matcher = StreamingStopSequenceMatcher(stopSequences: stopSequences)
    }

    mutating func feed(_ text: String) -> AnthropicStopMatch {
        let combined = pending + text
        let sequence = firstStopSequence(in: combined)
        let match = matcher.feed(text)
        if match.stopped {
            pending = ""
        } else {
            let tailCount = min(maxTailCount, combined.count)
            let emitEnd = combined.index(combined.endIndex, offsetBy: -tailCount)
            pending = String(combined[emitEnd...])
        }
        return AnthropicStopMatch(text: match.text, stopped: match.stopped, stopSequence: sequence)
    }

    mutating func finish() -> AnthropicStopMatch {
        let sequence = firstStopSequence(in: pending)
        let match = matcher.finish()
        pending = ""
        return AnthropicStopMatch(text: match.text, stopped: match.stopped, stopSequence: sequence)
    }

    private func firstStopSequence(in text: String) -> String? {
        var bestMatch: (range: Range<String.Index>, sequence: String)?
        for sequence in stopSequences {
            guard let range = text.range(of: sequence) else { continue }
            if let current = bestMatch {
                if range.lowerBound < current.range.lowerBound {
                    bestMatch = (range, sequence)
                }
            } else {
                bestMatch = (range, sequence)
            }
        }
        return bestMatch?.sequence
    }
}

private enum AnthropicContentBlockKind {
    case thinking
    case text
}

public struct AnthropicStreamFormatter {
    private let id: String
    private let model: String
    private let promptTokens: Int
    private let stopSequences: [String]
    private var stopMatcher: AnthropicStopSequenceMatcher
    private var thinkingParser = ThinkingParser()
    private var activeBlock: (kind: AnthropicContentBlockKind, index: Int)?
    private var nextBlockIndex = 0
    private var completionTokens = 0
    private var finishReason = "length"
    private var stoppedByTextStop = false
    private var stopSequence: String?

    public init(id: String, model: String, promptTokens: Int, stopSequences: [String]) {
        self.id = id
        self.model = model
        self.promptTokens = promptTokens
        self.stopSequences = stopSequences
        self.stopMatcher = AnthropicStopSequenceMatcher(stopSequences: stopSequences)
    }

    public mutating func startEvents() -> [AnthropicSSEEvent] {
        [
            AnthropicSSEEvent(
                name: "message_start",
                payload: [
                    "type": "message_start",
                    "message": [
                        "id": id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": NSNull(),
                        "stop_sequence": NSNull(),
                        "usage": ["input_tokens": promptTokens, "output_tokens": 0],
                    ],
                ]
            ),
            AnthropicSSEEvent(name: "ping", payload: ["type": "ping"]),
        ]
    }

    public mutating func feed(_ chunk: OpenAIChatChunk) -> [AnthropicSSEEvent] {
        guard !stoppedByTextStop else { return [] }

        completionTokens += 1
        if let chunkFinishReason = chunk.finishReason {
            finishReason = chunkFinishReason
        }

        let stopMatch = stopMatcher.feed(chunk.text)
        if stopMatch.stopped {
            finishReason = "stop"
            stoppedByTextStop = true
            stopSequence = stopMatch.stopSequence ?? stopSequences.first
        }

        return parserEvents(for: thinkingParser.feed(stopMatch.text))
    }

    public mutating func finishEvents() -> [AnthropicSSEEvent] {
        var events: [AnthropicSSEEvent] = []

        if !stoppedByTextStop {
            let stopMatch = stopMatcher.finish()
            if stopMatch.stopped {
                finishReason = "stop"
                stoppedByTextStop = true
                stopSequence = stopMatch.stopSequence ?? stopSequences.first
            }
            events.append(contentsOf: parserEvents(for: thinkingParser.feed(stopMatch.text)))
        }

        events.append(contentsOf: parserEvents(for: thinkingParser.finish()))
        if let activeBlock {
            events.append(blockStop(index: activeBlock.index))
            self.activeBlock = nil
        }

        events.append(
            AnthropicSSEEvent(
                name: "message_delta",
                payload: [
                    "type": "message_delta",
                    "delta": [
                        "stop_reason": anthropicStopReason(
                            finishReason: finishReason,
                            stoppedByTextStop: stoppedByTextStop
                        ),
                        "stop_sequence": stopSequence as Any? ?? NSNull(),
                    ],
                    "usage": ["output_tokens": completionTokens],
                ]
            )
        )
        events.append(AnthropicSSEEvent(name: "message_stop", payload: ["type": "message_stop"]))
        return events
    }

    private mutating func parserEvents(for delta: (reasoning: String, content: String)) -> [AnthropicSSEEvent] {
        var events: [AnthropicSSEEvent] = []
        if !delta.reasoning.isEmpty {
            events.append(contentsOf: deltaEvents(kind: .thinking, text: delta.reasoning))
        }
        if !delta.content.isEmpty {
            events.append(contentsOf: deltaEvents(kind: .text, text: delta.content))
        }
        return events
    }

    private mutating func deltaEvents(kind: AnthropicContentBlockKind, text: String) -> [AnthropicSSEEvent] {
        var events: [AnthropicSSEEvent] = []
        let index = ensureBlock(kind, events: &events)
        let delta: [String: Any]
        switch kind {
        case .thinking:
            delta = ["type": "thinking_delta", "thinking": text]
        case .text:
            delta = ["type": "text_delta", "text": text]
        }
        events.append(
            AnthropicSSEEvent(
                name: "content_block_delta",
                payload: ["type": "content_block_delta", "index": index, "delta": delta]
            )
        )
        return events
    }

    private mutating func ensureBlock(
        _ kind: AnthropicContentBlockKind,
        events: inout [AnthropicSSEEvent]
    ) -> Int {
        if let activeBlock, activeBlock.kind == kind {
            return activeBlock.index
        }
        if let activeBlock {
            events.append(blockStop(index: activeBlock.index))
        }

        let index = nextBlockIndex
        nextBlockIndex += 1
        activeBlock = (kind, index)
        let contentBlock: [String: Any]
        switch kind {
        case .thinking:
            contentBlock = ["type": "thinking", "thinking": ""]
        case .text:
            contentBlock = ["type": "text", "text": ""]
        }
        events.append(
            AnthropicSSEEvent(
                name: "content_block_start",
                payload: ["type": "content_block_start", "index": index, "content_block": contentBlock]
            )
        )
        return index
    }

    private func blockStop(index: Int) -> AnthropicSSEEvent {
        AnthropicSSEEvent(name: "content_block_stop", payload: ["type": "content_block_stop", "index": index])
    }
}

public func buildAnthropicMessageResponse(
    request: AnthropicMessagesRequest,
    completion: AnthropicBufferedCompletion,
    promptTokens: Int,
    id: String = "msg_\(UUID().uuidString.prefix(8))"
) -> [String: Any] {
    let extracted = extractThinking(completion.text)
    var content: [[String: Any]] = []
    if !extracted.reasoning.isEmpty {
        content.append(["type": "thinking", "thinking": extracted.reasoning])
    }
    if !extracted.content.isEmpty {
        content.append(["type": "text", "text": extracted.content])
    }

    return [
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": request.model,
        "content": content,
        "stop_reason": anthropicStopReason(
            finishReason: completion.finishReason,
            stoppedByTextStop: completion.stoppedByTextStop
        ),
        "stop_sequence": completion.stopSequence as Any? ?? NSNull(),
        "usage": ["input_tokens": promptTokens, "output_tokens": completion.completionTokens],
    ]
}

public func anthropicStopReason(finishReason: String, stoppedByTextStop: Bool) -> String {
    if stoppedByTextStop {
        return "stop_sequence"
    }
    switch finishReason {
    case "length":
        return "max_tokens"
    case "tool_calls", "tool_use":
        return "tool_use"
    default:
        return "end_turn"
    }
}

public func anthropicEstimatedTokenCount(_ text: String) -> Int {
    guard !text.isEmpty else { return 0 }
    let wordLikeTokens = text.split { $0.isWhitespace || $0.isPunctuation }.count
    return max(1, wordLikeTokens)
}

private func anthropicParsedBase(_ body: Data) throws -> (
    object: [String: Any],
    model: String,
    messages: [OpenAIChatMessage]
) {
    guard
        let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
        let model = object["model"] as? String,
        let rawMessages = object["messages"] as? [[String: Any]]
    else {
        throw OpenAIServerError.invalidJSON
    }

    var messages: [OpenAIChatMessage] = []
    if let system = try anthropicContentText(object["system"]), !system.isEmpty {
        messages.append(OpenAIChatMessage(role: "system", content: system))
    }

    for rawMessage in rawMessages {
        guard let role = rawMessage["role"] as? String else {
            throw OpenAIServerError.invalidJSON
        }
        let content = try anthropicContentText(rawMessage["content"]) ?? ""
        messages.append(OpenAIChatMessage(role: role, content: content))
    }
    guard !messages.isEmpty else { throw OpenAIServerError.invalidJSON }

    return (object, model, messages)
}

private func anthropicContentText(_ value: Any?) throws -> String? {
    guard let value else { return nil }
    if let string = value as? String {
        return string
    }
    if let blocks = value as? [[String: Any]] {
        var parts: [String] = []
        for block in blocks {
            let type = block["type"] as? String
            switch type {
            case "text":
                if let text = block["text"] as? String {
                    parts.append(text)
                }
            case "tool_result":
                if let text = try anthropicContentText(block["content"]) {
                    parts.append(text)
                }
            case "image":
                continue
            case "tool_use":
                let data = try JSONSerialization.data(withJSONObject: block, options: [.sortedKeys])
                parts.append(String(decoding: data, as: UTF8.self))
            default:
                continue
            }
        }
        return parts.joined()
    }
    throw OpenAIServerError.invalidJSON
}

private func anthropicChatTemplateKwargs(from object: [String: Any]) throws -> [String: OpenAIJSONValue]? {
    var kwargs: [String: OpenAIJSONValue] = [:]
    if let rawKwargs = object["chat_template_kwargs"] {
        guard let rawObject = rawKwargs as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }
        for (key, value) in rawObject {
            guard let jsonValue = OpenAIJSONValue(value) else {
                throw OpenAIServerError.invalidJSON
            }
            kwargs[key] = jsonValue
        }
    }

    for key in ["tools", "tool_choice", "thinking"] {
        guard kwargs[key] == nil, let value = object[key] else { continue }
        guard let jsonValue = OpenAIJSONValue(value) else {
            throw OpenAIServerError.invalidJSON
        }
        kwargs[key] = jsonValue
    }

    return kwargs.isEmpty ? nil : kwargs
}

private func anthropicStringArray(_ value: Any?) throws -> [String] {
    guard let value else { return [] }
    guard let strings = value as? [String] else {
        throw OpenAIServerError.invalidJSON
    }
    return strings
}

private func anthropicFloatValue(_ value: Any?) -> Float? {
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

private func anthropicIntValue(_ value: Any?) -> Int? {
    switch value {
    case let value as Int:
        return value
    case let value as NSNumber:
        return value.intValue
    default:
        return nil
    }
}
