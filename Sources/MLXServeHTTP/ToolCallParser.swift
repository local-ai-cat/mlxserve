import Foundation

private let toolCallOpenTag = "<tool_call>"
private let toolCallCloseTag = "</tool_call>"
private let llamaPythonTagPrefix = "<|python_tag|>"
private let toolCallsPrefix = "[TOOL_CALLS]"

public struct ParsedToolCall: Sendable, Equatable {
    public let id: String
    public let name: String
    public let arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

public struct ToolCallParseResult: Sendable, Equatable {
    public let content: String
    public let toolCalls: [ParsedToolCall]

    public init(content: String, toolCalls: [ParsedToolCall]) {
        self.content = content
        self.toolCalls = toolCalls
    }
}

public func parseToolCalls(
    from text: String,
    idGenerator: () -> String = { "call_" + String(UUID().uuidString.prefix(8)).lowercased() }
) -> ToolCallParseResult {
    if text.contains(toolCallOpenTag) {
        return parseTaggedToolCalls(from: text, idGenerator: idGenerator)
    }

    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.hasPrefix(llamaPythonTagPrefix) {
        let remainder = String(trimmed.dropFirst(llamaPythonTagPrefix.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return parseBareJSONToolCalls(from: remainder, originalText: text, idGenerator: idGenerator)
    }
    if trimmed.hasPrefix(toolCallsPrefix) {
        let remainder = String(trimmed.dropFirst(toolCallsPrefix.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return parseBareJSONToolCalls(from: remainder, originalText: text, idGenerator: idGenerator)
    }

    return parseBareJSONToolCalls(from: trimmed, originalText: text, idGenerator: idGenerator)
}

private func parseTaggedToolCalls(
    from text: String,
    idGenerator: () -> String
) -> ToolCallParseResult {
    var cursor = text.startIndex
    var contentParts: [String] = []
    var toolCalls: [ParsedToolCall] = []

    while let openRange = text[cursor...].range(of: toolCallOpenTag) {
        guard let closeRange = text[openRange.upperBound...].range(of: toolCallCloseTag) else {
            contentParts.append(String(text[cursor...]))
            cursor = text.endIndex
            break
        }

        contentParts.append(String(text[cursor..<openRange.lowerBound]))

        let innerText = String(text[openRange.upperBound..<closeRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let toolCall = parseToolCallObject(from: innerText, idGenerator: idGenerator) {
            toolCalls.append(toolCall)
        } else {
            contentParts.append(String(text[openRange.lowerBound..<closeRange.upperBound]))
        }

        cursor = closeRange.upperBound
    }

    if cursor < text.endIndex {
        contentParts.append(String(text[cursor...]))
    }

    guard !toolCalls.isEmpty else {
        return ToolCallParseResult(content: text, toolCalls: [])
    }

    return ToolCallParseResult(
        content: contentParts.joined().trimmingCharacters(in: .whitespacesAndNewlines),
        toolCalls: toolCalls
    )
}

private func parseBareJSONToolCalls(
    from text: String,
    originalText: String,
    idGenerator: () -> String
) -> ToolCallParseResult {
    guard let value = parseJSONValue(from: text) else {
        return ToolCallParseResult(content: originalText, toolCalls: [])
    }

    if let object = value as? [String: Any],
        let payload = parseToolCallPayload(from: object)
    {
        return ToolCallParseResult(content: "", toolCalls: [payload.toolCall(id: idGenerator())])
    }

    if let array = value as? [[String: Any]], !array.isEmpty {
        let payloads = array.compactMap(parseToolCallPayload)
        if payloads.count == array.count {
            let toolCalls = payloads.map { $0.toolCall(id: idGenerator()) }
            return ToolCallParseResult(content: "", toolCalls: toolCalls)
        }
    }

    return ToolCallParseResult(content: originalText, toolCalls: [])
}

private func parseToolCallObject(
    from text: String,
    idGenerator: () -> String
) -> ParsedToolCall? {
    guard let object = parseJSONValue(from: text) as? [String: Any] else {
        return nil
    }
    return parseToolCallObject(from: object, idGenerator: idGenerator)
}

private func parseToolCallObject(
    from object: [String: Any],
    idGenerator: () -> String
) -> ParsedToolCall? {
    guard let payload = parseToolCallPayload(from: object) else {
        return nil
    }
    return payload.toolCall(id: idGenerator())
}

private struct ToolCallPayload {
    let name: String
    let arguments: String

    func toolCall(id: String) -> ParsedToolCall {
        ParsedToolCall(id: id, name: name, arguments: arguments)
    }
}

private func parseToolCallPayload(from object: [String: Any]) -> ToolCallPayload? {
    guard let name = object["name"] as? String, !name.isEmpty else {
        return nil
    }

    let arguments = argumentsJSONString(from: object["arguments"]) ?? "{}"
    return ToolCallPayload(name: name, arguments: arguments)
}

private func argumentsJSONString(from value: Any?) -> String? {
    guard let value else {
        return "{}"
    }
    if let string = value as? String {
        return string
    }
    guard JSONSerialization.isValidJSONObject(value),
        let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
        let string = String(data: data, encoding: .utf8)
    else {
        return nil
    }
    return string
}

private func parseJSONValue(from text: String) -> Any? {
    guard let data = text.data(using: .utf8), !data.isEmpty else {
        return nil
    }
    return try? JSONSerialization.jsonObject(with: data)
}
