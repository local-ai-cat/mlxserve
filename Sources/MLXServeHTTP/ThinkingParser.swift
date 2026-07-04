import Foundation

private let openThinkTag = "<think>"
private let closeThinkTag = "</think>"
private let minimaxOpenThinkTag = "<mm:think>"
private let minimaxCloseThinkTag = "</mm:think>"

public func extractThinking(_ text: String) -> (reasoning: String, content: String) {
    guard !text.isEmpty else { return ("", "") }

    let normalized = text
        .replacingOccurrences(of: minimaxOpenThinkTag, with: openThinkTag)
        .replacingOccurrences(of: minimaxCloseThinkTag, with: closeThinkTag)

    var thinkingParts: [String] = []
    var remaining = normalized
    while let openRange = remaining.range(of: openThinkTag),
        let closeRange = remaining[openRange.upperBound...].range(of: closeThinkTag)
    {
        thinkingParts.append(String(remaining[openRange.upperBound..<closeRange.lowerBound]))
        remaining.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
    }

    if !thinkingParts.isEmpty {
        return (
            thinkingParts.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines),
            remaining.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    if normalized.contains(closeThinkTag) && !normalized.contains(openThinkTag),
        let closeRange = normalized.range(of: closeThinkTag)
    {
        return (
            String(normalized[..<closeRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines),
            String(normalized[closeRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    if normalized.contains(openThinkTag) && !normalized.contains(closeThinkTag),
        let openRange = normalized.range(of: openThinkTag)
    {
        let before = normalized[..<openRange.lowerBound]
        let after = normalized[openRange.upperBound...]
        return ("", String(before + after).trimmingCharacters(in: .whitespacesAndNewlines))
    }

    return ("", text)
}

public struct ThinkingParser: Sendable {
    private var inThinking: Bool
    private var buffer = ""
    private var closeSeen = false
    private var thinkingAccumulated: [String] = []
    private var contentEmitted = false

    public init(startInThinking: Bool = false) {
        self.inThinking = startInThinking
    }

    public mutating func feed(_ text: String) -> (reasoning: String, content: String) {
        guard !text.isEmpty else { return ("", "") }

        let text = buffer + text
        buffer = ""

        var thinkingOut = ""
        var contentOut = ""
        var index = text.startIndex

        while index < text.endIndex {
            if text[index] == "<" {
                let remaining = text[index...]

                if remaining.hasPrefix(openThinkTag) {
                    inThinking = true
                    index = text.index(index, offsetBy: openThinkTag.count)
                    continue
                }

                if remaining.hasPrefix(closeThinkTag) {
                    inThinking = false
                    closeSeen = true
                    index = text.index(index, offsetBy: closeThinkTag.count)
                    continue
                }

                if Self.couldBeTag(String(remaining)) {
                    buffer = String(remaining)
                    break
                }

                append("<", thinkingOut: &thinkingOut, contentOut: &contentOut)
                index = text.index(after: index)
            } else {
                append(String(text[index]), thinkingOut: &thinkingOut, contentOut: &contentOut)
                index = text.index(after: index)
            }
        }

        if !thinkingOut.isEmpty {
            thinkingAccumulated.append(thinkingOut)
        }
        if !contentOut.isEmpty {
            contentEmitted = true
        }
        return (thinkingOut, contentOut)
    }

    public mutating func finish() -> (reasoning: String, content: String) {
        let partial = buffer
        buffer = ""

        if inThinking && !closeSeen && !contentEmitted && !thinkingAccumulated.isEmpty {
            contentEmitted = true
            return ("", thinkingAccumulated.joined() + partial)
        }

        guard !partial.isEmpty else { return ("", "") }

        if inThinking {
            thinkingAccumulated.append(partial)
            return (partial, "")
        }

        contentEmitted = true
        return ("", partial)
    }

    private func append(_ value: String, thinkingOut: inout String, contentOut: inout String) {
        if inThinking {
            thinkingOut += value
        } else {
            contentOut += value
        }
    }

    private static func couldBeTag(_ text: String) -> Bool {
        guard text.count < closeThinkTag.count else { return false }
        return openThinkTag.hasPrefix(text) || closeThinkTag.hasPrefix(text)
    }
}
