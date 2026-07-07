import Foundation

/// Per-model tool-call output parser registry.
///
/// Different model families emit tool calls in their own native formats. A
/// single generic parser (`parseToolCalls`) only recognizes the Hermes/Qwen
/// `<tool_call>` form, the Llama `<|python_tag|>`/`[TOOL_CALLS]` prefixes, and
/// bare XML `<function=...>` — so a model that emits a tool call in another
/// native format finishes with `finish=stop, tools=0` (the call generated but
/// never recognized).
///
/// This registry mirrors omlx's `OutputParserFactory` / `detect_output_parser`
/// approach (`omlx/adapter/output_parser.py`): pick a parser by model *family*,
/// derived from the model id and, as a robustness fallback, from protocol
/// markers present in the raw output text. Each family parser is native
/// (it reads the model's own format) — no grammar-constrained decoding and no
/// strategy swap.
///
/// Adding a future family = one `case` in ``ToolCallModelFamily`` plus a
/// selection rule and a parse function.
public enum ToolCallModelFamily: String, Sendable, Equatable {
    /// Hermes/Qwen `<tool_call>`, Llama `<|python_tag|>`, bare XML `<function=>`,
    /// `[TOOL_CALLS]`, bare JSON — handled by the existing ``parseToolCalls``.
    case generic
    /// gpt-oss "harmony" — tool call carried in the `commentary` channel with a
    /// `to=functions.NAME` recipient (`omlx/adapter/harmony.py`).
    case harmony
    /// Gemma 4 — `<|tool_call>call:name{args}<tool_call|>` (`omlx/adapter/gemma4.py`).
    case gemma4
    /// DeepSeek — R1 native `<｜tool▁calls▁begin｜>…` and V4 DSML
    /// `<｜DSML｜tool_calls>…` (`omlx/patches/deepseek_v4/tool_parser_v4.py`).
    case deepseek
}

/// Reasoning, visible content, and tool calls extracted from one completion.
public struct ModelOutputParseResult: Sendable, Equatable {
    public let reasoning: String
    public let content: String
    public let toolCalls: [ParsedToolCall]

    public init(reasoning: String, content: String, toolCalls: [ParsedToolCall]) {
        self.reasoning = reasoning
        self.content = content
        self.toolCalls = toolCalls
    }
}

public func defaultToolCallID() -> String {
    "call_" + String(UUID().uuidString.prefix(8)).lowercased()
}

/// Select the tool-call family for a model by id, falling back to sniffing
/// protocol markers in the raw output. omlx keys off `model_type` from the
/// config; here the model id is the primary signal and the output markers are
/// a secondary one so a family parser still runs when the id is unrecognized
/// (e.g. an aliased or locally-renamed checkpoint).
public func toolCallModelFamily(forModel model: String, output text: String = "") -> ToolCallModelFamily {
    let id = model.lowercased()
    if id.contains("gpt-oss") || id.contains("gpt_oss") { return .harmony }
    if id.contains("gemma-4") || id.contains("gemma4") { return .gemma4 }
    if id.contains("deepseek") { return .deepseek }

    // Output-marker fallback: the model's native format is unambiguous even
    // when the id is not.
    if text.contains(harmonyChannelToken), text.contains(harmonyFunctionsRecipientPrefix) {
        return .harmony
    }
    if text.contains(deepseekR1CallsBegin) || text.contains(deepseekDSMLCallsOpen) {
        return .deepseek
    }
    if text.contains(gemma4ToolCallOpen), text.contains(gemma4CallPrefix) {
        return .gemma4
    }
    return .generic
}

/// Parse a full completion into reasoning/content/tool-calls using the family
/// parser for `model`. This is the family-aware replacement for the
/// `extractThinking(text)` + `parseToolCalls(from:)` pair at the response
/// paths.
///
/// - Parameter includeToolCalls: when `false` (tools not requested), tool
///   extraction is skipped and the result mirrors a plain `extractThinking`
///   split, exactly as the pre-registry code did.
public func parseModelOutput(
    _ text: String,
    model: String,
    includeToolCalls: Bool = true,
    idGenerator: () -> String = defaultToolCallID
) -> ModelOutputParseResult {
    switch toolCallModelFamily(forModel: model, output: text) {
    case .generic:
        return parseGenericOutput(text, includeToolCalls: includeToolCalls, idGenerator: idGenerator)
    case .harmony:
        return parseHarmonyOutput(text, includeToolCalls: includeToolCalls, idGenerator: idGenerator)
    case .gemma4:
        return parseGemma4Output(text, includeToolCalls: includeToolCalls, idGenerator: idGenerator)
    case .deepseek:
        return parseDeepSeekOutput(text, includeToolCalls: includeToolCalls, idGenerator: idGenerator)
    }
}

/// Streaming finish-time tool-call extraction for the Responses / Anthropic
/// dialects. Those paths already streamed reasoning/content deltas through the
/// (family-agnostic) incremental ``ThinkingParser``; at finish they need the
/// tool calls, which for non-generic families can only be recovered from the
/// raw text. Generic models keep the existing content-channel parse verbatim.
///
/// Known limitation: because the incremental split is not yet family-aware,
/// gpt-oss/gemma protocol markers may already have streamed inside the content
/// deltas during `feed`; this only corrects the final tool-call extraction and
/// the reconciled content, not those already-sent deltas.
public func streamingToolCallParse(
    model: String,
    rawText: String,
    bufferedContent: String,
    idGenerator: () -> String = defaultToolCallID
) -> ToolCallParseResult {
    switch toolCallModelFamily(forModel: model, output: rawText) {
    case .generic:
        return parseToolCalls(from: bufferedContent, idGenerator: idGenerator)
    case .harmony, .gemma4, .deepseek:
        let result = parseModelOutput(
            rawText,
            model: model,
            includeToolCalls: true,
            idGenerator: idGenerator
        )
        return ToolCallParseResult(content: result.content, toolCalls: result.toolCalls)
    }
}

// MARK: - Generic

private func parseGenericOutput(
    _ text: String,
    includeToolCalls: Bool,
    idGenerator: () -> String
) -> ModelOutputParseResult {
    let extracted = extractThinking(text)
    guard includeToolCalls else {
        return ModelOutputParseResult(
            reasoning: extracted.reasoning,
            content: extracted.content,
            toolCalls: []
        )
    }
    let parsed = parseToolCalls(from: extracted.content, idGenerator: idGenerator)
    return ModelOutputParseResult(
        reasoning: extracted.reasoning,
        content: parsed.content,
        toolCalls: parsed.toolCalls
    )
}

// MARK: - Harmony (gpt-oss)

let harmonyChannelToken = "<|channel|>"
private let harmonyMessageToken = "<|message|>"
private let harmonyBoundaryTokens = ["<|end|>", "<|return|>", "<|call|>", "<|start|>", harmonyChannelToken]
let harmonyFunctionsRecipientPrefix = "to=functions."

/// Walk harmony channels, routing `analysis`/`commentary` to reasoning and
/// `final` to content, but promoting any message with a `to=functions.NAME`
/// recipient to a tool call (its `<|message|>` body is the JSON arguments).
///
/// This is the omlx harmony behavior (`parse_tool_calls_from_tokens` +
/// channel routing) done on decoded text: the recipient header may sit either
/// in the role header (`<|start|>assistant to=functions.X<|channel|>…`) or in
/// the channel header (`<|channel|>commentary to=functions.X<|message|>…`), so
/// the whole header region ahead of `<|message|>` is searched.
private func parseHarmonyOutput(
    _ text: String,
    includeToolCalls: Bool,
    idGenerator: () -> String
) -> ModelOutputParseResult {
    var reasoning = ""
    var content = ""
    var toolCalls: [ParsedToolCall] = []

    var searchStart = text.startIndex
    while let channelRange = text.range(of: harmonyChannelToken, range: searchStart..<text.endIndex) {
        guard let messageRange = text.range(
            of: harmonyMessageToken,
            range: channelRange.upperBound..<text.endIndex
        ) else {
            break
        }

        let channelHeader = String(text[channelRange.upperBound..<messageRange.lowerBound])
        let channel = channelHeader
            .split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" })
            .first
            .map(String.init) ?? ""

        // The recipient can appear in the role header (before <|channel|>) or
        // in the channel header, so scan from the previous boundary through the
        // <|message|> marker.
        let headerRegion = String(text[searchStart..<messageRange.lowerBound])
        let recipient = harmonyFunctionsRecipient(in: headerRegion)

        let bodyStart = messageRange.upperBound
        let bodyEnd = harmonyBodyEnd(in: text, from: bodyStart)
        let body = String(text[bodyStart..<bodyEnd])

        if includeToolCalls, let name = recipient {
            toolCalls.append(
                ParsedToolCall(
                    id: idGenerator(),
                    name: name,
                    arguments: normalizedArgumentsJSON(from: body)
                )
            )
        } else if channel == "final" {
            content += body
        } else {
            reasoning += body
        }

        searchStart = bodyEnd
    }

    var result = ModelOutputParseResult(
        reasoning: reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
        content: content.trimmingCharacters(in: .whitespacesAndNewlines),
        toolCalls: toolCalls
    )

    // Fallback: a gpt-oss fine-tune that emitted a generic `<tool_call>` in the
    // final channel instead of a harmony recipient still gets parsed.
    if includeToolCalls, result.toolCalls.isEmpty {
        let parsed = parseToolCalls(from: result.content, idGenerator: idGenerator)
        if !parsed.toolCalls.isEmpty {
            result = ModelOutputParseResult(
                reasoning: result.reasoning,
                content: parsed.content,
                toolCalls: parsed.toolCalls
            )
        }
    }

    // No harmony markers at all: fall back to the generic split so a
    // misclassified model is never worse off than the default path.
    if toolCalls.isEmpty, reasoning.isEmpty, content.isEmpty, !text.isEmpty {
        return parseGenericOutput(text, includeToolCalls: includeToolCalls, idGenerator: idGenerator)
    }
    return result
}

private func harmonyFunctionsRecipient(in header: String) -> String? {
    guard let range = header.range(of: harmonyFunctionsRecipientPrefix) else {
        return nil
    }
    let tail = header[range.upperBound...]
    let name = tail.prefix { character in
        !(character == " " || character == "\t" || character == "\n" || character == "<")
    }
    return name.isEmpty ? nil : String(name)
}

private func harmonyBodyEnd(in text: String, from index: String.Index) -> String.Index {
    harmonyBoundaryTokens
        .compactMap { text.range(of: $0, range: index..<text.endIndex)?.lowerBound }
        .min() ?? text.endIndex
}

// MARK: - Gemma 4

let gemma4ToolCallOpen = "<|tool_call>"
private let gemma4ToolCallClose = "<tool_call|>"
let gemma4CallPrefix = "call:"
private let gemma4StringDelimiter = "<|\"|>"

/// Parse Gemma 4 `<|tool_call>call:name{args}<tool_call|>` blocks
/// (`omlx/adapter/gemma4.py` + `_parse_gemma4_tool_call_fallback`). The name may
/// be namespaced (`call:a:b:c{...}`); arguments may be strict JSON, bare
/// `key: value`, or `<|"|>`-delimited strings.
private func parseGemma4Output(
    _ text: String,
    includeToolCalls: Bool,
    idGenerator: () -> String
) -> ModelOutputParseResult {
    var toolCalls: [ParsedToolCall] = []
    var remaining = text

    if includeToolCalls {
        var cursor = remaining.startIndex
        var scanned = ""
        while let openRange = remaining.range(of: gemma4ToolCallOpen, range: cursor..<remaining.endIndex) {
            guard let closeRange = remaining.range(
                of: gemma4ToolCallClose,
                range: openRange.upperBound..<remaining.endIndex
            ) else {
                break
            }
            scanned += remaining[cursor..<openRange.lowerBound]
            let inner = String(remaining[openRange.upperBound..<closeRange.lowerBound])
            toolCalls.append(contentsOf: parseGemma4Calls(from: inner, idGenerator: idGenerator))
            cursor = closeRange.upperBound
        }
        if !toolCalls.isEmpty {
            scanned += remaining[cursor...]
            remaining = scanned
        }
    }

    // Strip any stray bare markers the model emits outside a well-formed block.
    remaining = remaining
        .replacingOccurrences(of: gemma4ToolCallOpen, with: "")
        .replacingOccurrences(of: gemma4ToolCallClose, with: "")

    let extracted = extractThinking(remaining)

    if includeToolCalls, toolCalls.isEmpty {
        // Gemma sometimes emits the bare `call:name{...}` without markers, or a
        // generic `<tool_call>`; recover via both paths.
        let bareCalls = parseGemma4Calls(from: extracted.content, idGenerator: idGenerator)
        if !bareCalls.isEmpty {
            return ModelOutputParseResult(
                reasoning: extracted.reasoning,
                content: strippedGemma4Calls(from: extracted.content),
                toolCalls: bareCalls
            )
        }
        let parsed = parseToolCalls(from: extracted.content, idGenerator: idGenerator)
        return ModelOutputParseResult(
            reasoning: extracted.reasoning,
            content: parsed.content,
            toolCalls: parsed.toolCalls
        )
    }

    return ModelOutputParseResult(
        reasoning: extracted.reasoning,
        content: extracted.content.trimmingCharacters(in: .whitespacesAndNewlines),
        toolCalls: toolCalls
    )
}

private func strippedGemma4Calls(from text: String) -> String {
    var output = ""
    var cursor = text.startIndex
    while let headRange = text.range(of: gemma4CallPrefix, range: cursor..<text.endIndex) {
        guard let braceIndex = text.range(of: "{", range: headRange.upperBound..<text.endIndex)?.lowerBound,
            let end = gemma4ArgumentSpanEnd(in: text, openBrace: braceIndex)
        else {
            break
        }
        output += text[cursor..<headRange.lowerBound]
        cursor = text.index(after: end)
    }
    output += text[cursor...]
    return output.trimmingCharacters(in: .whitespacesAndNewlines)
}

/// Parse one or more `call:name{args}` heads from marker-stripped text.
private func parseGemma4Calls(from text: String, idGenerator: () -> String) -> [ParsedToolCall] {
    var calls: [ParsedToolCall] = []
    var cursor = text.startIndex
    while let headRange = text.range(of: gemma4CallPrefix, range: cursor..<text.endIndex) {
        guard let braceIndex = text.range(of: "{", range: headRange.upperBound..<text.endIndex)?.lowerBound else {
            break
        }
        let name = String(text[headRange.upperBound..<braceIndex])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let end = gemma4ArgumentSpanEnd(in: text, openBrace: braceIndex) else {
            break
        }
        let argsBody = String(text[text.index(after: braceIndex)..<end])
        if !name.isEmpty {
            calls.append(
                ParsedToolCall(
                    id: idGenerator(),
                    name: name,
                    arguments: gemma4ArgumentsJSON(from: argsBody)
                )
            )
        }
        cursor = text.index(after: end)
    }
    return calls
}

/// Find the matching `}` for the `{` at `openBrace`, honoring nested braces and
/// double-quoted strings. The `<|"|>` delimiter opens with `"`, so a plain
/// double-quote scan already balances it correctly.
private func gemma4ArgumentSpanEnd(in text: String, openBrace: String.Index) -> String.Index? {
    var depth = 0
    var inString = false
    var escaped = false
    var index = openBrace
    while index < text.endIndex {
        let character = text[index]
        if inString {
            if escaped {
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == "\"" {
                inString = false
            }
        } else if character == "\"" {
            inString = true
        } else if character == "{" {
            depth += 1
        } else if character == "}" {
            depth -= 1
            if depth == 0 {
                return index
            }
        }
        index = text.index(after: index)
    }
    return nil
}

/// Transcode a Gemma 4 argument body to a JSON object string. Tries strict
/// JSON, then the `<|"|>`→`"` substitution, then a permissive `key: value`
/// split covering bare and delimited scalar values.
private func gemma4ArgumentsJSON(from body: String) -> String {
    let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty {
        return "{}"
    }

    let braced = "{\(trimmed)}"
    if let object = parseJSONObject(braced) {
        return serializeJSONObject(object)
    }

    let delimiterReplaced = braced.replacingOccurrences(of: gemma4StringDelimiter, with: "\"")
    if let object = parseJSONObject(delimiterReplaced) {
        return serializeJSONObject(object)
    }

    var arguments: [String: Any] = [:]
    for entry in splitTopLevel(trimmed, separator: ",") {
        guard let colon = topLevelColonIndex(in: entry) else { continue }
        let key = unquoteGemma4(String(entry[entry.startIndex..<colon])
            .trimmingCharacters(in: .whitespacesAndNewlines))
        let rawValue = String(entry[entry.index(after: colon)...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { continue }
        arguments[key] = gemma4Value(from: rawValue)
    }
    return serializeJSONObject(arguments)
}

private func gemma4Value(from raw: String) -> Any {
    if raw.hasPrefix(gemma4StringDelimiter), raw.hasSuffix(gemma4StringDelimiter),
        raw.count >= 2 * gemma4StringDelimiter.count {
        let start = raw.index(raw.startIndex, offsetBy: gemma4StringDelimiter.count)
        let end = raw.index(raw.endIndex, offsetBy: -gemma4StringDelimiter.count)
        return String(raw[start..<end])
    }
    if let fragment = parseJSONFragmentValue(raw) {
        return fragment
    }
    return unquoteGemma4(raw)
}

private func unquoteGemma4(_ text: String) -> String {
    var value = text
    if value.hasPrefix(gemma4StringDelimiter), value.hasSuffix(gemma4StringDelimiter),
        value.count >= 2 * gemma4StringDelimiter.count {
        let start = value.index(value.startIndex, offsetBy: gemma4StringDelimiter.count)
        let end = value.index(value.endIndex, offsetBy: -gemma4StringDelimiter.count)
        return String(value[start..<end])
    }
    if (value.hasPrefix("\"") && value.hasSuffix("\"") && value.count >= 2)
        || (value.hasPrefix("'") && value.hasSuffix("'") && value.count >= 2) {
        value.removeFirst()
        value.removeLast()
    }
    return value
}

// MARK: - DeepSeek (R1 native + V4 DSML)

// Full-width markers (U+FF5C `｜`, U+2581 `▁`) as emitted by DeepSeek templates.
let deepseekR1CallsBegin = "<｜tool▁calls▁begin｜>"
private let deepseekR1CallBegin = "<｜tool▁call▁begin｜>"
private let deepseekR1CallEnd = "<｜tool▁call▁end｜>"
private let deepseekR1Sep = "<｜tool▁sep｜>"
let deepseekDSMLCallsOpen = "<｜DSML｜tool_calls>"

/// Parse DeepSeek tool calls: the R1 native
/// `<｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\n{args}\n```<｜tool▁call▁end｜>`
/// form and the V4 DSML `<｜DSML｜invoke name="X">…` form
/// (`omlx/patches/deepseek_v4/tool_parser_v4.py`).
private func parseDeepSeekOutput(
    _ text: String,
    includeToolCalls: Bool,
    idGenerator: () -> String
) -> ModelOutputParseResult {
    let extracted = extractThinking(text)
    guard includeToolCalls else {
        return ModelOutputParseResult(
            reasoning: extracted.reasoning,
            content: extracted.content,
            toolCalls: []
        )
    }

    // DeepSeek carries the tool markers in the visible channel; parse from the
    // full text (post-think-strip) so both native forms are seen.
    let source = extracted.content
    if let calls = parseDeepSeekR1Calls(from: source, idGenerator: idGenerator), !calls.isEmpty {
        return ModelOutputParseResult(
            reasoning: extracted.reasoning,
            content: strippedRange(source, from: deepseekR1CallsBegin),
            toolCalls: calls
        )
    }
    if let calls = parseDeepSeekDSMLCalls(from: source, idGenerator: idGenerator), !calls.isEmpty {
        return ModelOutputParseResult(
            reasoning: extracted.reasoning,
            content: strippedRange(source, from: deepseekDSMLCallsOpen),
            toolCalls: calls
        )
    }

    let parsed = parseToolCalls(from: source, idGenerator: idGenerator)
    return ModelOutputParseResult(
        reasoning: extracted.reasoning,
        content: parsed.content,
        toolCalls: parsed.toolCalls
    )
}

private func parseDeepSeekR1Calls(from text: String, idGenerator: () -> String) -> [ParsedToolCall]? {
    guard text.contains(deepseekR1CallBegin) else { return nil }
    var calls: [ParsedToolCall] = []
    var cursor = text.startIndex
    while let beginRange = text.range(of: deepseekR1CallBegin, range: cursor..<text.endIndex) {
        let end = text.range(of: deepseekR1CallEnd, range: beginRange.upperBound..<text.endIndex)?.lowerBound
            ?? text.endIndex
        let block = String(text[beginRange.upperBound..<end])
        if let call = parseDeepSeekR1Block(block, idGenerator: idGenerator) {
            calls.append(call)
        }
        cursor = end
    }
    return calls.isEmpty ? nil : calls
}

/// A block is `function<｜tool▁sep｜>NAME\n```json\n{args}\n```` (the leading
/// `function` type and the ```` ```json ```` fence are both optional in the
/// wild, so name = text after the separator up to the first newline, and args
/// = the first `{…}`/```` ``` ```` fenced JSON found after it).
private func parseDeepSeekR1Block(_ block: String, idGenerator: () -> String) -> ParsedToolCall? {
    let afterSep: Substring
    if let sepRange = block.range(of: deepseekR1Sep) {
        afterSep = block[sepRange.upperBound...]
    } else {
        afterSep = block[...]
    }
    let nameEnd = afterSep.firstIndex(of: "\n") ?? afterSep.endIndex
    let name = afterSep[afterSep.startIndex..<nameEnd].trimmingCharacters(in: .whitespacesAndNewlines)
    guard !name.isEmpty else { return nil }

    let remainder = nameEnd < afterSep.endIndex ? String(afterSep[afterSep.index(after: nameEnd)...]) : ""
    let arguments = deepseekArgumentsJSON(from: remainder)
    return ParsedToolCall(id: idGenerator(), name: name, arguments: arguments)
}

private func deepseekArgumentsJSON(from remainder: String) -> String {
    var body = remainder.trimmingCharacters(in: .whitespacesAndNewlines)
    // Strip a ```json … ``` (or bare ``` … ```) fence if present.
    if body.hasPrefix("```") {
        if let firstNewline = body.firstIndex(of: "\n") {
            body = String(body[body.index(after: firstNewline)...])
        }
        if let fenceEnd = body.range(of: "```", options: .backwards) {
            body = String(body[body.startIndex..<fenceEnd.lowerBound])
        }
    }
    body = body.trimmingCharacters(in: .whitespacesAndNewlines)
    return normalizedArgumentsJSON(from: body)
}

/// V4 DSML: one or more `<｜DSML｜invoke name="X">…</｜DSML｜invoke>` blocks, each
/// with `<｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>`.
private func parseDeepSeekDSMLCalls(from text: String, idGenerator: () -> String) -> [ParsedToolCall]? {
    let invokeOpenPrefix = "<｜DSML｜invoke name=\""
    let invokeClose = "</｜DSML｜invoke>"
    guard text.contains(invokeOpenPrefix) else { return nil }

    var calls: [ParsedToolCall] = []
    var cursor = text.startIndex
    while let openRange = text.range(of: invokeOpenPrefix, range: cursor..<text.endIndex) {
        guard let nameEnd = text.range(of: "\"", range: openRange.upperBound..<text.endIndex)?.lowerBound,
            let headerEnd = text.range(of: ">", range: nameEnd..<text.endIndex)?.upperBound,
            let closeRange = text.range(of: invokeClose, range: headerEnd..<text.endIndex)
        else {
            break
        }
        let name = String(text[openRange.upperBound..<nameEnd])
        let body = String(text[headerEnd..<closeRange.lowerBound])
        if !name.isEmpty {
            calls.append(
                ParsedToolCall(
                    id: idGenerator(),
                    name: name,
                    arguments: deepseekDSMLArgumentsJSON(from: body)
                )
            )
        }
        cursor = closeRange.upperBound
    }
    return calls.isEmpty ? nil : calls
}

private func deepseekDSMLArgumentsJSON(from body: String) -> String {
    let paramOpen = "<｜DSML｜parameter name=\""
    let paramClose = "</｜DSML｜parameter>"
    var arguments: [String: Any] = [:]
    var cursor = body.startIndex
    while let openRange = body.range(of: paramOpen, range: cursor..<body.endIndex) {
        guard let keyEnd = body.range(of: "\"", range: openRange.upperBound..<body.endIndex)?.lowerBound,
            let headerEnd = body.range(of: ">", range: keyEnd..<body.endIndex)?.upperBound,
            let closeRange = body.range(of: paramClose, range: headerEnd..<body.endIndex)
        else {
            break
        }
        let key = String(body[openRange.upperBound..<keyEnd])
        let header = String(body[keyEnd..<headerEnd])
        let isString = header.contains("string=\"true\"")
        var rawValue = String(body[headerEnd..<closeRange.lowerBound])
        if rawValue.hasPrefix("\n") { rawValue.removeFirst() }
        if rawValue.hasSuffix("\n") { rawValue.removeLast() }
        if !key.isEmpty {
            if isString {
                arguments[key] = rawValue
            } else {
                arguments[key] = parseJSONFragmentValue(rawValue) ?? rawValue
            }
        }
        cursor = closeRange.upperBound
    }
    return serializeJSONObject(arguments)
}

// MARK: - Shared helpers

/// Return `text` up to the first occurrence of `marker`, trimmed — the visible
/// prose that precedes a tool-call block.
private func strippedRange(_ text: String, from marker: String) -> String {
    guard let range = text.range(of: marker) else {
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    return String(text[text.startIndex..<range.lowerBound])
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

/// Normalize a raw argument string to a compact JSON object. A valid JSON
/// object is re-serialized with sorted keys; anything else is returned as-is
/// (trimmed), falling back to `{}` when empty — mirroring the generic parser.
private func normalizedArgumentsJSON(from text: String) -> String {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return "{}" }
    if let object = parseJSONObject(trimmed) {
        return serializeJSONObject(object)
    }
    return trimmed
}

private func parseJSONObject(_ text: String) -> [String: Any]? {
    guard let data = text.data(using: .utf8), !data.isEmpty else { return nil }
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}

private func parseJSONFragmentValue(_ text: String) -> Any? {
    guard let data = text.data(using: .utf8), !data.isEmpty else { return nil }
    return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
}

private func serializeJSONObject(_ object: [String: Any]) -> String {
    guard JSONSerialization.isValidJSONObject(object),
        let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
        let string = String(data: data, encoding: .utf8)
    else {
        return "{}"
    }
    return string
}

/// Split `text` on `separator` at brace/bracket/quote depth 0.
private func splitTopLevel(_ text: String, separator: Character) -> [String] {
    var parts: [String] = []
    var current = ""
    var depth = 0
    var inString = false
    var stringDelimiter: Character = "\""
    var index = text.startIndex
    while index < text.endIndex {
        let character = text[index]
        if inString {
            if character == stringDelimiter { inString = false }
            current.append(character)
        } else if character == "\"" || character == "'" {
            inString = true
            stringDelimiter = character
            current.append(character)
        } else if character == "{" || character == "[" {
            depth += 1
            current.append(character)
        } else if character == "}" || character == "]" {
            depth -= 1
            current.append(character)
        } else if character == separator, depth == 0 {
            parts.append(current)
            current = ""
        } else {
            current.append(character)
        }
        index = text.index(after: index)
    }
    if !current.isEmpty {
        parts.append(current)
    }
    return parts
}

private func topLevelColonIndex(in text: String) -> String.Index? {
    var depth = 0
    var inString = false
    var stringDelimiter: Character = "\""
    var index = text.startIndex
    while index < text.endIndex {
        let character = text[index]
        if inString {
            if character == stringDelimiter { inString = false }
        } else if character == "\"" || character == "'" {
            inString = true
            stringDelimiter = character
        } else if character == "{" || character == "[" {
            depth += 1
        } else if character == "}" || character == "]" {
            depth -= 1
        } else if character == ":", depth == 0 {
            return index
        }
        index = text.index(after: index)
    }
    return nil
}
