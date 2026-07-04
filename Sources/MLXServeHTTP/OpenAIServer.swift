import Foundation
import Network

public struct OpenAIModelInfo: Sendable {
    public let id: String
    public let maxModelLength: Int?

    public init(id: String, maxModelLength: Int? = nil) {
        self.id = id
        self.maxModelLength = maxModelLength
    }
}

public struct OpenAIChatMessage: Sendable {
    public let role: String
    public let content: String
    public let reasoningContent: String?

    public init(role: String, content: String, reasoningContent: String? = nil) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
    }
}

public enum OpenAIJSONValue: Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: OpenAIJSONValue])
    case array([OpenAIJSONValue])
    case null

    init?(_ value: Any) {
        switch value {
        case let string as String:
            self = .string(string)
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                self = .bool(number.boolValue)
            } else {
                self = .number(number.doubleValue)
            }
        case let object as [String: Any]:
            var converted: [String: OpenAIJSONValue] = [:]
            for (key, value) in object {
                guard let jsonValue = OpenAIJSONValue(value) else { return nil }
                converted[key] = jsonValue
            }
            self = .object(converted)
        case let array as [Any]:
            var converted: [OpenAIJSONValue] = []
            for value in array {
                guard let jsonValue = OpenAIJSONValue(value) else { return nil }
                converted.append(jsonValue)
            }
            self = .array(converted)
        case _ as NSNull:
            self = .null
        default:
            return nil
        }
    }
}

public struct OpenAIChatRequest: Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let repetitionPenalty: Float
    public let minP: Float
    public let xtcProbability: Float
    public let xtcThreshold: Float
    public let presencePenalty: Float
    public let frequencyPenalty: Float
    public let stop: [String]
    public let seed: Int?
    public let stream: Bool
    public let includeUsage: Bool
    public let enableThinking: Bool?
    public let chatTemplateKwargs: [String: OpenAIJSONValue]?

    public init(
        model: String,
        messages: [OpenAIChatMessage],
        maxTokens: Int,
        temperature: Float = 0,
        topP: Float = 0,
        topK: Int = 0,
        repetitionPenalty: Float = 1,
        minP: Float = 0,
        xtcProbability: Float = 0,
        xtcThreshold: Float = 0.1,
        presencePenalty: Float = 0,
        frequencyPenalty: Float = 0,
        stop: [String] = [],
        seed: Int? = nil,
        stream: Bool = false,
        includeUsage: Bool = false,
        enableThinking: Bool? = nil,
        chatTemplateKwargs: [String: OpenAIJSONValue]? = nil
    ) {
        self.model = model
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.minP = minP
        self.xtcProbability = xtcProbability
        self.xtcThreshold = xtcThreshold
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.stop = stop
        self.seed = seed
        self.stream = stream
        self.includeUsage = includeUsage
        self.enableThinking = enableThinking
        self.chatTemplateKwargs = chatTemplateKwargs
    }
}

public struct OpenAIChatChunk: Sendable {
    public let text: String
    public let tokenID: Int
    public let finishReason: String?

    public init(text: String, tokenID: Int, finishReason: String? = nil) {
        self.text = text
        self.tokenID = tokenID
        self.finishReason = finishReason
    }
}

public struct OpenAIChatStream: Sendable {
    public let promptTokens: Int
    public let chunks: AsyncThrowingStream<OpenAIChatChunk, Error>

    public init(promptTokens: Int, chunks: AsyncThrowingStream<OpenAIChatChunk, Error>) {
        self.promptTokens = promptTokens
        self.chunks = chunks
    }
}

public struct StopSequenceMatch: Sendable, Equatable {
    public let text: String
    public let stopped: Bool

    public init(text: String, stopped: Bool) {
        self.text = text
        self.stopped = stopped
    }
}

public struct StreamingStopSequenceMatcher: Sendable {
    private let stopSequences: [String]
    private let maxTailCount: Int
    private var pending = ""

    public init(stopSequences: [String]) {
        let stopSequences = stopSequences.filter { !$0.isEmpty }
        self.stopSequences = stopSequences
        self.maxTailCount = max((stopSequences.map(\.count).max() ?? 0) - 1, 0)
    }

    public mutating func feed(_ text: String) -> StopSequenceMatch {
        guard !stopSequences.isEmpty else {
            return StopSequenceMatch(text: text, stopped: false)
        }

        let combined = pending + text
        if let range = firstStopRange(in: combined, stopSequences: stopSequences) {
            pending = ""
            return StopSequenceMatch(text: String(combined[..<range.lowerBound]), stopped: true)
        }

        let tailCount = min(maxTailCount, combined.count)
        let emitEnd = combined.index(combined.endIndex, offsetBy: -tailCount)
        let emitted = String(combined[..<emitEnd])
        pending = String(combined[emitEnd...])
        return StopSequenceMatch(text: emitted, stopped: false)
    }

    public mutating func finish() -> StopSequenceMatch {
        let text = pending
        pending = ""
        guard let range = firstStopRange(in: text, stopSequences: stopSequences) else {
            return StopSequenceMatch(text: text, stopped: false)
        }
        return StopSequenceMatch(text: String(text[..<range.lowerBound]), stopped: true)
    }
}

public func truncateAtStop(_ text: String, stopSequences: [String]) -> StopSequenceMatch {
    let stopSequences = stopSequences.filter { !$0.isEmpty }
    guard let range = firstStopRange(in: text, stopSequences: stopSequences) else {
        return StopSequenceMatch(text: text, stopped: false)
    }
    return StopSequenceMatch(text: String(text[..<range.lowerBound]), stopped: true)
}

private func firstStopRange(in text: String, stopSequences: [String]) -> Range<String.Index>? {
    var bestRange: Range<String.Index>?
    for stop in stopSequences {
        guard let range = text.range(of: stop) else { continue }
        if let current = bestRange {
            if range.lowerBound < current.lowerBound {
                bestRange = range
            }
        } else {
            bestRange = range
        }
    }
    return bestRange
}

public protocol OpenAIChatBackend: Sendable {
    var models: [OpenAIModelInfo] { get }
    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream
}

private final class ListenerStartState: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?

    init(_ continuation: CheckedContinuation<Void, Error>) {
        self.continuation = continuation
    }

    func resume(_ result: Result<Void, Error>) {
        lock.lock()
        let continuation = continuation
        self.continuation = nil
        lock.unlock()

        guard let continuation else { return }
        switch result {
        case .success:
            continuation.resume(returning: ())
        case .failure(let error):
            continuation.resume(throwing: error)
        }
    }
}

public final class OpenAIServer: @unchecked Sendable {
    private let host: NWEndpoint.Host
    private let port: NWEndpoint.Port
    private let backend: any OpenAIChatBackend
    private let listener: NWListener
    private let responsesStore: ResponsesStore

    public init(
        host: String = "127.0.0.1",
        port: UInt16,
        backend: any OpenAIChatBackend
    ) throws {
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            throw OpenAIServerError.invalidPort(port)
        }
        self.host = NWEndpoint.Host(host)
        self.port = nwPort
        self.backend = backend
        self.listener = try NWListener(using: .tcp, on: nwPort)
        self.responsesStore = ResponsesStore()
    }

    public func start() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let startState = ListenerStartState(continuation)
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    startState.resume(.success(()))
                case .failed(let error):
                    startState.resume(.failure(OpenAIServerError.listenerFailed(String(describing: error))))
                case .cancelled:
                    startState.resume(.failure(OpenAIServerError.listenerFailed("listener cancelled")))
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                guard let self else {
                    connection.cancel()
                    return
                }
                connection.start(queue: .global(qos: .userInitiated))
                Task {
                    await self.handle(connection)
                }
            }
            listener.start(queue: .global(qos: .userInitiated))
        }
    }

    public func waitForever() {
        DispatchSemaphore(value: 0).wait()
    }

    private func handle(_ connection: NWConnection) async {
        do {
            let request = try await HTTPRequest.read(from: connection)
            switch (request.method, request.path) {
            case ("GET", "/v1/models"):
                try await sendJSON(modelsResponse(), status: 200, connection: connection)
            case ("POST", "/v1/chat/completions"):
                try await handleChatCompletion(request, connection: connection)
            case ("POST", "/v1/messages"):
                try await AnthropicMessagesHandler(backend: backend).handleMessages(request, connection: connection)
            case ("POST", "/v1/messages/count_tokens"):
                try await AnthropicMessagesHandler(backend: backend).handleCountTokens(request, connection: connection)
            case ("POST", "/v1/responses"):
                try await ResponsesHandler(backend: backend, store: responsesStore).handleCreate(request, connection: connection)
            default:
                if request.method == "GET", let id = responseID(from: request.path) {
                    try await ResponsesHandler(backend: backend, store: responsesStore).handleGet(id: id, connection: connection)
                    return
                }
                if request.method == "DELETE", let id = responseID(from: request.path) {
                    try await ResponsesHandler(backend: backend, store: responsesStore).handleDelete(id: id, connection: connection)
                    return
                }
                try await sendJSON(openAIErrorBody(message: "not found", status: 404), status: 404, connection: connection)
            }
        } catch {
            try? await sendJSON(
                openAIErrorBody(message: String(describing: error), status: 500),
                status: 500,
                connection: connection
            )
        }
    }

    private func responseID(from path: String) -> String? {
        let prefix = "/v1/responses/"
        guard path.hasPrefix(prefix) else { return nil }
        let id = String(path.dropFirst(prefix.count))
        return id.isEmpty ? nil : id
    }

    private func handleChatCompletion(_ request: HTTPRequest, connection: NWConnection) async throws {
        let chatRequest: OpenAIChatRequest
        do {
            chatRequest = try OpenAIChatRequest.parse(request.body)
        } catch {
            let status = (error as? OpenAIServerError)?.httpStatus ?? 422
            try await sendJSON(
                openAIErrorBody(message: String(describing: error), status: status),
                status: status,
                connection: connection
            )
            return
        }
        let started = DispatchTime.now().uptimeNanoseconds
        let stream = try await backend.startChatCompletion(chatRequest)

        if chatRequest.stream {
            try await sendStreamingChat(
                request: chatRequest,
                stream: stream,
                started: started,
                connection: connection
            )
        } else {
            try await sendBufferedChat(
                request: chatRequest,
                stream: stream,
                started: started,
                connection: connection
            )
        }
    }

    private func sendStreamingChat(
        request: OpenAIChatRequest,
        stream: OpenAIChatStream,
        started: UInt64,
        connection: NWConnection
    ) async throws {
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let created = Int(Date().timeIntervalSince1970)
        var firstToken: UInt64?
        var lastToken: UInt64?
        var completionTokens = 0
        var finishReason = "length"
        // Stop matching runs on raw decoded model text before thinking tags are split,
        // matching omlx's output_text-level stop behavior.
        var stopMatcher = StreamingStopSequenceMatcher(stopSequences: request.stop)
        var thinkingParser = ThinkingParser()
        var stoppedByTextStop = false

        try await connection.send(
            data: Data(
                (
                    "HTTP/1.1 200 OK\r\n"
                        + "Content-Type: text/event-stream\r\n"
                        + "Cache-Control: no-cache\r\n"
                        + "Connection: close\r\n"
                        + "X-Accel-Buffering: no\r\n"
                        + "\r\n"
                ).utf8
            )
        )
        try await sendSSE(
            [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    [
                        "index": 0,
                        "delta": ["role": "assistant"],
                        "finish_reason": NSNull(),
                    ]
                ],
            ],
            connection: connection
        )

        do {
            for try await chunk in stream.chunks {
                let now = DispatchTime.now().uptimeNanoseconds
                if firstToken == nil {
                    firstToken = now
                }
                lastToken = now
                completionTokens += 1
                if let chunkFinishReason = chunk.finishReason {
                    finishReason = chunkFinishReason
                }
                let stopMatch = stopMatcher.feed(chunk.text)
                if stopMatch.stopped {
                    finishReason = "stop"
                    stoppedByTextStop = true
                }
                if !stopMatch.text.isEmpty {
                    try await sendParserDelta(
                        thinkingParser.feed(stopMatch.text),
                        id: id,
                        created: created,
                        model: request.model,
                        connection: connection
                    )
                }
                if stoppedByTextStop {
                    break
                }
            }
        } catch {
            try await sendSSE(
                [
                    "id": id,
                    "object": "error",
                    "created": created,
                    "error": openAIErrorObject(message: String(describing: error), status: 500),
                ],
                connection: connection
            )
            try await connection.sendFinal(data: Data("data: [DONE]\n\n".utf8))
            return
        }

        if !stoppedByTextStop {
            let stopMatch = stopMatcher.finish()
            if stopMatch.stopped {
                finishReason = "stop"
                stoppedByTextStop = true
            }
            if !stopMatch.text.isEmpty {
                try await sendParserDelta(
                    thinkingParser.feed(stopMatch.text),
                    id: id,
                    created: created,
                    model: request.model,
                    connection: connection
                )
            }
        }
        try await sendParserDelta(
            thinkingParser.finish(),
            id: id,
            created: created,
            model: request.model,
            connection: connection
        )

        let ended = DispatchTime.now().uptimeNanoseconds
        try await sendSSE(
            [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    [
                        "index": 0,
                        "delta": [:],
                        "finish_reason": finishReason,
                    ]
                ],
            ],
            connection: connection
        )
        if request.includeUsage {
            try await sendSSE(
                [
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [],
                    "usage": usage(
                        promptTokens: stream.promptTokens,
                        completionTokens: completionTokens,
                        started: started,
                        firstToken: firstToken ?? ended,
                        lastToken: lastToken ?? ended,
                        ended: ended
                    ),
                ],
                connection: connection
            )
        }
        try await connection.sendFinal(data: Data("data: [DONE]\n\n".utf8))
    }

    private func sendBufferedChat(
        request: OpenAIChatRequest,
        stream: OpenAIChatStream,
        started: UInt64,
        connection: NWConnection
    ) async throws {
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        var firstToken: UInt64?
        var lastToken: UInt64?
        var completionTokens = 0
        var finishReason = "length"
        var text = ""
        // Stop matching runs on raw decoded model text before thinking tags are split,
        // matching omlx's output_text-level stop behavior.
        var stopMatcher = StreamingStopSequenceMatcher(stopSequences: request.stop)
        var stoppedByTextStop = false

        for try await chunk in stream.chunks {
            let now = DispatchTime.now().uptimeNanoseconds
            if firstToken == nil {
                firstToken = now
            }
            lastToken = now
            completionTokens += 1
            if let chunkFinishReason = chunk.finishReason {
                finishReason = chunkFinishReason
            }
            let stopMatch = stopMatcher.feed(chunk.text)
            text += stopMatch.text
            if stopMatch.stopped {
                finishReason = "stop"
                stoppedByTextStop = true
                break
            }
        }
        if !stoppedByTextStop {
            let stopMatch = stopMatcher.finish()
            text += stopMatch.text
            if stopMatch.stopped {
                finishReason = "stop"
            }
        }

        let extracted = extractThinking(text)
        var message: [String: Any] = [
            "role": "assistant",
            "content": extracted.content,
        ]
        if !extracted.reasoning.isEmpty {
            message["reasoning_content"] = extracted.reasoning
        }

        let ended = DispatchTime.now().uptimeNanoseconds
        try await sendJSON(
            [
                "id": id,
                "object": "chat.completion",
                "created": Int(Date().timeIntervalSince1970),
                "model": request.model,
                "choices": [
                    [
                        "index": 0,
                        "message": message,
                        "finish_reason": finishReason,
                    ]
                ],
                "usage": usage(
                    promptTokens: stream.promptTokens,
                    completionTokens: completionTokens,
                    started: started,
                    firstToken: firstToken ?? ended,
                    lastToken: lastToken ?? ended,
                    ended: ended
                ),
            ],
            status: 200,
            connection: connection
        )
    }

    private func modelsResponse() -> [String: Any] {
        [
            "object": "list",
            "data": backend.models.map { model in
                var payload: [String: Any] = [
                    "id": model.id,
                    "object": "model",
                    "created": Int(Date().timeIntervalSince1970),
                    "owned_by": "mlxserve-native",
                ]
                if let maxModelLength = model.maxModelLength {
                    payload["max_model_len"] = maxModelLength
                }
                return payload
            },
        ]
    }

    private func usage(
        promptTokens: Int,
        completionTokens: Int,
        started: UInt64,
        firstToken: UInt64,
        lastToken: UInt64,
        ended: UInt64
    ) -> [String: Any] {
        let ttft = seconds(firstToken - started)
        let total = seconds(ended - started)
        let generation = max(seconds((lastToken >= firstToken ? lastToken - firstToken : 0)), 1e-9)
        return [
            "prompt_tokens": promptTokens,
            "completion_tokens": completionTokens,
            "total_tokens": promptTokens + completionTokens,
            "input_tokens": promptTokens,
            "output_tokens": completionTokens,
            "prompt_tokens_details": ["cached_tokens": 0],
            "time_to_first_token": ttft,
            "total_time": total,
            "prompt_eval_duration": ttft,
            "generation_duration": generation,
            "prompt_tokens_per_second": Double(promptTokens) / max(ttft, 1e-9),
            "generation_tokens_per_second": Double(completionTokens) / generation,
        ]
    }

    private func seconds(_ nanoseconds: UInt64) -> Double {
        Double(nanoseconds) / 1_000_000_000
    }

    private func sendSSE(_ object: [String: Any], connection: NWConnection) async throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        let line = "data: \(String(decoding: data, as: UTF8.self))\n\n"
        try await connection.send(data: Data(line.utf8))
    }

    private func sendParserDelta(
        _ delta: (reasoning: String, content: String),
        id: String,
        created: Int,
        model: String,
        connection: NWConnection
    ) async throws {
        var payload: [String: Any] = [:]
        if !delta.reasoning.isEmpty {
            payload["reasoning_content"] = delta.reasoning
        }
        if !delta.content.isEmpty {
            payload["content"] = delta.content
        }
        guard !payload.isEmpty else { return }

        try await sendSSE(
            [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    [
                        "index": 0,
                        "delta": payload,
                        "finish_reason": NSNull(),
                    ]
                ],
            ],
            connection: connection
        )
    }

    private func sendJSON(
        _ object: [String: Any],
        status: Int,
        connection: NWConnection
    ) async throws {
        let body = try JSONSerialization.data(withJSONObject: object, options: [])
        let reason = httpReasonPhrase(status)
        let header = "HTTP/1.1 \(status) \(reason)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.count)\r\n"
            + "Connection: close\r\n"
            + "\r\n"
        try await connection.sendFinal(data: Data(header.utf8) + body)
    }
}

private func httpReasonPhrase(_ status: Int) -> String {
    switch status {
    case 200:
        return "OK"
    case 400:
        return "Bad Request"
    case 401:
        return "Unauthorized"
    case 404:
        return "Not Found"
    case 409:
        return "Conflict"
    case 413:
        return "Payload Too Large"
    case 422:
        return "Unprocessable Entity"
    case 429:
        return "Too Many Requests"
    case 503:
        return "Service Unavailable"
    case 507:
        return "Insufficient Storage"
    case 500:
        return "Internal Server Error"
    default:
        return "OK"
    }
}

public enum OpenAIServerError: Error, Equatable, CustomStringConvertible {
    case invalidPort(UInt16)
    case invalidRequest
    case invalidJSON
    case missingField(String)
    case unsupportedContent
    case invalidContentLength
    case payloadTooLarge
    case listenerFailed(String)

    var httpStatus: Int {
        switch self {
        case .invalidJSON, .missingField:
            return 422
        case .payloadTooLarge:
            return 413
        case .invalidRequest, .unsupportedContent, .invalidContentLength:
            return 400
        case .invalidPort, .listenerFailed:
            return 500
        }
    }

    public var description: String {
        switch self {
        case .invalidPort(let port):
            return "invalid port: \(port)"
        case .invalidRequest:
            return "invalid HTTP request"
        case .invalidJSON:
            return "invalid JSON request body"
        case .missingField(let field):
            return "missing required field: \(field)"
        case .unsupportedContent:
            return "unsupported request content"
        case .invalidContentLength:
            return "invalid Content-Length"
        case .payloadTooLarge:
            return "HTTP payload too large"
        case .listenerFailed(let message):
            return "listener failed: \(message)"
        }
    }
}

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    private static let maxHeaderBytes = 64 * 1024
    private static let maxBodyBytes = 16 * 1024 * 1024

    static func read(from connection: NWConnection) async throws -> HTTPRequest {
        var buffer = Data()
        var headerEnd: Range<Data.Index>?
        while headerEnd == nil {
            let chunk = try await connection.receive()
            guard !chunk.isEmpty else { throw OpenAIServerError.invalidRequest }
            buffer.append(chunk)
            guard buffer.count <= maxHeaderBytes else {
                throw OpenAIServerError.payloadTooLarge
            }
            headerEnd = buffer.range(of: Data("\r\n\r\n".utf8))
        }

        let request = try parseComplete(buffer)
        var body = request.body
        let contentLength = try parsedContentLength(request.headers)
        while body.count < contentLength {
            let chunk = try await connection.receive()
            guard !chunk.isEmpty else { throw OpenAIServerError.invalidRequest }
            body.append(chunk)
        }
        if body.count > contentLength {
            body = body.prefix(contentLength)
        }

        return HTTPRequest(
            method: request.method,
            path: request.path,
            headers: request.headers,
            body: body
        )
    }

    static func parseComplete(_ buffer: Data) throws -> HTTPRequest {
        guard let headerRange = buffer.range(of: Data("\r\n\r\n".utf8)) else {
            throw OpenAIServerError.invalidRequest
        }
        let headerData = buffer[..<headerRange.lowerBound]
        guard let headerText = String(data: headerData, encoding: .utf8) else {
            throw OpenAIServerError.invalidRequest
        }
        let lines = headerText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { throw OpenAIServerError.invalidRequest }
        let requestParts = requestLine.split(separator: " ", maxSplits: 2).map(String.init)
        guard requestParts.count >= 2 else { throw OpenAIServerError.invalidRequest }

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            let parts = line.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { continue }
            headers[parts[0].lowercased()] = parts[1].trimmingCharacters(in: .whitespaces)
        }

        let contentLength = try parsedContentLength(headers)
        let bodyStart = headerRange.upperBound
        var body = Data(buffer[bodyStart...])
        if body.count > contentLength {
            body = body.prefix(contentLength)
        }

        return HTTPRequest(
            method: requestParts[0],
            path: requestParts[1].split(separator: "?", maxSplits: 1).first.map(String.init) ?? requestParts[1],
            headers: headers,
            body: body
        )
    }

    private static func parsedContentLength(_ headers: [String: String]) throws -> Int {
        guard let contentLength = Int(headers["content-length"] ?? "0"),
            contentLength >= 0
        else {
            throw OpenAIServerError.invalidContentLength
        }
        guard contentLength <= maxBodyBytes else {
            throw OpenAIServerError.payloadTooLarge
        }
        return contentLength
    }
}

public extension OpenAIChatRequest {
    static func parse(_ body: Data) throws -> OpenAIChatRequest {
        let rawObject: Any
        do {
            rawObject = try JSONSerialization.jsonObject(with: body)
        } catch {
            throw OpenAIServerError.invalidJSON
        }

        guard let object = rawObject as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }
        guard let model = object["model"] as? String, !model.isEmpty else {
            throw OpenAIServerError.missingField("model")
        }
        guard let rawMessages = object["messages"] as? [[String: Any]] else {
            throw OpenAIServerError.missingField("messages")
        }

        let messages = rawMessages.compactMap { raw -> OpenAIChatMessage? in
            guard let role = raw["role"] as? String else { return nil }
            let reasoningContent = raw["reasoning_content"] as? String
            if let content = raw["content"] as? String {
                return OpenAIChatMessage(role: role, content: content, reasoningContent: reasoningContent)
            }
            if let parts = raw["content"] as? [[String: Any]] {
                let text = parts.compactMap { part in
                    part["type"] as? String == "text" ? part["text"] as? String : nil
                }.joined()
                return OpenAIChatMessage(role: role, content: text, reasoningContent: reasoningContent)
            }
            return nil
        }
        guard !messages.isEmpty else { throw OpenAIServerError.missingField("messages") }
        let streamOptions = object["stream_options"] as? [String: Any]
        let chatTemplateKwargs = try parseChatTemplateKwargs(object["chat_template_kwargs"])

        return OpenAIChatRequest(
            model: model,
            messages: messages,
            maxTokens: intValue(object["max_tokens"] ?? object["max_completion_tokens"]) ?? 16,
            temperature: floatValue(object["temperature"]) ?? 0,
            topP: floatValue(object["top_p"]) ?? 0,
            topK: intValue(object["top_k"]) ?? 0,
            repetitionPenalty: floatValue(object["repetition_penalty"]) ?? 1,
            minP: floatValue(object["min_p"]) ?? 0,
            xtcProbability: floatValue(object["xtc_probability"]) ?? 0,
            xtcThreshold: floatValue(object["xtc_threshold"]) ?? 0.1,
            presencePenalty: floatValue(object["presence_penalty"]) ?? 0,
            frequencyPenalty: floatValue(object["frequency_penalty"]) ?? 0,
            stop: try stopValues(object["stop"]),
            seed: intValue(object["seed"]),
            stream: object["stream"] as? Bool ?? false,
            includeUsage: streamOptions?["include_usage"] as? Bool ?? false,
            enableThinking: object["enable_thinking"] as? Bool,
            chatTemplateKwargs: chatTemplateKwargs
        )
    }

    private static func floatValue(_ value: Any?) -> Float? {
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

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            return value
        case let value as NSNumber:
            return value.intValue
        default:
            return nil
        }
    }

    private static func stopValues(_ value: Any?) throws -> [String] {
        guard let value else { return [] }
        if let string = value as? String {
            return [string]
        }
        if let strings = value as? [String] {
            return strings
        }
        throw OpenAIServerError.invalidJSON
    }

    private static func parseChatTemplateKwargs(_ value: Any?) throws -> [String: OpenAIJSONValue]? {
        guard let value else { return nil }
        guard let object = value as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }
        var converted: [String: OpenAIJSONValue] = [:]
        for (key, value) in object {
            guard let jsonValue = OpenAIJSONValue(value) else {
                throw OpenAIServerError.invalidJSON
            }
            converted[key] = jsonValue
        }
        return converted
    }
}

private extension NWConnection {
    func receive() async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { data, _, isComplete, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                if let data {
                    continuation.resume(returning: data)
                    return
                }
                if isComplete {
                    continuation.resume(returning: Data())
                    return
                }
                continuation.resume(throwing: OpenAIServerError.invalidRequest)
            }
        }
    }
}

extension NWConnection {
    func send(data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            send(content: data, completion: .contentProcessed { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            })
        }
    }

    func sendFinal(data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            send(content: data, isComplete: true, completion: .contentProcessed { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            })
        }
    }
}
