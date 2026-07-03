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

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct OpenAIChatRequest: Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let maxTokens: Int
    public let temperature: Float
    public let stream: Bool

    public init(
        model: String,
        messages: [OpenAIChatMessage],
        maxTokens: Int,
        temperature: Float,
        stream: Bool
    ) {
        self.model = model
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.stream = stream
    }
}

public struct OpenAIChatChunk: Sendable {
    public let text: String
    public let tokenID: Int

    public init(text: String, tokenID: Int) {
        self.text = text
        self.tokenID = tokenID
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

public protocol OpenAIChatBackend: Sendable {
    var models: [OpenAIModelInfo] { get }
    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream
}

public final class OpenAIServer: @unchecked Sendable {
    private let host: NWEndpoint.Host
    private let port: NWEndpoint.Port
    private let backend: any OpenAIChatBackend
    private let listener: NWListener

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
    }

    public func start() {
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
            default:
                try await sendJSON(["error": ["message": "not found"]], status: 404, connection: connection)
            }
        } catch {
            try? await sendJSON(
                ["error": ["message": String(describing: error)]],
                status: 500,
                connection: connection
            )
        }
    }

    private func handleChatCompletion(_ request: HTTPRequest, connection: NWConnection) async throws {
        let chatRequest = try OpenAIChatRequest.parse(request.body)
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

        for try await chunk in stream.chunks {
            let now = DispatchTime.now().uptimeNanoseconds
            if firstToken == nil {
                firstToken = now
            }
            lastToken = now
            completionTokens += 1
            try await sendSSE(
                [
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        [
                            "index": 0,
                            "delta": ["content": chunk.text],
                            "finish_reason": NSNull(),
                        ]
                    ],
                ],
                connection: connection
            )
        }

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
                        "finish_reason": "length",
                    ]
                ],
            ],
            connection: connection
        )
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
        var text = ""

        for try await chunk in stream.chunks {
            let now = DispatchTime.now().uptimeNanoseconds
            if firstToken == nil {
                firstToken = now
            }
            lastToken = now
            completionTokens += 1
            text += chunk.text
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
                        "message": ["role": "assistant", "content": text],
                        "finish_reason": "length",
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

    private func sendJSON(
        _ object: [String: Any],
        status: Int,
        connection: NWConnection
    ) async throws {
        let body = try JSONSerialization.data(withJSONObject: object, options: [])
        let reason = status == 200 ? "OK" : status == 404 ? "Not Found" : "Internal Server Error"
        let header = "HTTP/1.1 \(status) \(reason)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.count)\r\n"
            + "Connection: close\r\n"
            + "\r\n"
        try await connection.sendFinal(data: Data(header.utf8) + body)
    }
}

public enum OpenAIServerError: Error, CustomStringConvertible {
    case invalidPort(UInt16)
    case invalidRequest
    case invalidJSON
    case unsupportedContent

    public var description: String {
        switch self {
        case .invalidPort(let port):
            return "invalid port: \(port)"
        case .invalidRequest:
            return "invalid HTTP request"
        case .invalidJSON:
            return "invalid JSON request body"
        case .unsupportedContent:
            return "unsupported request content"
        }
    }
}

private struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    static func read(from connection: NWConnection) async throws -> HTTPRequest {
        var buffer = Data()
        var headerEnd: Range<Data.Index>?
        while headerEnd == nil {
            let chunk = try await connection.receive()
            guard !chunk.isEmpty else { throw OpenAIServerError.invalidRequest }
            buffer.append(chunk)
            headerEnd = buffer.range(of: Data("\r\n\r\n".utf8))
        }

        guard let headerRange = headerEnd else { throw OpenAIServerError.invalidRequest }
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

        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        let bodyStart = headerRange.upperBound
        var body = Data(buffer[bodyStart...])
        while body.count < contentLength {
            let chunk = try await connection.receive()
            guard !chunk.isEmpty else { throw OpenAIServerError.invalidRequest }
            body.append(chunk)
        }
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
}

private extension OpenAIChatRequest {
    static func parse(_ body: Data) throws -> OpenAIChatRequest {
        guard
            let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
            let model = object["model"] as? String,
            let rawMessages = object["messages"] as? [[String: Any]]
        else {
            throw OpenAIServerError.invalidJSON
        }

        let messages = rawMessages.compactMap { raw -> OpenAIChatMessage? in
            guard let role = raw["role"] as? String else { return nil }
            if let content = raw["content"] as? String {
                return OpenAIChatMessage(role: role, content: content)
            }
            if let parts = raw["content"] as? [[String: Any]] {
                let text = parts.compactMap { part in
                    part["type"] as? String == "text" ? part["text"] as? String : nil
                }.joined()
                return OpenAIChatMessage(role: role, content: text)
            }
            return nil
        }
        guard !messages.isEmpty else { throw OpenAIServerError.invalidJSON }

        return OpenAIChatRequest(
            model: model,
            messages: messages,
            maxTokens: object["max_tokens"] as? Int ?? 16,
            temperature: Float((object["temperature"] as? Double) ?? 0),
            stream: object["stream"] as? Bool ?? false
        )
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
