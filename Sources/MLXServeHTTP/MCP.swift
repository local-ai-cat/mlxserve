import Foundation

public struct MCPServerConfig: Sendable, Equatable {
    public let name: String
    public let command: String
    public let args: [String]
    public let env: [String: String]
    public let timeoutMs: Int

    public init(name: String, command: String, args: [String] = [], env: [String: String] = [:], timeoutMs: Int = 30_000) {
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.timeoutMs = timeoutMs
    }
}

public struct MCPConfig: Sendable, Equatable {
    public let servers: [MCPServerConfig]

    public init(servers: [MCPServerConfig]) {
        self.servers = servers
    }

    public static func load(from url: URL) throws -> MCPConfig {
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }
        return try parse(object)
    }

    public static func parse(_ object: [String: Any]) throws -> MCPConfig {
        let rawServers = object["mcpServers"] ?? object["servers"]
        guard let serversObject = rawServers as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }

        var servers: [MCPServerConfig] = []
        for name in serversObject.keys.sorted() {
            guard let serverObject = serversObject[name] as? [String: Any] else {
                throw OpenAIServerError.invalidJSON
            }
            let enabled = serverObject["enabled"] as? Bool ?? true
            guard enabled else { continue }
            let transport = serverObject["transport"] as? String ?? "stdio"
            guard transport == "stdio" else { continue }
            guard let command = serverObject["command"] as? String, !command.isEmpty else {
                throw OpenAIServerError.invalidJSON
            }
            let args = serverObject["args"] as? [String] ?? []
            let env = serverObject["env"] as? [String: String] ?? [:]
            let timeoutMs = serverObject["timeoutMs"] as? Int ?? 30_000
            guard timeoutMs > 0 else {
                throw OpenAIServerError.invalidJSON
            }
            servers.append(MCPServerConfig(name: name, command: command, args: args, env: env, timeoutMs: timeoutMs))
        }
        return MCPConfig(servers: servers)
    }
}

public struct MCPTool: Sendable, Equatable {
    public let serverName: String
    public let name: String
    public let description: String
    public let inputSchema: OpenAIJSONValue

    public var fullName: String {
        "\(serverName)__\(name)"
    }

    public init(serverName: String, name: String, description: String, inputSchema: OpenAIJSONValue) {
        self.serverName = serverName
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
    }
}

public struct MCPServerStatus: Sendable, Equatable {
    public let name: String
    public let state: String
    public let transport: String
    public let toolsCount: Int
    public let error: String?

    public init(name: String, state: String, transport: String, toolsCount: Int, error: String?) {
        self.name = name
        self.state = state
        self.transport = transport
        self.toolsCount = toolsCount
        self.error = error
    }
}

public struct MCPToolExecutionResult: Sendable, Equatable {
    public let toolName: String
    public let content: OpenAIJSONValue
    public let isError: Bool
    public let errorMessage: String?

    public init(toolName: String, content: OpenAIJSONValue, isError: Bool = false, errorMessage: String? = nil) {
        self.toolName = toolName
        self.content = content
        self.isError = isError
        self.errorMessage = errorMessage
    }
}

public actor MCPManager {
    private var clients: [String: MCPStdioClient]

    public init(config: MCPConfig) {
        var clients: [String: MCPStdioClient] = [:]
        for server in config.servers {
            clients[server.name] = MCPStdioClient(config: server)
        }
        self.clients = clients
    }

    public static func load(from url: URL) throws -> MCPManager {
        MCPManager(config: try MCPConfig.load(from: url))
    }

    public func connectAll() async {
        for name in clients.keys.sorted() {
            guard let client = clients[name] else { continue }
            do {
                try await client.connect()
            } catch {
                await client.markError(String(describing: error))
            }
        }
    }

    public func shutdown() async {
        for name in clients.keys.sorted() {
            guard let client = clients[name] else { continue }
            await client.shutdown()
        }
    }

    public func allTools() async -> [MCPTool] {
        var tools: [MCPTool] = []
        for name in clients.keys.sorted() {
            guard let client = clients[name] else { continue }
            tools.append(contentsOf: await client.tools)
        }
        return tools
    }

    public func serverStatuses() async -> [MCPServerStatus] {
        var statuses: [MCPServerStatus] = []
        for name in clients.keys.sorted() {
            guard let client = clients[name] else { continue }
            statuses.append(await client.status)
        }
        return statuses
    }

    public func execute(toolName: String, arguments: OpenAIJSONValue) async -> MCPToolExecutionResult {
        let (serverName, localName) = splitToolName(toolName)
        if !serverName.isEmpty {
            guard let client = clients[serverName] else {
                return MCPToolExecutionResult(
                    toolName: toolName,
                    content: .null,
                    isError: true,
                    errorMessage: "MCP server not found: \(serverName)"
                )
            }
            return await client.callTool(fullName: toolName, localName: localName, arguments: arguments)
        }

        for name in clients.keys.sorted() {
            guard let client = clients[name], await client.hasTool(named: localName) else { continue }
            return await client.callTool(fullName: toolName, localName: localName, arguments: arguments)
        }

        return MCPToolExecutionResult(
            toolName: toolName,
            content: .null,
            isError: true,
            errorMessage: "MCP tool not found: \(toolName)"
        )
    }

    public func mergedOpenAITools(
        userTools: [OpenAIJSONValue]?,
        toolChoice: OpenAIToolChoice?
    ) async -> [OpenAIJSONValue]? {
        if case .some(OpenAIToolChoice.none) = toolChoice {
            return nil
        }

        var mergedByName: [String: OpenAIJSONValue] = [:]
        for tool in await allTools() {
            mergedByName[tool.fullName] = openAITool(from: tool)
        }
        for tool in userTools ?? [] {
            guard let name = openAIToolFunctionName(tool) else { continue }
            mergedByName[name] = tool
        }
        let names = mergedByName.keys.sorted()
        let merged = names.compactMap { mergedByName[$0] }
        return merged.isEmpty ? nil : merged
    }

    private func splitToolName(_ toolName: String) -> (server: String, local: String) {
        guard let range = toolName.range(of: "__") else {
            return ("", toolName)
        }
        return (String(toolName[..<range.lowerBound]), String(toolName[range.upperBound...]))
    }

    private func openAITool(from tool: MCPTool) -> OpenAIJSONValue {
        .object(
            [
                "type": .string("function"),
                "function": .object(
                    [
                        "name": .string(tool.fullName),
                        "description": .string(tool.description),
                        "parameters": tool.inputSchema,
                    ]
                ),
            ]
        )
    }
}

public func openAIChatRequestByReplacingTools(
    _ request: OpenAIChatRequest,
    tools: [OpenAIJSONValue]?
) -> OpenAIChatRequest {
    OpenAIChatRequest(
        model: request.model,
        messages: request.messages,
        maxTokens: request.maxTokens,
        temperature: request.temperature,
        topP: request.topP,
        topK: request.topK,
        repetitionPenalty: request.repetitionPenalty,
        minP: request.minP,
        xtcProbability: request.xtcProbability,
        xtcThreshold: request.xtcThreshold,
        presencePenalty: request.presencePenalty,
        frequencyPenalty: request.frequencyPenalty,
        stop: request.stop,
        seed: request.seed,
        stream: request.stream,
        includeUsage: request.includeUsage,
        enableThinking: request.enableThinking,
        chatTemplateKwargs: request.chatTemplateKwargs,
        structuredOutput: request.structuredOutput,
        tools: tools,
        toolChoice: request.toolChoice
    )
}

private actor MCPStdioClient {
    private let config: MCPServerConfig
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var nextID = 1
    private var state = "disconnected"
    private var error: String?
    private var discoveredTools: [MCPTool] = []
    private var readBuffer = Data()
    private var queuedLines: [Data] = []
    private var pendingRead: (id: UUID, continuation: CheckedContinuation<Data, Error>)?
    private var streamError: Error?
    private var activeRequestID: UUID?

    init(config: MCPServerConfig) {
        self.config = config
    }

    var tools: [MCPTool] {
        discoveredTools
    }

    var status: MCPServerStatus {
        MCPServerStatus(
            name: config.name,
            state: state,
            transport: "stdio",
            toolsCount: discoveredTools.count,
            error: error
        )
    }

    func markError(_ message: String) {
        state = "error"
        error = message
        discoveredTools = []
    }

    func shutdown() {
        closeProcess()
        process = nil
        state = "disconnected"
    }

    func hasTool(named name: String) -> Bool {
        discoveredTools.contains { $0.name == name || $0.fullName == name }
    }

    func connect() async throws {
        guard process == nil else { return }
        state = "connecting"
        error = nil
        streamError = nil
        readBuffer = Data()
        queuedLines = []

        let process = Process()
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        if config.command.contains("/") {
            process.executableURL = URL(fileURLWithPath: config.command)
            process.arguments = config.args
        } else {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [config.command] + config.args
        }
        var environment = ProcessInfo.processInfo.environment
        for (key, value) in config.env {
            environment[key] = value
        }
        process.environment = environment
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = FileHandle.nullDevice
        try process.run()

        self.process = process
        input = stdinPipe.fileHandleForWriting
        output = stdoutPipe.fileHandleForReading
        output?.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            Task { await self?.receiveOutputData(data) }
        }

        _ = try await sendRequest(
            method: "initialize",
            params: [
                "protocolVersion": "2024-11-05",
                "capabilities": [:],
                "clientInfo": ["name": "MLXServe", "version": "0"],
            ]
        )
        try sendNotification(method: "notifications/initialized", params: [:])
        let toolsResult = try await sendRequest(method: "tools/list", params: [:])
        discoveredTools = parseTools(from: toolsResult)
        state = "connected"
    }

    func callTool(fullName: String, localName: String, arguments: OpenAIJSONValue) async -> MCPToolExecutionResult {
        do {
            if process == nil {
                try await connect()
            }
            let result = try await sendRequest(
                method: "tools/call",
                params: [
                    "name": localName,
                    "arguments": jsonObject(from: arguments),
                ]
            )
            let resultObject = result as? [String: Any] ?? [:]
            let isError = resultObject["isError"] as? Bool ?? false
            let contentValue = resultObject["content"] ?? result
            let content = OpenAIJSONValue(contentValue) ?? .null
            return MCPToolExecutionResult(
                toolName: fullName,
                content: content,
                isError: isError,
                errorMessage: isError ? mcpErrorMessage(from: content) : nil
            )
        } catch {
            return MCPToolExecutionResult(
                toolName: fullName,
                content: .null,
                isError: true,
                errorMessage: String(describing: error)
            )
        }
    }

    private func sendRequest(method: String, params: [String: Any]) async throws -> Any {
        let id = nextID
        nextID += 1
        try writeMessage(
            [
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            ]
        )

        let requestID = UUID()
        activeRequestID = requestID
        let timeoutMs = config.timeoutMs
        let timeoutTask = Task {
            do {
                try await Task.sleep(nanoseconds: UInt64(timeoutMs) * 1_000_000)
            } catch {
                return
            }
            self.timeoutActiveRequest(id: requestID, timeoutMs: timeoutMs)
        }
        defer {
            timeoutTask.cancel()
            if activeRequestID == requestID {
                activeRequestID = nil
            }
        }

        do {
            return try await readMatchingResponse(id: id)
        } catch let error as MCPStdioError {
            if case .timeout = error {
                resetAfterTimeout(error.description)
            }
            throw error
        }
    }

    private func timeoutActiveRequest(id: UUID, timeoutMs: Int) {
        guard activeRequestID == id else { return }
        let error = MCPStdioError.timeout(timeoutMs)
        streamError = error
        finishPendingRead(.failure(error))
    }

    private func readMatchingResponse(id: Int) async throws -> Any {
        while true {
            let response = try await readMessage()
            guard let responseID = response["id"] as? Int, responseID == id else {
                continue
            }
            if let error = response["error"] as? [String: Any] {
                throw MCPStdioError.rpcError(error["message"] as? String ?? String(describing: error))
            }
            return response["result"] ?? [:]
        }
    }

    private func sendNotification(method: String, params: [String: Any]) throws {
        try writeMessage(
            [
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            ]
        )
    }

    private func writeMessage(_ message: [String: Any]) throws {
        guard let input else {
            throw MCPStdioError.notConnected
        }
        let body = try JSONSerialization.data(withJSONObject: message, options: [])
        input.write(body + Data("\n".utf8))
    }

    private func readMessage() async throws -> [String: Any] {
        guard output != nil else {
            throw MCPStdioError.notConnected
        }
        if !queuedLines.isEmpty {
            return try parseMessageLine(queuedLines.removeFirst())
        }
        if let streamError {
            throw streamError
        }

        let readID = UUID()
        return try await withTaskCancellationHandler(
            operation: {
                let line = try await withCheckedThrowingContinuation { continuation in
                    pendingRead = (readID, continuation)
                }
                return try parseMessageLine(line)
            },
            onCancel: {
                Task { await self.cancelPendingRead(id: readID) }
            }
        )
    }

    private func receiveOutputData(_ data: Data) {
        guard !data.isEmpty else {
            streamError = MCPStdioError.closed
            finishPendingRead(.failure(MCPStdioError.closed))
            return
        }

        readBuffer.append(data)
        while let newline = readBuffer.firstIndex(of: UInt8(ascii: "\n")) {
            let lineRange = readBuffer.startIndex..<newline
            var line = Data(readBuffer[lineRange])
            let nextIndex = readBuffer.index(after: newline)
            readBuffer.removeSubrange(readBuffer.startIndex..<nextIndex)
            if line.last == UInt8(ascii: "\r") {
                line.removeLast()
            }
            guard !line.isEmpty else {
                continue
            }
            enqueueLine(line)
        }
    }

    private func enqueueLine(_ line: Data) {
        if let pendingRead {
            self.pendingRead = nil
            pendingRead.continuation.resume(returning: line)
        } else {
            queuedLines.append(line)
        }
    }

    private func finishPendingRead(_ result: Result<Data, Error>) {
        guard let pendingRead else { return }
        self.pendingRead = nil
        switch result {
        case .success(let line):
            pendingRead.continuation.resume(returning: line)
        case .failure(let error):
            pendingRead.continuation.resume(throwing: error)
        }
    }

    private func cancelPendingRead(id: UUID) {
        guard let pendingRead, pendingRead.id == id else { return }
        self.pendingRead = nil
        pendingRead.continuation.resume(throwing: CancellationError())
    }

    private func resetAfterTimeout(_ message: String) {
        closeProcess()
        process = nil
        readBuffer = Data()
        queuedLines = []
        streamError = nil
        state = "error"
        error = message
    }

    private func closeProcess() {
        output?.readabilityHandler = nil
        input?.closeFile()
        output?.closeFile()
        if let process, process.isRunning {
            process.terminate()
        }
        input = nil
        output = nil
        finishPendingRead(.failure(MCPStdioError.closed))
    }

    private func parseMessageLine(_ line: Data) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: line) as? [String: Any] else {
            throw MCPStdioError.invalidMessage
        }
        return object
    }

    private func parseTools(from result: Any) -> [MCPTool] {
        guard let object = result as? [String: Any], let tools = object["tools"] as? [[String: Any]] else {
            return []
        }
        return tools.compactMap { tool in
            guard let name = tool["name"] as? String, !name.isEmpty else {
                return nil
            }
            let description = tool["description"] as? String ?? ""
            let rawSchema = tool["inputSchema"] ?? [
                "type": "object",
                "properties": [:],
            ]
            let inputSchema = OpenAIJSONValue(rawSchema) ?? .object(
                [
                    "type": .string("object"),
                    "properties": .object([:]),
                ]
            )
            return MCPTool(
                serverName: config.name,
                name: name,
                description: description,
                inputSchema: inputSchema
            )
        }
    }

    private func mcpErrorMessage(from content: OpenAIJSONValue) -> String? {
        if case .string(let string) = content {
            return string
        }
        if case .array(let parts) = content {
            for part in parts {
                guard case .object(let object) = part, case .string(let text)? = object["text"] else {
                    continue
                }
                return text
            }
        }
        return nil
    }
}

private enum MCPStdioError: Error, CustomStringConvertible {
    case notConnected
    case closed
    case invalidMessage
    case rpcError(String)
    case timeout(Int)

    var description: String {
        switch self {
        case .notConnected:
            return "MCP stdio client is not connected"
        case .closed:
            return "MCP stdio server closed its output"
        case .invalidMessage:
            return "invalid MCP stdio message"
        case .rpcError(let message):
            return "MCP JSON-RPC error: \(message)"
        case .timeout(let timeoutMs):
            return "tool failed: timeout after \(timeoutMs)ms"
        }
    }
}

func jsonObject(from value: OpenAIJSONValue) -> Any {
    switch value {
    case .string(let string):
        return string
    case .number(let number):
        return number
    case .bool(let bool):
        return bool
    case .object(let object):
        return object.mapValues { jsonObject(from: $0) }
    case .array(let array):
        return array.map { jsonObject(from: $0) }
    case .null:
        return NSNull()
    }
}
