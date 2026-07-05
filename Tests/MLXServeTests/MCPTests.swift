import Foundation
@testable import MLXServeHTTP
import XCTest

final class MCPTests: XCTestCase {
    func testMCPConfigParsesClaudeDesktopShape() throws {
        let config = try MCPConfig.parse(
            [
                "mcpServers": [
                    "fake": [
                        "command": "python3",
                        "args": ["server.py"],
                        "env": ["TOKEN": "abc"],
                    ],
                    "disabled": [
                        "command": "ignored",
                        "enabled": false,
                    ],
                    "sse": [
                        "transport": "sse",
                        "url": "http://127.0.0.1:1",
                    ],
                ]
            ]
        )

        XCTAssertEqual(
            config.servers,
            [MCPServerConfig(name: "fake", command: "python3", args: ["server.py"], env: ["TOKEN": "abc"])]
        )
    }

    func testMCPConfigParsesPerServerTimeout() throws {
        let config = try MCPConfig.parse(
            [
                "mcpServers": [
                    "fake": [
                        "command": "python3",
                        "args": ["server.py"],
                        "timeoutMs": 250,
                    ]
                ]
            ]
        )

        XCTAssertEqual(config.servers.first?.timeoutMs, 250)
    }

    func testMCPHandlerUnconfiguredEndpointShapes() async {
        let handler = MCPHandler(manager: nil)

        let tools = await handler.toolsResponse()
        XCTAssertEqual(tools.status, 200)
        XCTAssertEqual(tools.body["count"] as? Int, 0)
        XCTAssertEqual((tools.body["tools"] as? [Any])?.count, 0)

        let servers = await handler.serversResponse()
        XCTAssertEqual(servers.status, 200)
        XCTAssertEqual((servers.body["servers"] as? [Any])?.count, 0)

        let execute = await handler.executeResponse(body: Data(#"{"tool":"fake__echo","arguments":{}}"#.utf8))
        XCTAssertEqual(execute.status, 503)
        XCTAssertNotNil(execute.body["error"])
    }

    func testMCPStdioClientDiscoversAndExecutesFakeServer() async throws {
        let manager = try makeFakeMCPManager()
        await manager.connectAll()

        let statuses = await manager.serverStatuses()
        XCTAssertEqual(statuses.first?.state, "connected")
        XCTAssertEqual(statuses.first?.toolsCount, 1)

        let tools = await manager.allTools()
        XCTAssertEqual(tools.count, 1)
        XCTAssertEqual(tools[0].fullName, "fake__echo")
        XCTAssertEqual(tools[0].description, "Echo input")

        let result = await manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("hello")])
        )
        XCTAssertFalse(result.isError)
        guard case .array(let content) = result.content,
            case .object(let first)? = content.first,
            case .string(let text)? = first["text"]
        else {
            XCTFail("expected MCP text content")
            return
        }
        XCTAssertEqual(text, "hello")
        await manager.shutdown()
    }

    func testMCPHandlerExecuteShapeWithFakeServer() async throws {
        let manager = try makeFakeMCPManager()
        await manager.connectAll()
        let handler = MCPHandler(manager: manager)

        let response = await handler.executeResponse(
            body: Data(#"{"tool":"fake__echo","arguments":{"message":"route"}}"#.utf8)
        )

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(response.body["tool_name"] as? String, "fake__echo")
        XCTAssertEqual(response.body["is_error"] as? Bool, false)
        let content = try XCTUnwrap(response.body["content"] as? [[String: Any]])
        XCTAssertEqual(content.first?["text"] as? String, "route")
        await manager.shutdown()
    }

    func testMergedOpenAIChatRequestAddsDiscoveredTools() async throws {
        let manager = try makeFakeMCPManager()
        await manager.connectAll()
        let userTool = OpenAIJSONValue(
            [
                "type": "function",
                "function": [
                    "name": "user_tool",
                    "parameters": ["type": "object", "properties": [:]],
                ],
            ]
        )

        let request = OpenAIChatRequest(
            model: "test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            maxTokens: 1,
            tools: userTool.map { [$0] }
        )
        let tools = await manager.mergedOpenAITools(
            userTools: request.tools,
            toolChoice: request.toolChoice
        )
        let mergedRequest = openAIChatRequestByReplacingTools(request, tools: tools)

        let names = (mergedRequest.tools ?? []).compactMap(openAIToolFunctionName).sorted()
        XCTAssertEqual(names, ["fake__echo", "user_tool"])
        await manager.shutdown()
    }

    func testMergedAnthropicMessagesRequestAddsDiscoveredTools() async throws {
        let manager = try makeFakeMCPManager()
        await manager.connectAll()
        let request = try AnthropicMessagesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hello"}],
                  "tools": [
                    {
                      "name": "user_tool",
                      "description": "User tool",
                      "input_schema": {"type": "object", "properties": {}}
                    }
                  ]
                }
                """.utf8
            )
        )

        let mergedRequest = await anthropicMessagesRequestByMergingMCPTools(request, manager: manager)
        let names = (mergedRequest.openAIRequest().tools ?? []).compactMap(openAIToolFunctionName).sorted()

        XCTAssertEqual(names, ["fake__echo", "user_tool"])
        await manager.shutdown()
    }

    func testMergedResponsesRequestAddsDiscoveredTools() async throws {
        let manager = try makeFakeMCPManager()
        await manager.connectAll()
        let request = try ResponsesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "input": "hello",
                  "tools": [
                    {
                      "type": "function",
                      "name": "user_tool",
                      "description": "User tool",
                      "parameters": {"type": "object", "properties": {}}
                    }
                  ]
                }
                """.utf8
            )
        )

        let mergedRequest = await responsesRequestByMergingMCPTools(request, manager: manager)
        let names = (mergedRequest.openAIRequest().tools ?? []).compactMap(openAIToolFunctionName).sorted()

        XCTAssertEqual(names, ["fake__echo", "user_tool"])
        await manager.shutdown()
    }

    func testMCPStdioTimeoutReturnsToolErrorWithinDeadline() async throws {
        let manager = try makeFakeMCPManager(mode: "hang-call", timeoutMs: 150)
        await manager.connectAll()

        let started = Date()
        let result = await manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("never")])
        )
        let elapsed = Date().timeIntervalSince(started)

        XCTAssertTrue(result.isError)
        XCTAssertEqual(result.errorMessage, "tool failed: timeout after 150ms")
        XCTAssertLessThan(elapsed, 1.5)
        let statuses = await manager.serverStatuses()
        XCTAssertEqual(statuses.first?.state, "error")
        XCTAssertEqual(statuses.first?.error, "tool failed: timeout after 150ms")
        await manager.shutdown()
    }

    func testMCPStdioSlowReplyWithinTimeoutSucceeds() async throws {
        let manager = try makeFakeMCPManager(mode: "slow-call", timeoutMs: 800)
        await manager.connectAll()

        let result = await manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("slow")])
        )

        XCTAssertFalse(result.isError)
        guard case .array(let content) = result.content,
            case .object(let first)? = content.first,
            case .string(let text)? = first["text"]
        else {
            XCTFail("expected MCP text content")
            return
        }
        XCTAssertEqual(text, "slow")
        await manager.shutdown()
    }

    func testMCPStdioTimeoutRespawnsServerOnNextUse() async throws {
        let markerURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxserve-mcp-respawn-\(UUID().uuidString)")
        let manager = try makeFakeMCPManager(mode: "hang-once", timeoutMs: 150, markerURL: markerURL)
        await manager.connectAll()

        let timedOut = await manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("first")])
        )
        XCTAssertTrue(timedOut.isError)
        XCTAssertEqual(timedOut.errorMessage, "tool failed: timeout after 150ms")

        let respawned = await manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("second")])
        )
        XCTAssertFalse(respawned.isError)
        guard case .array(let content) = respawned.content,
            case .object(let first)? = content.first,
            case .string(let text)? = first["text"]
        else {
            XCTFail("expected MCP text content")
            return
        }
        XCTAssertEqual(text, "second")
        let statuses = await manager.serverStatuses()
        XCTAssertEqual(statuses.first?.state, "connected")
        await manager.shutdown()
    }

    private func makeFakeMCPManager(mode: String = "echo", timeoutMs: Int = 30_000, markerURL: URL? = nil) throws -> MCPManager {
        let scriptURL = try fakeMCPServerScriptURL()
        var args = [scriptURL.path, mode]
        if let markerURL {
            args.append(markerURL.path)
        }
        return MCPManager(
            config: MCPConfig(
                servers: [
                    MCPServerConfig(name: "fake", command: "python3", args: args, timeoutMs: timeoutMs)
                ]
            )
        )
    }

    private func fakeMCPServerScriptURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxserve-mcp-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let scriptURL = directory.appendingPathComponent("fake_mcp_server.py")
        try fakeMCPServerScript.write(to: scriptURL, atomically: true, encoding: .utf8)
        return scriptURL
    }

    private var fakeMCPServerScript: String {
        #"""
        import json
        import os
        import sys
        import time

        mode = sys.argv[1] if len(sys.argv) > 1 else "echo"
        marker_path = sys.argv[2] if len(sys.argv) > 2 else None

        def read_message():
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            return json.loads(line.decode("utf-8"))

        def write_message(message):
            body = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
            sys.stdout.buffer.write(body)
            sys.stdout.buffer.flush()

        while True:
            message = read_message()
            if message is None:
                break
            method = message.get("method")
            request_id = message.get("id")
            if request_id is None:
                continue
            if method == "initialize":
                write_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "fake", "version": "1"}
                    }
                })
            elif method == "tools/list":
                write_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [{
                            "name": "echo",
                            "description": "Echo input",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}}
                            }
                        }]
                    }
                })
            elif method == "tools/call":
                arguments = message.get("params", {}).get("arguments", {})
                if mode == "hang-call":
                    while True:
                        time.sleep(1)
                if mode == "slow-call":
                    time.sleep(0.2)
                if mode == "hang-once" and marker_path and not os.path.exists(marker_path):
                    with open(marker_path, "w", encoding="utf-8") as marker:
                        marker.write("hung")
                    while True:
                        time.sleep(1)
                write_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": arguments.get("message", "")}],
                        "isError": False
                    }
                })
            else:
                write_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "method not found"}
                })
        """#
    }
}
