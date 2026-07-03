import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import MLXServeHTTP
import Tokenizers

private let defaultModelPath =
    "/Users/timapple/Library/Caches/models/mlx-community/Qwen3-0.6B-4bit"

@main
struct MLXServeHTTPServerMain {
    static func main() async throws {
        let config = try ServerConfig.parse(CommandLine.arguments)
        let modelURL = URL(fileURLWithPath: config.modelPath)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let backend = NativeChatBackend(
                context: context,
                modelID: config.modelID ?? modelURL.lastPathComponent,
                maxConcurrentRequests: config.maxConcurrentRequests
            )
            let server = try OpenAIServer(
                host: config.host,
                port: config.port,
                backend: backend
            )
            server.start()
            print("MLXServeHTTP listening on http://\(config.host):\(config.port)")
            fflush(stdout)
            server.waitForever()
        }
    }
}

private final class NativeChatBackend: OpenAIChatBackend, @unchecked Sendable {
    let models: [OpenAIModelInfo]

    private let context: ModelContext
    private let engine: MLXServeEngine
    private let pump: NativeChatPump
    private let parameters: GenerateParameters

    init(
        context: ModelContext,
        modelID: String,
        maxConcurrentRequests: Int
    ) {
        self.context = context
        self.parameters = GenerateParameters(maxTokens: 16, temperature: 0)
        self.engine = MLXServeEngine(
            model: context.model,
            parameters: parameters,
            maxConcurrentRequests: maxConcurrentRequests
        )
        self.pump = NativeChatPump(engine: engine)
        self.models = [OpenAIModelInfo(id: modelID, maxModelLength: nil)]
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        let input = try await context.processor.prepare(input: userInput(from: request.messages))
        let promptTokens = try countPromptTokens(input)
        let uid = "chat-\(UUID().uuidString)"
        let mlxRequest = Request(
            uid: uid,
            input: input,
            maxTokens: request.maxTokens,
            sampling: SamplingParameters(temperature: request.temperature)
        )

        let responseStream = pump.stream(mlxRequest)
        let chunks = AsyncThrowingStream<OpenAIChatChunk, Error> { continuation in
            Task {
                do {
                    for try await response in responseStream {
                        guard response.token >= 0 else { continue }
                        let text = self.context.tokenizer.decode(
                            tokenIds: [response.token],
                            skipSpecialTokens: true
                        )
                        continuation.yield(OpenAIChatChunk(text: text, tokenID: response.token))
                        if response.finishReason != nil {
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        return OpenAIChatStream(promptTokens: promptTokens, chunks: chunks)
    }

    private func countPromptTokens(_ input: LMInput) throws -> Int {
        input.text.tokens.dim(0)
    }

    private func userInput(from messages: [OpenAIChatMessage]) -> UserInput {
        UserInput(
            chat: messages.map { message in
                switch message.role {
                case "system":
                    return .system(message.content)
                case "assistant":
                    return .assistant(message.content)
                case "tool":
                    return .tool(message.content)
                default:
                    return .user(message.content)
                }
            }
        )
    }
}

private actor NativeChatPump {
    private let engine: MLXServeEngine
    private var continuations: [String: AsyncThrowingStream<Response, Error>.Continuation] = [:]
    private var isRunning = false

    init(engine: MLXServeEngine) {
        self.engine = engine
    }

    nonisolated func stream(_ request: Request) -> AsyncThrowingStream<Response, Error> {
        AsyncThrowingStream { continuation in
            Task {
                await self.submit(request, continuation: continuation)
            }
        }
    }

    private func submit(
        _ request: Request,
        continuation: AsyncThrowingStream<Response, Error>.Continuation
    ) async {
        do {
            try await engine.submit(request)
            continuations[request.uid] = continuation
            startIfNeeded()
        } catch {
            continuation.finish(throwing: error)
        }
    }

    private func startIfNeeded() {
        guard !isRunning else { return }
        isRunning = true
        Task {
            await self.run()
        }
    }

    private func run() async {
        defer { isRunning = false }

        do {
            while true {
                if await engine.isIdle {
                    if continuations.isEmpty {
                        return
                    }
                    await Task.yield()
                    continue
                }

                let responses = try await engine.step()
                for response in responses {
                    guard let continuation = continuations[response.uid] else { continue }
                    continuation.yield(response)
                    if response.finishReason != nil {
                        continuation.finish()
                        continuations.removeValue(forKey: response.uid)
                    }
                }

                if responses.isEmpty {
                    await Task.yield()
                }
            }
        } catch {
            for continuation in continuations.values {
                continuation.finish(throwing: error)
            }
            continuations.removeAll()
        }
    }
}

private struct ServerConfig {
    let host: String
    let port: UInt16
    let modelPath: String
    let modelID: String?
    let maxConcurrentRequests: Int

    static func parse(_ arguments: [String]) throws -> ServerConfig {
        var host = "127.0.0.1"
        var port: UInt16 = 18181
        var modelPath = ProcessInfo.processInfo.environment["MLXSERVE_TEST_MODEL"] ?? defaultModelPath
        var modelID: String?
        var maxConcurrentRequests = 8

        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--host":
                index += 1
                host = try value(arguments, at: index, for: argument)
            case "--port":
                index += 1
                guard let parsed = UInt16(try value(arguments, at: index, for: argument)) else {
                    throw ServerConfigError.invalidArgument("invalid port")
                }
                port = parsed
            case "--model-dir":
                index += 1
                modelPath = try value(arguments, at: index, for: argument)
            case "--model-id":
                index += 1
                modelID = try value(arguments, at: index, for: argument)
            case "--max-concurrent-requests":
                index += 1
                guard let parsed = Int(try value(arguments, at: index, for: argument)) else {
                    throw ServerConfigError.invalidArgument("invalid max concurrent requests")
                }
                maxConcurrentRequests = parsed
            case "--help", "-h":
                printHelp()
                Foundation.exit(0)
            default:
                throw ServerConfigError.invalidArgument(argument)
            }
            index += 1
        }

        return ServerConfig(
            host: host,
            port: port,
            modelPath: modelPath,
            modelID: modelID,
            maxConcurrentRequests: maxConcurrentRequests
        )
    }

    private static func value(_ arguments: [String], at index: Int, for option: String) throws -> String {
        guard arguments.indices.contains(index) else {
            throw ServerConfigError.invalidArgument("missing value for \(option)")
        }
        return arguments[index]
    }

    private static func printHelp() {
        print(
            """
            Usage: swift run mlxserve-http [options]

            Options:
              --host HOST                       Bind host. Default: 127.0.0.1
              --port PORT                       Bind port. Default: 18181
              --model-dir PATH                  MLX model directory.
              --model-id ID                     OpenAI model id. Defaults to model directory name.
              --max-concurrent-requests N       Scheduler concurrency cap. Default: 8
            """
        )
    }
}

private enum ServerConfigError: Error, CustomStringConvertible {
    case invalidArgument(String)

    var description: String {
        switch self {
        case .invalidArgument(let message):
            return message
        }
    }
}
