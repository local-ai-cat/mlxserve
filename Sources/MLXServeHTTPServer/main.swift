import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import MLXServeHTTP
import Tokenizers

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
            try await server.start()
            print("MLXServeHTTP listening on http://\(config.host):\(config.port)")
            fflush(stdout)
            server.waitForever()
        }
    }
}

private final class NativeChatBackend: OpenAIChatBackend, OpenAICompletionBackend, OpenAIHealthProviding, @unchecked Sendable {
    let models: [OpenAIModelInfo]
    var healthInfo: OpenAIHealthInfo {
        OpenAIHealthInfo(
            defaultModel: models.first?.id,
            enginePool: OpenAIHealthEnginePool(modelCount: models.count, loadedCount: models.count)
        )
    }

    private let context: ModelContext
    private let engine: MLXServeEngine
    private let parameters: GenerateParameters
    private let eosTokenIds: Set<Int>

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
        var eosTokenIds = context.configuration.eosTokenIds
        if let tokenizerEosTokenId = context.tokenizer.eosTokenId {
            eosTokenIds.insert(tokenizerEosTokenId)
        }
        self.eosTokenIds = eosTokenIds
        self.models = [OpenAIModelInfo(id: modelID, maxModelLength: nil)]
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        let input = try await context.processor.prepare(input: userInput(from: request))
        let promptTokens = try countPromptTokens(input)
        return streamGeneration(
            uidPrefix: "chat",
            input: input,
            promptTokens: promptTokens,
            maxTokens: request.maxTokens,
            sampling: samplingParameters(
                temperature: request.temperature,
                topP: request.topP,
                topK: request.topK,
                minP: request.minP,
                repetitionPenalty: request.repetitionPenalty,
                presencePenalty: request.presencePenalty,
                frequencyPenalty: request.frequencyPenalty,
                xtcProbability: request.xtcProbability,
                xtcThreshold: request.xtcThreshold,
                seed: request.seed
            )
        )
    }

    func startCompletion(_ request: OpenAICompletionRequest) async throws -> OpenAIChatStream {
        guard case .string(let prompt) = request.prompt else {
            throw NativeChatBackendError.invalidPrompt
        }
        let tokenIDs = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
        let input = LMInput(tokens: MLXArray(tokenIDs))
        return streamGeneration(
            uidPrefix: "cmpl",
            input: input,
            promptTokens: tokenIDs.count,
            maxTokens: request.maxTokens,
            sampling: samplingParameters(
                temperature: request.temperature,
                topP: request.topP,
                topK: request.topK,
                minP: request.minP,
                repetitionPenalty: request.repetitionPenalty,
                presencePenalty: request.presencePenalty,
                frequencyPenalty: request.frequencyPenalty,
                seed: request.seed
            )
        )
    }

    private func streamGeneration(
        uidPrefix: String,
        input: LMInput,
        promptTokens: Int,
        maxTokens: Int,
        sampling: SamplingParameters
    ) -> OpenAIChatStream {
        let mlxRequest = Request(
            uid: "\(uidPrefix)-\(UUID().uuidString)",
            input: input,
            maxTokens: maxTokens,
            sampling: sampling,
            eosTokenIds: eosTokenIds
        )

        let responseStream = engine.stream(mlxRequest)
        let chunks = AsyncThrowingStream<OpenAIChatChunk, Error> { continuation in
            let task = Task {
                do {
                    for try await response in responseStream {
                        if case .failed(let message)? = response.finishReason {
                            throw NativeChatBackendError.generationFailed(message)
                        }
                        guard response.token >= 0 else {
                            if response.finishReason != nil {
                                break
                            }
                            continue
                        }
                        let text = self.context.tokenizer.decode(
                            tokenIds: [response.token],
                            skipSpecialTokens: true
                        )
                        continuation.yield(
                            OpenAIChatChunk(
                                text: text,
                                tokenID: response.token,
                                finishReason: openAIFinishReason(response.finishReason)
                            )
                        )
                        if response.finishReason != nil {
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }

        return OpenAIChatStream(promptTokens: promptTokens, chunks: chunks)
    }

    private func samplingParameters(
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        repetitionPenalty: Float,
        presencePenalty: Float,
        frequencyPenalty: Float,
        xtcProbability: Float = 0,
        xtcThreshold: Float = 0.1,
        seed: Int?
    ) -> SamplingParameters {
        SamplingParameters(
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            frequencyPenalty: frequencyPenalty,
            xtcProbability: xtcProbability,
            xtcThreshold: xtcThreshold,
            xtcSpecialTokens: xtcSpecialTokens(),
            seed: seed
        )
    }

    private func countPromptTokens(_ input: LMInput) throws -> Int {
        input.text.tokens.dim(0)
    }

    private func xtcSpecialTokens() -> [Int] {
        let newlineTokens = context.tokenizer.encode(text: "\n", addSpecialTokens: false)
        return Array(Set(newlineTokens).union(eosTokenIds)).sorted()
    }

    private func userInput(from request: OpenAIChatRequest) -> UserInput {
        UserInput(
            chat: request.messages.map { message in
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
            },
            additionalContext: additionalContext(from: request)
        )
    }

    private func additionalContext(from request: OpenAIChatRequest) -> [String: any Sendable]? {
        var context: [String: any Sendable] = [:]
        if let chatTemplateKwargs = request.chatTemplateKwargs {
            for (key, value) in chatTemplateKwargs {
                context[key] = sendableValue(from: value)
            }
        }
        if let enableThinking = request.enableThinking {
            context["enable_thinking"] = enableThinking
        }
        return context.isEmpty ? nil : context
    }

    private func sendableValue(from value: OpenAIJSONValue) -> any Sendable {
        switch value {
        case .string(let string):
            return string
        case .number(let number):
            return number
        case .bool(let bool):
            return bool
        case .object(let object):
            return object.mapValues { sendableValue(from: $0) }
        case .array(let array):
            return array.map { sendableValue(from: $0) }
        case .null:
            return Optional<String>.none as String?
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
        var modelPath = ProcessInfo.processInfo.environment["MLXSERVE_MODEL_DIR"]
            ?? ProcessInfo.processInfo.environment["MLXSERVE_TEST_MODEL"]
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

        guard let modelPath else {
            throw ServerConfigError.invalidArgument("missing --model-dir or MLXSERVE_MODEL_DIR")
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
              --model-dir PATH                  MLX model directory. Required unless MLXSERVE_MODEL_DIR is set.
              --model-id ID                     OpenAI model id. Defaults to model directory name.
              --max-concurrent-requests N       Scheduler concurrency cap. Default: 8
            """
        )
    }
}

private enum NativeChatBackendError: Error, CustomStringConvertible {
    case generationFailed(String)
    case invalidPrompt

    var description: String {
        switch self {
        case .generationFailed(let message):
            return message
        case .invalidPrompt:
            return "completion backend requires a single prompt"
        }
    }
}

private func openAIFinishReason(_ finishReason: FinishReason?) -> String? {
    switch finishReason {
    case .stop:
        return "stop"
    case .length:
        return "length"
    case .cancelled:
        return "stop"
    case .failed:
        return "stop"
    case nil:
        return nil
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
