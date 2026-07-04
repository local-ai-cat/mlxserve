import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import MLXServeHTTP
import Tokenizers

final class NativeModelEngine: @unchecked Sendable {
    let modelID: String

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
        self.modelID = modelID
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
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        let input = try await context.processor.prepare(input: userInput(from: request))
        let promptTokens = try countPromptTokens(input)
        let uid = "chat-\(UUID().uuidString)"
        let mlxRequest = Request(
            uid: uid,
            input: input,
            maxTokens: request.maxTokens,
            sampling: SamplingParameters(
                temperature: request.temperature,
                topP: request.topP,
                topK: request.topK,
                minP: request.minP,
                repetitionPenalty: request.repetitionPenalty,
                presencePenalty: request.presencePenalty,
                frequencyPenalty: request.frequencyPenalty,
                xtcProbability: request.xtcProbability,
                xtcThreshold: request.xtcThreshold,
                xtcSpecialTokens: xtcSpecialTokens(),
                seed: request.seed
            ),
            eosTokenIds: eosTokenIds
        )

        return makeStream(from: mlxRequest, promptTokens: promptTokens)
    }

    func startCompletion(_ request: OpenAICompletionRequest) async throws -> OpenAIChatStream {
        guard case .string(let prompt) = request.prompt else {
            throw NativeModelEngineError.invalidPrompt
        }
        let tokenIDs = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
        let input = LMInput(tokens: MLXArray(tokenIDs))
        let uid = "cmpl-\(UUID().uuidString)"
        let mlxRequest = Request(
            uid: uid,
            input: input,
            maxTokens: request.maxTokens,
            sampling: SamplingParameters(
                temperature: request.temperature,
                topP: request.topP,
                topK: request.topK,
                minP: request.minP,
                repetitionPenalty: request.repetitionPenalty,
                presencePenalty: request.presencePenalty,
                frequencyPenalty: request.frequencyPenalty,
                xtcProbability: 0,
                xtcThreshold: 0.1,
                xtcSpecialTokens: xtcSpecialTokens(),
                seed: request.seed
            ),
            eosTokenIds: eosTokenIds
        )
        return makeStream(from: mlxRequest, promptTokens: tokenIDs.count)
    }

    private func makeStream(from mlxRequest: Request, promptTokens: Int) -> OpenAIChatStream {
        let responseStream = engine.stream(mlxRequest)
        let chunks = AsyncThrowingStream<OpenAIChatChunk, Error> { continuation in
            let task = Task {
                do {
                    for try await response in responseStream {
                        if case .failed(let message)? = response.finishReason {
                            throw NativeModelEngineError.generationFailed(message)
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

// NOTE: Cross-model concurrent MLX eval is not process-serialized here. Each
// model has its own scheduler; a process-wide MLX eval gate is a known follow-up
// if native serving needs to mirror omlx's single executor more tightly.
struct NativeModelLoader: EnginePoolModelLoader {
    let maxConcurrentRequests: Int

    func loadModel(id: String, modelURL: URL) async throws -> NativeModelEngine {
        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: #huggingFaceTokenizerLoader()
        )
        return await container.perform { context in
            NativeModelEngine(
                context: context,
                modelID: id,
                maxConcurrentRequests: maxConcurrentRequests
            )
        }
    }

    func unloadModel(_ engine: NativeModelEngine, id: String) async {
        Memory.clearCache()
    }
}

private enum NativeModelEngineError: Error, CustomStringConvertible {
    case generationFailed(String)
    case invalidPrompt

    var description: String {
        switch self {
        case .generationFailed(let message):
            return message
        case .invalidPrompt:
            return "completion backend requires a single string prompt"
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
