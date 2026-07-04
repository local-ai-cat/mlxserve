import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import MLXServeHTTP
import MLXVLM
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
        maxConcurrentRequests: Int,
        serializedDecode: Bool
    ) {
        self.context = context
        self.modelID = modelID
        self.parameters = GenerateParameters(maxTokens: 16, temperature: 0)
        self.engine = MLXServeEngine(
            model: context.model,
            parameters: parameters,
            maxConcurrentRequests: maxConcurrentRequests,
            serializedDecode: serializedDecode
        )
        var eosTokenIds = context.configuration.eosTokenIds
        if let tokenizerEosTokenId = context.tokenizer.eosTokenId {
            eosTokenIds.insert(tokenizerEosTokenId)
        }
        self.eosTokenIds = eosTokenIds
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        let input = try await context.processor.prepare(input: try userInput(from: request))
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

    private func userInput(from request: OpenAIChatRequest) throws -> UserInput {
        UserInput(
            chat: try request.messages.map(chatMessage),
            tools: toolSpecDictionaries(
                from: selectOpenAITools(tools: request.tools, toolChoice: request.toolChoice)
            ),
            additionalContext: additionalContext(from: request)
        )
    }

    private func chatMessage(from message: OpenAIChatMessage) throws -> Chat.Message {
        let role = Chat.Message.Role(rawValue: message.role) ?? .user
        let images = try message.imageReferences.map(image)
        // TODO(M6b): render prior assistant tool_calls once OpenAIChatMessage carries them.
        return Chat.Message(role: role, content: message.content, images: images)
    }

    private func image(from reference: OpenAIChatImageReference) throws -> UserInput.Image {
        switch reference.source {
        case .dataURI(let dataURI):
            let image = try ciImage(fromDataURI: dataURI)
            return .ciImage(image)
        case .url(let value):
            guard let url = imageURL(from: value) else {
                throw NativeModelEngineError.invalidImageReference(value)
            }
            return .url(url)
        }
    }

    private func ciImage(fromDataURI dataURI: String) throws -> CIImage {
        guard let commaIndex = dataURI.firstIndex(of: ",") else {
            throw NativeModelEngineError.invalidImageReference("data URI is missing image data")
        }

        let metadata = dataURI[..<commaIndex].lowercased()
        guard metadata.hasPrefix("data:"), metadata.contains(";base64") else {
            throw NativeModelEngineError.invalidImageReference("data URI image data must be base64 encoded")
        }

        let encodedData = String(dataURI[dataURI.index(after: commaIndex)...])
        guard let data = Data(base64Encoded: encodedData, options: [.ignoreUnknownCharacters]),
            let image = CIImage(data: data)
        else {
            throw NativeModelEngineError.invalidImageReference("data URI image data could not be decoded")
        }
        return image
    }

    private func imageURL(from value: String) -> URL? {
        if let url = URL(string: value),
            let scheme = url.scheme?.lowercased(),
            ["http", "https", "file"].contains(scheme)
        {
            return url
        }
        if value.hasPrefix("/") {
            return URL(fileURLWithPath: value)
        }
        return nil
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

    private func toolSpecDictionaries(from tools: [OpenAIJSONValue]?) -> [[String: any Sendable]]? {
        guard let tools else { return nil }

        var toolSpecs: [[String: any Sendable]] = []
        for tool in tools {
            guard case .object(let object) = tool else { continue }
            var converted: [String: any Sendable] = [:]
            for (key, value) in object {
                converted[key] = sendableValue(from: value)
            }
            toolSpecs.append(converted)
        }
        return toolSpecs.isEmpty ? nil : toolSpecs
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
        let isVLM = try isVLMModelDirectory(modelURL)
        let container =
            if isVLM {
                try await VLMModelFactory.shared.loadContainer(
                    from: modelURL,
                    using: #huggingFaceTokenizerLoader()
                )
            } else {
                try await LLMModelFactory.shared.loadContainer(
                    from: modelURL,
                    using: #huggingFaceTokenizerLoader()
                )
            }
        return await container.perform { context in
            NativeModelEngine(
                context: context,
                modelID: id,
                maxConcurrentRequests: maxConcurrentRequests,
                serializedDecode: isVLM
            )
        }
    }

    func unloadModel(_ engine: NativeModelEngine, id: String) async {
        Memory.clearCache()
    }

    private func isVLMModelDirectory(_ modelURL: URL) throws -> Bool {
        let configURL = modelURL.appending(component: "config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder.json5().decode(ModelKindConfiguration.self, from: configData)
        let modelType = config.modelType.lowercased()

        if let processorClass = processorClass(in: modelURL),
            Self.vlmProcessorClasses.contains(processorClass)
        {
            return true
        }

        guard Self.vlmModelTypes.contains(modelType) else {
            return false
        }
        return hasProcessorConfiguration(in: modelURL)
    }

    private func processorClass(in modelURL: URL) -> String? {
        for url in processorConfigurationURLs(in: modelURL) {
            guard FileManager.default.fileExists(atPath: url.path) else {
                continue
            }
            guard let data = try? Data(contentsOf: url),
                let config = try? JSONDecoder.json5().decode(ProcessorKindConfiguration.self, from: data)
            else {
                continue
            }
            return config.processorClass
        }
        return nil
    }

    private func hasProcessorConfiguration(in modelURL: URL) -> Bool {
        processorConfigurationURLs(in: modelURL).contains { url in
            FileManager.default.fileExists(atPath: url.path)
        }
    }

    private func processorConfigurationURLs(in modelURL: URL) -> [URL] {
        [
            modelURL.appending(component: "preprocessor_config.json"),
            modelURL.appending(component: "processor_config.json"),
        ]
    }

    private struct ModelKindConfiguration: Decodable {
        let modelType: String

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
        }
    }

    private struct ProcessorKindConfiguration: Decodable {
        let processorClass: String

        enum CodingKeys: String, CodingKey {
            case processorClass = "processor_class"
        }
    }

    private static let vlmModelTypes: Set<String> = [
        "paligemma",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "idefics3",
        "gemma3",
        "gemma4",
        "smolvlm",
        "fastvlm",
        "llava_qwen2",
        "pixtral",
        "mistral3",
        "lfm2_vl",
        "lfm2-vl",
        "glm_ocr",
    ]

    private static let vlmProcessorClasses: Set<String> = [
        "PaliGemmaProcessor",
        "Qwen2VLProcessor",
        "Qwen2_5_VLProcessor",
        "Qwen3VLProcessor",
        "Idefics3Processor",
        "Gemma3Processor",
        "Gemma4Processor",
        "SmolVLMProcessor",
        "FastVLMProcessor",
        "PixtralProcessor",
        "Mistral3Processor",
        "Lfm2VlProcessor",
        "Glm46VProcessor",
    ]
}

private enum NativeModelEngineError: Error, CustomStringConvertible {
    case generationFailed(String)
    case invalidPrompt
    case invalidImageReference(String)

    var description: String {
        switch self {
        case .generationFailed(let message):
            return message
        case .invalidPrompt:
            return "completion backend requires a single string prompt"
        case .invalidImageReference(let message):
            return "invalid image reference: \(message)"
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
