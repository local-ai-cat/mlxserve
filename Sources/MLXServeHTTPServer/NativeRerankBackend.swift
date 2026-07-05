import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import MLXServeHTTP
import Tokenizers

final class NativeRerankBackend: OpenAIRerankBackend, @unchecked Sendable {
    let rerankModels: [OpenAIModelInfo]

    private let modelURLsByID: [String: URL]
    private let cache = NativeRerankModelCache()

    init(models: [String: DiscoveredModel]) {
        self.modelURLsByID = models.mapValues(\.modelURL)
        self.rerankModels = models.keys.sorted().map { OpenAIModelInfo(id: $0, maxModelLength: nil) }
    }

    func rerank(_ request: OpenAIRerankRequest) async throws -> OpenAIRerankResult {
        guard let modelURL = modelURLsByID[request.model] else {
            throw OpenAIHTTPError(status: 404, message: "Rerank model '\(request.model)' not found.")
        }
        let container = try await cache.container(for: request.model, modelURL: modelURL)
        return try await NativeRerankScorer.score(request, container: container)
    }

    static func isRerankModelDirectory(_ url: URL) -> Bool {
        let name = url.lastPathComponent.lowercased()
        guard name.contains("rerank") || name.contains("reranker") else {
            return false
        }
        guard
            let data = try? Data(contentsOf: url.appendingPathComponent("config.json")),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return true
        }

        let modelType = (object["model_type"] as? String)?.lowercased() ?? ""
        let architectures = (object["architectures"] as? [String] ?? []).map { $0.lowercased() }
        return modelType.contains("qwen")
            || architectures.contains(where: { $0.contains("sequenceclassification") || $0.contains("causallm") })
    }
}

private actor NativeRerankModelCache {
    private var containers: [String: ModelContainer] = [:]

    func container(for id: String, modelURL: URL) async throws -> ModelContainer {
        if let container = containers[id] {
            return container
        }
        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: #huggingFaceTokenizerLoader()
        )
        containers[id] = container
        return container
    }
}

private enum NativeRerankScorer {
    private static let maxLength = 8_192
    private static let systemPrompt =
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    private static let instruction =
        "Given a web search query, retrieve relevant passages that answer the query"
    private static let suffix = "<think>\n\n</think>\n\n"
    private static let sentinel = "<<__MLXSERVE_RERANK_CONTENT__>>"

    static func score(_ request: OpenAIRerankRequest, container: ModelContainer) async throws -> OpenAIRerankResult {
        try await container.perform { context in
            guard
                let yesTokenID = context.tokenizer.convertTokenToId("yes"),
                let noTokenID = context.tokenizer.convertTokenToId("no")
            else {
                throw OpenAIHTTPError(status: 400, message: "rerank tokenizer is missing yes/no token IDs")
            }

            let template = try promptTemplate(tokenizer: context.tokenizer)
            var scores: [Float] = []
            var totalTokens = 0
            scores.reserveCapacity(request.documents.count)

            for document in request.documents {
                let content = "<Instruct>: \(instruction)\n<Query>: \(request.query.text)\n<Document>: \(document.text)"
                let contentTokens = context.tokenizer.encode(text: content, addSpecialTokens: false)
                let budget = max(0, maxLength - template.prefix.count - template.suffix.count)
                let promptTokens = template.prefix + contentTokens.prefix(budget) + template.suffix
                totalTokens += promptTokens.count

                let input = MLXArray(Array(promptTokens))[.newAxis, 0...]
                let output = context.model(input, cache: nil)
                let logits = output[0, -1, 0...]
                logits.eval()
                let noLogit = Double(logits[noTokenID].item(Float.self))
                let yesLogit = Double(logits[yesTokenID].item(Float.self))
                scores.append(Float(softmaxYes(noLogit: noLogit, yesLogit: yesLogit)))
            }

            let sortedIndices = scores.indices.sorted {
                if scores[$0] == scores[$1] {
                    return $0 < $1
                }
                return scores[$0] > scores[$1]
            }
            let topIndices = request.topN.map { Array(sortedIndices.prefix(min($0, sortedIndices.count))) }
                ?? sortedIndices
            return OpenAIRerankResult(scores: scores, indices: topIndices, totalTokens: totalTokens)
        }
    }

    private static func promptTemplate(tokenizer: any MLXLMCommon.Tokenizer) throws -> (prefix: [Int], suffix: [Int]) {
        let templateTokens = try tokenizer.applyChatTemplate(
            messages: [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": sentinel],
            ],
            tools: nil,
            additionalContext: ["add_generation_prompt": true]
        )
        let sentinelTokens = tokenizer.encode(text: sentinel, addSpecialTokens: false)
        guard
            !sentinelTokens.isEmpty,
            let range = templateTokens.firstRange(of: sentinelTokens)
        else {
            throw OpenAIHTTPError(status: 400, message: "rerank tokenizer chat template could not be prepared")
        }

        let suffixTokens = tokenizer.encode(text: suffix, addSpecialTokens: false)
        return (
            prefix: Array(templateTokens[..<range.lowerBound]),
            suffix: Array(templateTokens[range.upperBound...]) + suffixTokens
        )
    }

    private static func softmaxYes(noLogit: Double, yesLogit: Double) -> Double {
        let maximum = max(noLogit, yesLogit)
        let noExp = exp(noLogit - maximum)
        let yesExp = exp(yesLogit - maximum)
        return yesExp / (noExp + yesExp)
    }
}

private extension Array where Element: Equatable {
    func firstRange(of needle: [Element]) -> Range<Int>? {
        guard !needle.isEmpty, needle.count <= count else {
            return nil
        }
        for start in 0...(count - needle.count) {
            if Array(self[start..<(start + needle.count)]) == needle {
                return start..<(start + needle.count)
            }
        }
        return nil
    }
}
