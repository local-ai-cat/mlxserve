import Foundation

public enum OpenAIRerankText: Sendable, Equatable {
    case string(String)
    case object([String: String])

    public var text: String {
        switch self {
        case .string(let value):
            return value
        case .object(let value):
            return value["text"] ?? ""
        }
    }

    public var documentPayload: [String: String] {
        switch self {
        case .string(let value):
            return ["text": value]
        case .object(let value):
            return value
        }
    }
}

public struct OpenAIRerankRequest: Sendable, Equatable {
    public let model: String
    public let query: OpenAIRerankText
    public let documents: [OpenAIRerankText]
    public let topN: Int?
    public let returnDocuments: Bool

    public init(
        model: String,
        query: OpenAIRerankText,
        documents: [OpenAIRerankText],
        topN: Int? = nil,
        returnDocuments: Bool = true
    ) {
        self.model = model
        self.query = query
        self.documents = documents
        self.topN = topN
        self.returnDocuments = returnDocuments
    }

    public static func parse(_ body: Data) throws -> OpenAIRerankRequest {
        guard
            let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
            let model = object["model"] as? String,
            !model.isEmpty
        else {
            throw OpenAIServerError.invalidJSON
        }

        let query = try parseText(object["query"])
        let documents = try parseDocuments(object["documents"])
        guard !query.text.isEmpty else { throw OpenAIServerError.invalidRequest }
        guard !documents.isEmpty else { throw OpenAIServerError.invalidRequest }

        let topN = openAIIntValue(object["top_n"])
        if let topN, topN <= 0 {
            throw OpenAIServerError.invalidJSON
        }

        let returnDocuments = object["return_documents"] as? Bool ?? true
        return OpenAIRerankRequest(
            model: model,
            query: query,
            documents: documents,
            topN: topN,
            returnDocuments: returnDocuments
        )
    }

    private static func parseDocuments(_ value: Any?) throws -> [OpenAIRerankText] {
        if let strings = value as? [String] {
            return strings.map(OpenAIRerankText.string)
        }
        if let objects = value as? [[String: String]] {
            return objects.map(OpenAIRerankText.object)
        }
        if let values = value as? [Any] {
            return try values.map(parseText)
        }
        throw OpenAIServerError.invalidJSON
    }

    private static func parseText(_ value: Any?) throws -> OpenAIRerankText {
        if let string = value as? String {
            return .string(string)
        }
        if let object = value as? [String: String] {
            return .object(object)
        }
        throw OpenAIServerError.invalidJSON
    }
}

public struct OpenAIRerankResult: Sendable, Equatable {
    public let scores: [Float]
    public let indices: [Int]
    public let totalTokens: Int

    public init(scores: [Float], indices: [Int], totalTokens: Int) {
        self.scores = scores
        self.indices = indices
        self.totalTokens = totalTokens
    }
}

public protocol OpenAIRerankBackend: Sendable {
    var rerankModels: [OpenAIModelInfo] { get }
    func rerank(_ request: OpenAIRerankRequest) async throws -> OpenAIRerankResult
}
