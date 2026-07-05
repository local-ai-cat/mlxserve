import Foundation
import MLXServe
@testable import MLXServeHTTP
@testable import MLXServeHTTPServer
import XCTest

final class RerankTests: XCTestCase {
    func testRerankRequestParsesStringDocumentsWithDefaults() throws {
        let request = try OpenAIRerankRequest.parse(
            Data(
                """
                {
                  "model": "rerank-model",
                  "query": "capital of France",
                  "documents": ["Paris is the capital.", "Bananas are yellow."]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.model, "rerank-model")
        XCTAssertEqual(request.query, .string("capital of France"))
        XCTAssertEqual(request.documents, [.string("Paris is the capital."), .string("Bananas are yellow.")])
        XCTAssertNil(request.topN)
        XCTAssertTrue(request.returnDocuments)
    }

    func testRerankRequestParsesObjectDocumentsTopNAndReturnDocuments() throws {
        let request = try OpenAIRerankRequest.parse(
            Data(
                """
                {
                  "model": "rerank-model",
                  "query": {"text": "capital of France"},
                  "documents": [{"text": "Paris", "title": "France"}, {"text": "Berlin"}],
                  "top_n": 1,
                  "return_documents": false
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.query, .object(["text": "capital of France"]))
        XCTAssertEqual(request.documents[0], .object(["text": "Paris", "title": "France"]))
        XCTAssertEqual(request.topN, 1)
        XCTAssertFalse(request.returnDocuments)
    }

    func testRerankResponseShapeSortAndTopN() async throws {
        let backend = FakeRerankBackend(scores: [0.2, 0.9, 0.5])
        let request = OpenAIRerankRequest(
            model: "rerank-model",
            query: .string("capital"),
            documents: [.string("weak"), .string("best"), .string("middle")],
            topN: 2
        )

        let response = buildRerankResponse(request: request, result: try await backend.rerank(request))

        XCTAssertEqual(response["model"] as? String, "rerank-model")
        XCTAssertTrue((response["id"] as? String)?.hasPrefix("rerank-") == true)
        let results = try XCTUnwrap(response["results"] as? [[String: Any]])
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0]["index"] as? Int, 1)
        XCTAssertEqual(results[0]["relevance_score"] as? Float, 0.9)
        XCTAssertEqual(results[0]["document"] as? [String: String], ["text": "best"])
        XCTAssertEqual(results[1]["index"] as? Int, 2)
        let usage = try XCTUnwrap(response["usage"] as? [String: Int])
        XCTAssertEqual(usage["total_tokens"], 42)
    }

    func testRerankResponseCanReturnNullDocuments() {
        let request = OpenAIRerankRequest(
            model: "rerank-model",
            query: .string("capital"),
            documents: [.string("Paris")],
            returnDocuments: false
        )
        let response = buildRerankResponse(
            request: request,
            result: OpenAIRerankResult(scores: [0.7], indices: [0], totalTokens: 1)
        )

        let results = response["results"] as? [[String: Any]]
        XCTAssertTrue(results?[0]["document"] is NSNull)
    }

    func testRerankUnknownModelReturnsNotFound() throws {
        let response = try XCTUnwrap(
            rerankModelAvailabilityResponse(
                requestedModel: "missing",
                rerankModels: [OpenAIModelInfo(id: "rerank-model")]
            )
        )

        XCTAssertEqual(response.status, 404)
        let error = try XCTUnwrap(response.body["error"] as? [String: Any])
        XCTAssertEqual(error["message"] as? String, "Rerank model 'missing' not found.")
        XCTAssertEqual(error["type"] as? String, "not_found_error")
    }

    func testRerankEmptyDocumentsIsBadRequest() {
        XCTAssertThrowsError(
            try OpenAIRerankRequest.parse(
                Data(
                    """
                    {
                      "model": "rerank-model",
                      "query": "capital",
                      "documents": []
                    }
                    """.utf8
                )
            )
        ) { error in
            XCTAssertEqual((error as? OpenAIServerError)?.httpStatus, 400)
        }
    }

    func testRerankModelDiscoveryClassifiesQwenRerankerDirectories() throws {
        let directory = try temporaryModelDirectory(
            name: "Qwen3-Reranker-0.6B",
            config: [
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
            ]
        )

        XCTAssertTrue(NativeRerankBackend.isRerankModelDirectory(directory))
    }

    func testNativeRerankSmokeWhenModelGateIsSet() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_RERANK_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_RERANK_TEST_MODEL to run the rerank scoring smoke test.")
        }
        try MLXMetalRuntime.requireAvailable()

        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        let model = DiscoveredModel(
            id: modelURL.lastPathComponent,
            modelURL: modelURL,
            estimatedSize: 0
        )
        let backend = NativeRerankBackend(models: [model.id: model])
        let request = OpenAIRerankRequest(
            model: model.id,
            query: .string("capital of France"),
            documents: [.string("Paris is the capital of France."), .string("Bananas are yellow.")],
            topN: 1
        )

        let result = try await backend.rerank(request)

        XCTAssertEqual(result.scores.count, 2)
        XCTAssertEqual(result.indices.count, 1)
        XCTAssertTrue(result.scores.allSatisfy { $0.isFinite })
        XCTAssertGreaterThan(result.totalTokens, 0)
    }

    private func temporaryModelDirectory(name: String, config: [String: Any]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: directory.appendingPathComponent("config.json"))
        return directory
    }
}

private struct FakeRerankBackend: OpenAIRerankBackend {
    let rerankModels = [OpenAIModelInfo(id: "rerank-model")]
    let scores: [Float]

    func rerank(_ request: OpenAIRerankRequest) async throws -> OpenAIRerankResult {
        let sortedIndices = scores.indices.sorted {
            if scores[$0] == scores[$1] {
                return $0 < $1
            }
            return scores[$0] > scores[$1]
        }
        let indices = request.topN.map { Array(sortedIndices.prefix($0)) } ?? sortedIndices
        return OpenAIRerankResult(scores: scores, indices: indices, totalTokens: 42)
    }
}
