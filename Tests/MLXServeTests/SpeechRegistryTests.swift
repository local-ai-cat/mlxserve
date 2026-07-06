import Foundation
@testable import MLXServeHTTP
@testable import MLXServeHTTPServer
@testable import MLXServeSpeech
import XCTest

// MARK: - Fake adapter

/// In-memory adapter used both to test the registry and as the reference
/// implementation of the conformance battery below.
final class FakeSpeechAdapter: SpeechEngineAdapter, @unchecked Sendable {
    let engineID: String
    let capabilities: SpeechEngineCapabilities
    private let models: [SpeechModelInfo]

    var displayName: String { "Fake (\(engineID))" }

    init(
        engineID: String,
        silicon: SpeechEngineSilicon,
        streaming: Bool = true,
        modelIDs: [String]
    ) {
        self.engineID = engineID
        self.capabilities = SpeechEngineCapabilities(
            supportsFileTranscription: true,
            supportsStreaming: streaming,
            silicon: silicon,
            wordTimestamps: true
        )
        self.models = modelIDs.map {
            SpeechModelInfo(id: $0, engineID: engineID, displayName: $0)
        }
    }

    func availableModels() async -> [SpeechModelInfo] { models }

    func transcribeFile(_ request: SpeechFileTranscriptionRequest) async throws -> SpeechTranscriptionResult {
        SpeechTranscriptionResult(
            text: "fake transcript via \(engineID)/\(request.model)",
            language: request.language ?? "en",
            duration: 1.0,
            words: [
                SpeechWord(text: "fake", start: 0.0, end: 0.4),
                SpeechWord(text: "transcript", start: 0.4, end: 1.0),
            ]
        )
    }

    func makeStreamSession(modelID: String, language: String?) async throws -> any SpeechStreamSession {
        guard capabilities.supportsStreaming else {
            throw SpeechEngineError.streamingUnsupported(engineID: engineID)
        }
        return FakeStreamSession()
    }
}

/// Emits one partial per pushed second of audio, then finalizes on finish() —
/// exercising the partial→final identity contract.
actor FakeStreamSession: SpeechStreamSession {
    nonisolated let segments: AsyncThrowingStream<SpeechStreamSegment, Error>
    private let continuation: AsyncThrowingStream<SpeechStreamSegment, Error>.Continuation
    private var fedSeconds: Double = 0
    private var emissions = 0

    init() {
        var continuation: AsyncThrowingStream<SpeechStreamSegment, Error>.Continuation!
        self.segments = AsyncThrowingStream { continuation = $0 }
        self.continuation = continuation
    }

    func push(_ buffer: SpeechPCMBuffer) async throws {
        fedSeconds += buffer.duration
        emissions += 1
        continuation.yield(
            SpeechStreamSegment(
                id: 0,
                kind: .partial,
                text: "partial-\(emissions)",
                start: 0,
                end: fedSeconds
            )
        )
    }

    func finish() async throws {
        continuation.yield(
            SpeechStreamSegment(
                id: 0,
                kind: .final,
                text: "final transcript",
                start: 0,
                end: fedSeconds,
                words: [SpeechWord(text: "final", start: 0, end: fedSeconds)]
            )
        )
        continuation.finish()
    }

    func cancel() async {
        continuation.finish()
    }

    func latencyStats() async -> SpeechSessionLatencyStats {
        SpeechSessionLatencyStats(sampleCount: emissions, p50Seconds: 0.1, p95Seconds: 0.2)
    }
}

/// Claims models in its catalog but fails every load — the corrupt-folder case.
final class BrokenSpeechAdapter: SpeechEngineAdapter, @unchecked Sendable {
    let engineID: String
    let capabilities: SpeechEngineCapabilities
    private let models: [SpeechModelInfo]

    var displayName: String { "Broken (\(engineID))" }

    init(engineID: String, silicon: SpeechEngineSilicon, modelIDs: [String]) {
        self.engineID = engineID
        self.capabilities = SpeechEngineCapabilities(
            supportsFileTranscription: true,
            supportsStreaming: false,
            silicon: silicon
        )
        self.models = modelIDs.map { SpeechModelInfo(id: $0, engineID: engineID, displayName: $0) }
    }

    func availableModels() async -> [SpeechModelInfo] { models }

    func transcribeFile(_ request: SpeechFileTranscriptionRequest) async throws -> SpeechTranscriptionResult {
        throw SpeechEngineError.engineFailure("corrupt model folder")
    }

    func makeStreamSession(modelID: String, language: String?) async throws -> any SpeechStreamSession {
        throw SpeechEngineError.streamingUnsupported(engineID: engineID)
    }
}

// MARK: - Registry tests

final class SpeechRegistryTests: XCTestCase {
    func testResolveFindsModelAcrossAdapters() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny", "base"]))
        await registry.register(FakeSpeechAdapter(engineID: "parakeet", silicon: .ane, modelIDs: ["nemotron"]))

        let (adapter, modelID) = try await registry.resolve(model: "nemotron")
        XCTAssertEqual(adapter.engineID, "parakeet")
        XCTAssertEqual(modelID, "nemotron")
    }

    func testResolveUnknownModelThrows() async {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny"]))

        do {
            _ = try await registry.resolve(model: "nope")
            XCTFail("expected unknownModel")
        } catch let error as SpeechEngineError {
            XCTAssertEqual(error, .unknownModel("nope"))
        } catch {
            XCTFail("unexpected error \(error)")
        }
    }

    func testExplicitEngineNamespaceDisambiguates() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny"]))
        await registry.register(FakeSpeechAdapter(engineID: "whispercpp", silicon: .cpu, modelIDs: ["tiny"]))

        let (adapter, _) = try await registry.resolve(model: "whispercpp:tiny")
        XCTAssertEqual(adapter.engineID, "whispercpp")
    }

    func testSiliconPreferenceSteersResolution() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "mlxasr", silicon: .gpu, modelIDs: ["shared"]))
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["shared"]))

        let (defaulted, _) = try await registry.resolve(model: "shared")
        XCTAssertEqual(defaulted.engineID, "mlxasr", "registration order wins without a preference")

        let (preferred, _) = try await registry.resolve(
            model: "shared",
            preferences: SpeechSessionPreferences(preferredSilicon: .ane)
        )
        XCTAssertEqual(preferred.engineID, "whisperkit")
    }

    func testRequireStreamingFiltersNonStreamingAdapters() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(
            FakeSpeechAdapter(engineID: "fileonly", silicon: .cpu, streaming: false, modelIDs: ["m"])
        )
        await registry.register(FakeSpeechAdapter(engineID: "streamer", silicon: .ane, modelIDs: ["m"]))

        let (adapter, _) = try await registry.resolve(
            model: "m",
            preferences: SpeechSessionPreferences(requireStreaming: true)
        )
        XCTAssertEqual(adapter.engineID, "streamer")
    }

    func testModelReferenceSplitting() {
        XCTAssertEqual(SpeechEngineRegistry.splitModelReference("tiny").engineID, nil)
        XCTAssertEqual(SpeechEngineRegistry.splitModelReference("wk:tiny").engineID, "wk")
        XCTAssertEqual(SpeechEngineRegistry.splitModelReference("wk:tiny").modelID, "tiny")
        XCTAssertEqual(SpeechEngineRegistry.splitModelReference(":tiny").engineID, nil)
    }
}

// MARK: - Stream session conformance

final class SpeechStreamSessionConformanceTests: XCTestCase {
    func testPartialsThenFinalWithStableIdentityAndMonotoneTimes() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "fake", silicon: .ane, modelIDs: ["m"]))
        let (adapter, modelID) = try await registry.resolve(model: "m")
        let session = try await adapter.makeStreamSession(modelID: modelID, language: nil)

        let tone = SpeechPCMBuffer(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000)
        try await session.push(tone)
        try await session.push(tone)
        try await session.finish()

        var collected: [SpeechStreamSegment] = []
        for try await segment in session.segments {
            collected.append(segment)
        }

        XCTAssertEqual(collected.count, 3)
        XCTAssertEqual(collected.map(\.kind), [.partial, .partial, .final])
        XCTAssertEqual(Set(collected.map(\.id)).count, 1, "final must share the partials' identity")
        let ends = collected.map(\.end)
        XCTAssertEqual(ends, ends.sorted(), "segment end times must be monotone")
        XCTAssertEqual(collected.last?.end ?? 0, 2.0, accuracy: 0.001)

        let stats = await session.latencyStats()
        XCTAssertGreaterThan(stats.sampleCount, 0)
    }
}

// MARK: - HTTP bridge

final class RegistrySpeechBackendTests: XCTestCase {
    func testBridgeServesRegistryModelsAndTranscribes() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny"]))
        let backend = await RegistrySpeechBackend(registry: registry)

        XCTAssertEqual(backend.transcriptionModels.map(\.id), ["tiny"])

        let result = try await backend.transcribe(
            AudioTranscriptionRequest(model: "tiny", fileName: "t.wav", fileData: Data([0x00]))
        )
        XCTAssertEqual(result.text, "fake transcript via whisperkit/tiny")
        XCTAssertEqual(result.segments?.count, 2)
        XCTAssertEqual(result.segments?.first?.start, 0.0)
    }

    func testBridgeUnknownModelMapsTo404WithModelList() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny"]))
        let backend = await RegistrySpeechBackend(registry: registry)

        do {
            _ = try await backend.transcribe(
                AudioTranscriptionRequest(model: "ghost", fileName: "t.wav", fileData: Data())
            )
            XCTFail("expected 404")
        } catch let error as OpenAIHTTPError {
            XCTAssertEqual(error.status, 404)
            XCTAssertTrue(error.message.contains("tiny"), "404 body should list available models")
        }
    }

    func testBridgeFallsThroughToNextCandidateOnLoadFailure() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(
            BrokenSpeechAdapter(engineID: "corrupt-ane", silicon: .ane, modelIDs: ["shared"])
        )
        await registry.register(FakeSpeechAdapter(engineID: "healthy", silicon: .cpu, modelIDs: ["shared"]))
        let backend = await RegistrySpeechBackend(registry: registry)

        let result = try await backend.transcribe(
            AudioTranscriptionRequest(model: "shared", fileName: "t.wav", fileData: Data([0x00]))
        )
        XCTAssertEqual(result.text, "fake transcript via healthy/shared")
    }
}
