import Foundation
import MLXServeSpeech
import WhisperKit

/// WhisperKit (CoreML/ANE) behind the speech registry. Phase-1 adapter: wraps
/// the engine as-is — model discovery scans a local models root, load is
/// per-model lazy, streaming is windowed re-transcription over pushed PCM.
public actor WhisperKitSpeechAdapter: SpeechEngineAdapter {
    public nonisolated let engineID = "whisperkit"
    public nonisolated var displayName: String { "WhisperKit (CoreML/ANE)" }

    public nonisolated var capabilities: SpeechEngineCapabilities {
        SpeechEngineCapabilities(
            supportsFileTranscription: true,
            supportsStreaming: true,
            silicon: .ane,
            wordTimestamps: true,
            confidence: true
        )
    }

    private let modelsRoot: URL?
    private let allowDownload: Bool
    private var loaded: [String: WhisperKitPipeline] = [:]

    /// - Parameters:
    ///   - modelsRoot: directory whose subdirectories are WhisperKit model
    ///     folders (compiled `.mlmodelc` bundles). nil = WhisperKit's default
    ///     resolution (download-on-demand when `allowDownload`).
    ///   - allowDownload: permit WhisperKit to fetch a model that isn't local.
    public init(modelsRoot: URL? = nil, allowDownload: Bool = false) {
        self.modelsRoot = modelsRoot
        self.allowDownload = allowDownload
    }

    public func availableModels() -> [SpeechModelInfo] {
        guard let modelsRoot else { return [] }
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: modelsRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return contents
            .filter { url in
                (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            }
            .map { url in
                SpeechModelInfo(
                    id: url.lastPathComponent,
                    engineID: engineID,
                    displayName: url.lastPathComponent
                )
            }
            .sorted { $0.id < $1.id }
    }

    public func loadModel(_ modelID: String) async throws {
        _ = try await pipeline(for: modelID)
    }

    public func unloadModel(_ modelID: String) {
        loaded[modelID] = nil
    }

    public func loadedFootprint() -> Int64 {
        // CoreML working-set size isn't queryable; report model count as a
        // placeholder until pool citizenship (M8b-2) defines accounting.
        Int64(loaded.count)
    }

    public func transcribeFile(_ request: SpeechFileTranscriptionRequest) async throws -> SpeechTranscriptionResult {
        let pipeline = try await pipeline(for: request.model)

        let temporaryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxserve-\(UUID().uuidString)-\(request.fileName)")
        try request.fileData.write(to: temporaryURL)
        defer { try? FileManager.default.removeItem(at: temporaryURL) }

        let results = try await pipeline.transcribe(
            audioPath: temporaryURL.path,
            language: request.language,
            temperature: request.temperature
        )
        return Self.merge(results)
    }

    public func makeStreamSession(modelID: String, language: String?) async throws -> any SpeechStreamSession {
        let pipeline = try await pipeline(for: modelID)
        return WhisperKitStreamSession(pipeline: pipeline, language: language)
    }

    private func pipeline(for modelID: String) async throws -> WhisperKitPipeline {
        if let existing = loaded[modelID] {
            return existing
        }
        if let modelsRoot {
            let folder = modelsRoot.appendingPathComponent(modelID)
            if FileManager.default.fileExists(atPath: folder.path) {
                let pipeline = try await WhisperKitPipeline(modelFolder: folder.path)
                loaded[modelID] = pipeline
                return pipeline
            }
        }
        guard allowDownload else {
            throw SpeechEngineError.unknownModel(modelID)
        }
        let pipeline = try await WhisperKitPipeline(model: modelID)
        loaded[modelID] = pipeline
        return pipeline
    }

    static func merge(_ results: [TranscriptionResult]) -> SpeechTranscriptionResult {
        let text = results.map(\.text).joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let words: [SpeechWord] = results.flatMap { result in
            result.allWords.map { timing in
                SpeechWord(
                    text: timing.word.trimmingCharacters(in: .whitespaces),
                    start: Double(timing.start),
                    end: Double(timing.end),
                    confidence: timing.probability
                )
            }
        }
        let duration = results.compactMap { result in
            result.segments.map(\.end).max().map(Double.init)
        }.max()
        return SpeechTranscriptionResult(
            text: text,
            language: results.first?.language,
            duration: duration,
            words: words.isEmpty ? nil : words
        )
    }
}

/// Owns one non-Sendable `WhisperKit` instance and serializes every transcribe
/// call through it — the unit both the file lane and stream sessions share.
actor WhisperKitPipeline {
    private let whisperKit: WhisperKit

    init(modelFolder: String) async throws {
        self.whisperKit = try await WhisperKit(WhisperKitConfig(modelFolder: modelFolder, load: true))
    }

    init(model: String) async throws {
        self.whisperKit = try await WhisperKit(WhisperKitConfig(model: model, load: true))
    }

    func transcribe(
        audioPath: String,
        language: String?,
        temperature: Float
    ) async throws -> [TranscriptionResult] {
        try await whisperKit.transcribe(
            audioPath: audioPath,
            decodeOptions: options(language: language, temperature: temperature)
        )
    }

    func transcribe(
        audioArray: [Float],
        language: String?
    ) async throws -> [TranscriptionResult] {
        try await whisperKit.transcribe(
            audioArray: audioArray,
            decodeOptions: options(language: language, temperature: 0)
        )
    }

    private func options(language: String?, temperature: Float) -> DecodingOptions {
        DecodingOptions(
            task: .transcribe,
            language: language,
            temperature: temperature,
            wordTimestamps: true
        )
    }
}

/// Windowed re-transcription streaming: pushed PCM accumulates; every
/// `partialInterval` seconds of new audio the whole window is re-transcribed
/// and emitted as a revised partial (stable id 0); finish() runs the final
/// pass and commits. Capture stays with the caller — only PCM arrives here.
actor WhisperKitStreamSession: SpeechStreamSession {
    nonisolated let segments: AsyncThrowingStream<SpeechStreamSegment, Error>
    private let continuation: AsyncThrowingStream<SpeechStreamSegment, Error>.Continuation

    private let pipeline: WhisperKitPipeline
    private let language: String?
    private let partialInterval: Double

    private var samples: [Float] = []
    private var sampleRate: Double = 16_000
    private var secondsSinceLastPartial: Double = 0
    private var latencySamples: [Double] = []
    private var finished = false

    init(pipeline: WhisperKitPipeline, language: String?, partialInterval: Double = 2.0) {
        self.pipeline = pipeline
        self.language = language
        self.partialInterval = partialInterval
        var continuation: AsyncThrowingStream<SpeechStreamSegment, Error>.Continuation!
        self.segments = AsyncThrowingStream { continuation = $0 }
        self.continuation = continuation
    }

    func push(_ buffer: SpeechPCMBuffer) async throws {
        guard !finished else { return }
        sampleRate = buffer.sampleRate
        samples.append(contentsOf: buffer.samples)
        secondsSinceLastPartial += buffer.duration
        if secondsSinceLastPartial >= partialInterval {
            secondsSinceLastPartial = 0
            try await emit(kind: .partial)
        }
    }

    func finish() async throws {
        guard !finished else { return }
        finished = true
        try await emit(kind: .final)
        continuation.finish()
    }

    func cancel() {
        finished = true
        continuation.finish()
    }

    func latencyStats() -> SpeechSessionLatencyStats {
        guard !latencySamples.isEmpty else { return .empty }
        let sorted = latencySamples.sorted()
        return SpeechSessionLatencyStats(
            sampleCount: sorted.count,
            p50Seconds: sorted[sorted.count / 2],
            p95Seconds: sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]
        )
    }

    private func emit(kind: SpeechSegmentKind) async throws {
        guard !samples.isEmpty else {
            if kind == .final {
                continuation.yield(SpeechStreamSegment(id: 0, kind: .final, text: "", start: 0, end: 0))
            }
            return
        }
        let fedSeconds = Double(samples.count) / sampleRate
        let started = DispatchTime.now()
        let results = try await pipeline.transcribe(audioArray: samples, language: language)
        let merged = WhisperKitSpeechAdapter.merge(results)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1_000_000_000
        latencySamples.append(elapsed)
        continuation.yield(
            SpeechStreamSegment(
                id: 0,
                kind: kind,
                text: merged.text,
                start: 0,
                end: fedSeconds,
                words: merged.words
            )
        )
    }
}
