import Foundation
import MLXServeHTTP
import MLXServeSpeech

/// Serves `/v1/audio/transcriptions` from the speech engine registry — the
/// route stops 501ing the moment any adapter is registered.
struct RegistrySpeechBackend: AudioTranscriptionBackend {
    let registry: SpeechEngineRegistry
    let models: [OpenAIModelInfo]

    init(registry: SpeechEngineRegistry) async {
        self.registry = registry
        self.models = await registry.allModels().map { model in
            OpenAIModelInfo(id: model.id, maxModelLength: nil)
        }
    }

    var transcriptionModels: [OpenAIModelInfo] { models }

    func transcribe(_ request: AudioTranscriptionRequest) async throws -> AudioTranscriptionResult {
        let (adapter, modelID) = try await registry.resolve(model: request.model)
        let result = try await adapter.transcribeFile(
            SpeechFileTranscriptionRequest(
                model: modelID,
                fileName: request.fileName,
                fileData: request.fileData,
                language: request.language,
                temperature: request.temperature
            )
        )
        return AudioTranscriptionResult(
            text: result.text,
            language: result.language,
            duration: result.duration,
            segments: segments(from: result)
        )
    }

    private func segments(from result: SpeechTranscriptionResult) -> [AudioTranscriptionSegment]? {
        guard let words = result.words, !words.isEmpty else { return nil }
        return words.enumerated().map { index, word in
            AudioTranscriptionSegment(id: index, start: word.start, end: word.end, text: word.text)
        }
    }
}
