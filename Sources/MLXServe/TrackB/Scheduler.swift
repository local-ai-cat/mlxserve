import MLX
import MLXLMCommon

public final class LanguageModelBox: @unchecked Sendable {
    let model: any LanguageModel

    public init(_ model: any LanguageModel) {
        self.model = model
    }
}

public actor Scheduler {
    private let generator: ContinuousBatchGenerator
    private let maxConcurrentRequests: Int
    private let queueLimit: Int
    private var waiting: [Request] = []
    private var running: [String: RunningRequest] = [:]
    private var pendingCancellation: Set<String> = []
    private let collector = OutputCollector()

    public init(
        modelBox: LanguageModelBox,
        parameters: GenerateParameters,
        maxConcurrentRequests: Int
    ) {
        self.generator = ContinuousBatchGenerator(model: modelBox.model, parameters: parameters)
        self.maxConcurrentRequests = maxConcurrentRequests
        self.queueLimit = max(maxConcurrentRequests * 4, 32)
    }

    public var isIdle: Bool {
        waiting.isEmpty && running.isEmpty && generator.isEmpty
    }

    public var queueDepth: Int {
        waiting.count + running.count
    }

    public func submit(_ request: Request) throws {
        if queueDepth >= queueLimit {
            throw SchedulerError.queueFull(retryAfterSteps: max(1, queueDepth / max(1, maxConcurrentRequests)))
        }
        if waiting.contains(where: { $0.uid == request.uid }) || running[request.uid] != nil {
            throw SchedulerError.duplicateRequest(request.uid)
        }
        waiting.append(request)
    }

    public func cancel(uid: String) {
        if let waitingIndex = waiting.firstIndex(where: { $0.uid == uid }) {
            waiting.remove(at: waitingIndex)
            collector.record(Response(uid: uid, token: -1, finishReason: .cancelled))
            return
        }
        if running[uid] != nil {
            pendingCancellation.insert(uid)
        }
    }

    public func step() throws -> [Response] {
        applyPendingCancellation()
        try admitWaiting()

        guard !generator.isEmpty else { return [] }

        let rawResponses = generator.next()
        var processed: [Response] = []
        var finishedUIDs: [String] = []

        for response in rawResponses {
            guard var runningRequest = running[response.uid] else { continue }
            runningRequest.generatedTokenCount += 1

            let finishReason: FinishReason?
            if runningRequest.request.eosTokenIds.contains(response.token) {
                finishReason = .stop
            } else if runningRequest.generatedTokenCount >= runningRequest.request.maxTokens {
                finishReason = .length
            } else {
                finishReason = nil
            }

            running[response.uid] = runningRequest
            let processedResponse = Response(
                uid: response.uid,
                token: response.token,
                finishReason: finishReason,
                logprobs: response.logprobs
            )
            processed.append(processedResponse)
            collector.record(processedResponse)

            if finishReason != nil {
                finishedUIDs.append(response.uid)
            }
        }

        if !finishedUIDs.isEmpty {
            Stream.gpu.synchronize()
            for uid in finishedUIDs {
                generator.remove(uid: uid)
                running.removeValue(forKey: uid)
            }
        }

        try admitWaiting()
        return processed
    }

    public func collectedTokens() -> [String: [Int]] {
        collector.allTokens
    }

    public func responses(for uid: String) -> [Response] {
        collector.responses(for: uid)
    }

    private func admitWaiting() throws {
        while running.count < maxConcurrentRequests, !waiting.isEmpty {
            let request = waiting.removeFirst()
            try generator.insert(
                uid: request.uid,
                input: request.input,
                sampling: request.sampling
            )
            running[request.uid] = RunningRequest(request: request, generatedTokenCount: 0)
        }
    }

    private func applyPendingCancellation() {
        guard !pendingCancellation.isEmpty else { return }

        Stream.gpu.synchronize()
        for uid in pendingCancellation {
            guard running[uid] != nil else { continue }
            generator.remove(uid: uid)
            running.removeValue(forKey: uid)
            collector.record(Response(uid: uid, token: -1, finishReason: .cancelled))
        }
        pendingCancellation.removeAll()
    }
}
