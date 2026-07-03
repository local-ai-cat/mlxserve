import MLXLMCommon

public final class MLXServeEngine: @unchecked Sendable {
    private let scheduler: Scheduler

    public init(
        model: any LanguageModel,
        parameters: GenerateParameters = GenerateParameters(),
        maxConcurrentRequests: Int,
        prefixStore: (any PrefixKVStore)? = nil
    ) {
        self.scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: parameters,
            maxConcurrentRequests: maxConcurrentRequests,
            prefixStore: prefixStore
        )
    }

    public func submit(_ request: Request) async throws {
        try await scheduler.submit(request)
    }

    public func cancel(uid: String) async {
        await scheduler.cancel(uid: uid)
    }

    @discardableResult
    public func step() async throws -> [Response] {
        try await scheduler.step()
    }

    public func generate(_ requests: [Request]) async throws -> [String: [Int]] {
        for request in requests {
            try await scheduler.submit(request)
        }

        while await !scheduler.isIdle {
            _ = try await scheduler.step()
        }

        return await scheduler.collectedTokens()
    }

    public func stream(_ request: Request) -> AsyncThrowingStream<Response, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await scheduler.submit(request)
                    while await !scheduler.isIdle {
                        let responses = try await scheduler.step()
                        for response in responses where response.uid == request.uid {
                            continuation.yield(response)
                            if response.finishReason != nil {
                                continuation.finish()
                                return
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func responses(for uid: String) async -> [Response] {
        await scheduler.responses(for: uid)
    }

    public func tokens(for uid: String) async -> [Int] {
        await scheduler.collectedTokens()[uid, default: []]
    }

    public var isIdle: Bool {
        get async {
            await scheduler.isIdle
        }
    }
}
