public final class OutputCollector: @unchecked Sendable {
    private var responsesByUID: [String: [Response]] = [:]
    private var finishReasonsByUID: [String: FinishReason] = [:]

    public init() {}

    public func record(_ response: Response) {
        responsesByUID[response.uid, default: []].append(response)
        if let finishReason = response.finishReason {
            finishReasonsByUID[response.uid] = finishReason
        }
    }

    public func record(_ responses: [Response]) {
        for response in responses {
            record(response)
        }
    }

    public func tokens(for uid: String) -> [Int] {
        responsesByUID[uid, default: []]
            .filter { $0.finishReason != .cancelled }
            .map(\.token)
    }

    public func responses(for uid: String) -> [Response] {
        responsesByUID[uid, default: []]
    }

    public func finishReason(for uid: String) -> FinishReason? {
        finishReasonsByUID[uid]
    }

    public var allTokens: [String: [Int]] {
        responsesByUID.mapValues { responses in
            responses
                .filter { $0.finishReason != .cancelled }
                .map(\.token)
        }
    }
}
