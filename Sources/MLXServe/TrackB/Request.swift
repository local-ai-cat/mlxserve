import MLXLMCommon

public struct Request: @unchecked Sendable {
    public let uid: String
    public let input: LMInput
    public let maxTokens: Int
    public let sampling: SamplingParameters
    public let eosTokenIds: Set<Int>
    public let cacheSession: String?

    public init(
        uid: String,
        input: LMInput,
        maxTokens: Int,
        sampling: SamplingParameters = SamplingParameters(),
        eosTokenIds: Set<Int> = [],
        cacheSession: String? = nil
    ) {
        self.uid = uid
        self.input = input
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.eosTokenIds = eosTokenIds
        self.cacheSession = cacheSession
    }
}

struct RunningRequest {
    let request: Request
    let promptTokens: [Int]
    let prefixHit: PrefixKVStoreHit?
    var generatedTokenCount: Int
    var promptCacheSnapshot: [SerializedKVLayer]?
}

public enum SchedulerError: Error, Equatable {
    case queueFull(retryAfterSteps: Int)
    case duplicateRequest(String)
    case invalidPrefixHit
}
