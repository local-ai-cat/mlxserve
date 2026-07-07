public struct ModelCacheCapabilities: Sendable, Equatable {
    public let usesWindowedKVCache: Bool

    public init(usesWindowedKVCache: Bool = false) {
        self.usesWindowedKVCache = usesWindowedKVCache
    }

    public static let `default` = ModelCacheCapabilities()
}
