import MLX

public struct SamplingParameters: Sendable, Equatable {
    public var temperature: Float
    public var logprobCount: Int?

    public init(temperature: Float = 0, logprobCount: Int? = nil) {
        self.temperature = temperature
        self.logprobCount = logprobCount
    }
}

public enum TokenSampler {
    public static func sample(logprobs: MLXArray, parameters: SamplingParameters) -> MLXArray {
        if parameters.temperature == 0 {
            return argMax(logprobs, axis: -1).reshaped([1])
        }

        let scaled = logprobs / MLXArray(parameters.temperature)
        return MLXRandom.categorical(scaled, axis: -1).asType(.int32).reshaped([1])
    }
}
