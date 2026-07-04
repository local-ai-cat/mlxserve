import Foundation
import MLX
import MLXServe
import MLXServeHTTP

struct MLXServeHTTPServerMain {
    static func main() async throws {
        let config = try ServerConfig.parse(CommandLine.arguments)
        let modelRootURL = URL(fileURLWithPath: config.modelPath, isDirectory: true)
        let discovered = try applyModelIDOverrideIfNeeded(
            try ModelDiscovery.discoverModels(in: modelRootURL),
            override: config.modelID
        )
        guard !discovered.isEmpty else {
            throw ServerConfigError.invalidArgument("no models discovered in \(config.modelPath)")
        }

        let finalCeiling = finalMemoryCeiling(for: config.memoryGuardTier)
        let pool = EnginePool(
            models: discovered,
            loader: NativeModelLoader(maxConcurrentRequests: config.maxConcurrentRequests),
            finalCeiling: finalCeiling,
            idleTimeout: config.idleTimeout
        )
        for pinnedModelID in config.pinnedModelIDs {
            try await pool.setPinned(true, for: pinnedModelID)
        }

        let backend = PoolBackedChatBackend(
            pool: pool,
            modelIDs: Array(discovered.keys)
        )
        let server = try OpenAIServer(
            host: config.host,
            port: config.port,
            backend: backend
        )
        if config.idleTimeout != nil {
            startIdleSweep(pool: pool, interval: config.idleTimeout ?? 0)
        }

        try await server.start()
        print("MLXServeHTTP listening on http://\(config.host):\(config.port)")
        print("Discovered \(discovered.count) model(s); memory ceiling: \(ModelDiscovery.formatSize(finalCeiling))")
        fflush(stdout)
        server.waitForever()
    }
}

try await MLXServeHTTPServerMain.main()

private struct ServerConfig {
    let host: String
    let port: UInt16
    let modelPath: String
    let modelID: String?
    let maxConcurrentRequests: Int
    let memoryGuardTier: MemoryGuardTier?
    let idleTimeout: TimeInterval?
    let pinnedModelIDs: [String]

    static func parse(_ arguments: [String]) throws -> ServerConfig {
        var host = "127.0.0.1"
        var port: UInt16 = 18181
        var modelPath = ProcessInfo.processInfo.environment["MLXSERVE_MODEL_DIR"]
            ?? ProcessInfo.processInfo.environment["MLXSERVE_TEST_MODEL"]
        var modelID: String?
        var maxConcurrentRequests = 8
        var memoryGuardTier: MemoryGuardTier?
        var idleTimeout: TimeInterval?
        var pinnedModelIDs: [String] = []

        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--host":
                index += 1
                host = try value(arguments, at: index, for: argument)
            case "--port":
                index += 1
                guard let parsed = UInt16(try value(arguments, at: index, for: argument)) else {
                    throw ServerConfigError.invalidArgument("invalid port")
                }
                port = parsed
            case "--model-dir":
                index += 1
                modelPath = try value(arguments, at: index, for: argument)
            case "--model-id":
                index += 1
                modelID = try value(arguments, at: index, for: argument)
            case "--max-concurrent-requests":
                index += 1
                guard let parsed = Int(try value(arguments, at: index, for: argument)) else {
                    throw ServerConfigError.invalidArgument("invalid max concurrent requests")
                }
                maxConcurrentRequests = parsed
            case "--memory-guard-tier":
                index += 1
                let rawValue = try value(arguments, at: index, for: argument)
                guard let tier = MemoryGuardTier(rawValue: rawValue) else {
                    throw ServerConfigError.invalidArgument("invalid memory guard tier: \(rawValue)")
                }
                memoryGuardTier = tier
            case "--idle-timeout":
                index += 1
                guard let parsed = TimeInterval(try value(arguments, at: index, for: argument)), parsed > 0 else {
                    throw ServerConfigError.invalidArgument("invalid idle timeout")
                }
                idleTimeout = parsed
            case "--pin":
                index += 1
                pinnedModelIDs.append(try value(arguments, at: index, for: argument))
            case "--help", "-h":
                printHelp()
                Foundation.exit(0)
            default:
                throw ServerConfigError.invalidArgument(argument)
            }
            index += 1
        }

        guard let modelPath else {
            throw ServerConfigError.invalidArgument("missing --model-dir or MLXSERVE_MODEL_DIR")
        }

        return ServerConfig(
            host: host,
            port: port,
            modelPath: modelPath,
            modelID: modelID,
            maxConcurrentRequests: maxConcurrentRequests,
            memoryGuardTier: memoryGuardTier,
            idleTimeout: idleTimeout,
            pinnedModelIDs: pinnedModelIDs
        )
    }

    private static func value(_ arguments: [String], at index: Int, for option: String) throws -> String {
        guard arguments.indices.contains(index) else {
            throw ServerConfigError.invalidArgument("missing value for \(option)")
        }
        return arguments[index]
    }

    private static func printHelp() {
        print(
            """
            Usage: swift run mlxserve-http [options]

            Options:
              --host HOST                       Bind host. Default: 127.0.0.1
              --port PORT                       Bind port. Default: 18181
              --model-dir PATH                  MLX model directory. Required unless MLXSERVE_MODEL_DIR is set.
              --model-id ID                     Single-model compatibility override.
              --max-concurrent-requests N       Scheduler concurrency cap. Default: 8
              --memory-guard-tier TIER          safe, balanced, or aggressive. Default: off.
              --idle-timeout SECONDS            Unload idle, unpinned models after this many seconds. Default: off.
              --pin ID                          Pin a discovered model in memory. Repeatable.
            """
        )
    }
}

private func applyModelIDOverrideIfNeeded(
    _ discovered: [String: DiscoveredModel],
    override: String?
) throws -> [String: DiscoveredModel] {
    guard let override else {
        return discovered
    }
    guard discovered.count == 1, let model = discovered.values.first else {
        throw ServerConfigError.invalidArgument("--model-id can only be used with a single discovered model")
    }
    return [
        override: DiscoveredModel(
            id: override,
            modelURL: model.modelURL,
            estimatedSize: model.estimatedSize
        )
    ]
}

private func finalMemoryCeiling(for tier: MemoryGuardTier?) -> Int64 {
    guard let tier else {
        return 0
    }
    let recommendedWorkingSet = Int64(GPU.maxRecommendedWorkingSetBytes() ?? 0)
    return MemoryGuard.finalCeiling(
        recommendedWorkingSetBytes: recommendedWorkingSet,
        tier: tier
    )
}

private func startIdleSweep<Loader: EnginePoolModelLoader>(
    pool: EnginePool<Loader>,
    interval: TimeInterval
) {
    let sleepNanoseconds = UInt64(max(interval, 1) * 1_000_000_000)
    Task.detached {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: sleepNanoseconds)
            let unloaded = await pool.sweepIdleModels()
            if !unloaded.isEmpty {
                print("Idle timeout unloaded model(s): \(unloaded.joined(separator: ", "))")
                fflush(stdout)
            }
        }
    }
}

private enum ServerConfigError: Error, CustomStringConvertible {
    case invalidArgument(String)

    var description: String {
        switch self {
        case .invalidArgument(let message):
            return message
        }
    }
}
