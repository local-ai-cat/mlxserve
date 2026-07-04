import Foundation
import MLXServe
import MLXServeHTTP

final class PoolBackedChatBackend<Loader: EnginePoolModelLoader>: OpenAIModelLifecycleBackend, @unchecked Sendable
where Loader.Engine == NativeModelEngine {
    let models: [OpenAIModelInfo]

    private let pool: EnginePool<Loader>

    init(pool: EnginePool<Loader>, modelIDs: [String]) {
        self.pool = pool
        self.models = modelIDs.sorted().map { OpenAIModelInfo(id: $0, maxModelLength: nil) }
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        let ticket: EnginePoolQueueTicket
        do {
            ticket = try await pool.admitWaitingRequest()
        } catch {
            throw openAIError(from: error)
        }

        do {
            let lease = try await pool.acquire(request.model)
            let stream: OpenAIChatStream
            do {
                stream = try await lease.engine.startChatCompletion(request)
            } catch {
                await pool.release(lease)
                await pool.finishWaitingRequest(ticket)
                throw error
            }
            return leasedStream(stream, lease: lease, ticket: ticket)
        } catch {
            await pool.finishWaitingRequest(ticket)
            throw openAIError(from: error)
        }
    }

    func modelPoolStatus() async throws -> OpenAIModelPoolStatus {
        OpenAIModelPoolStatus(await pool.status())
    }

    func loadModel(_ id: String) async throws -> OpenAIModelLifecycleResult {
        do {
            let result = try await pool.load(id)
            let message = result.alreadyLoaded ? "Already loaded: \(id)" : "Loaded: \(id)"
            return OpenAIModelLifecycleResult(modelID: id, message: message)
        } catch {
            throw openAIError(from: error)
        }
    }

    func unloadModel(_ id: String) async throws -> OpenAIModelLifecycleResult {
        do {
            try await pool.unload(id)
            return OpenAIModelLifecycleResult(modelID: id)
        } catch {
            throw openAIError(from: error)
        }
    }

    private func leasedStream(
        _ stream: OpenAIChatStream,
        lease: EnginePoolLease<NativeModelEngine>,
        ticket: EnginePoolQueueTicket
    ) -> OpenAIChatStream {
        let releaseState = PoolLeaseReleaseState(pool: pool, lease: lease, ticket: ticket)
        let chunks = AsyncThrowingStream<OpenAIChatChunk, Error> { continuation in
            let task = Task {
                defer {
                    Task {
                        await releaseState.release()
                    }
                }
                do {
                    for try await chunk in stream.chunks {
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
                Task {
                    await releaseState.release()
                }
            }
        }
        return OpenAIChatStream(promptTokens: stream.promptTokens, chunks: chunks)
    }

    private func openAIError(from error: Error) -> OpenAIHTTPError {
        guard let poolError = error as? EnginePoolError else {
            if let httpError = error as? OpenAIHTTPError {
                return httpError
            }
            return OpenAIHTTPError(status: 500, message: String(describing: error))
        }
        return OpenAIHTTPError(
            status: poolError.httpStatus,
            message: poolError.message,
            retryAfterSeconds: poolError.retryAfterSeconds
        )
    }
}

private actor PoolLeaseReleaseState<Loader: EnginePoolModelLoader>
where Loader.Engine == NativeModelEngine {
    private let pool: EnginePool<Loader>
    private let lease: EnginePoolLease<NativeModelEngine>
    private let ticket: EnginePoolQueueTicket
    private var released = false

    init(
        pool: EnginePool<Loader>,
        lease: EnginePoolLease<NativeModelEngine>,
        ticket: EnginePoolQueueTicket
    ) {
        self.pool = pool
        self.lease = lease
        self.ticket = ticket
    }

    func release() async {
        guard !released else { return }
        released = true
        await pool.release(lease)
        await pool.finishWaitingRequest(ticket)
    }
}

private extension OpenAIModelPoolStatus {
    init(_ status: EnginePoolStatus) {
        self.init(
            finalCeiling: status.finalCeiling,
            currentModelMemory: status.currentModelMemory,
            modelCount: status.modelCount,
            loadedCount: status.loadedCount,
            models: status.models.map(OpenAIModelRuntimeStatus.init)
        )
    }
}

private extension OpenAIModelRuntimeStatus {
    init(_ status: EnginePoolModelStatus) {
        self.init(
            id: status.id,
            modelPath: status.modelPath,
            loaded: status.loaded,
            isLoading: status.isLoading,
            estimatedSize: status.estimatedSize,
            actualSize: status.actualSize,
            pinned: status.pinned,
            lastAccess: status.lastAccess,
            inUse: status.inUse
        )
    }
}
