// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "mlxserve",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "MLXServe",
            targets: ["MLXServe"]
        ),
        .library(
            name: "MLXServeHTTP",
            targets: ["MLXServeHTTP"]
        ),
        .library(
            name: "MLXServeSpeech",
            targets: ["MLXServeSpeech"]
        ),
        .library(
            name: "MLXServeSpeechWhisperKit",
            targets: ["MLXServeSpeechWhisperKit"]
        ),
        .executable(
            name: "mlxserve-bench",
            targets: ["MLXServeBench"]
        ),
        .executable(
            name: "mlxserve-http",
            targets: ["MLXServeHTTPServer"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/atlas-open-sources/mlx-swift-lm.git",
            revision: "098cf970a96c26dca1fb5b036abbf198c0b74ad4"
        ),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // Pinned to main by revision: tagged releases (≤0.17) pin swift-transformers
        // <1.2 which conflicts with mlx-swift-lm's ≥1.3; main dropped the dep.
        .package(
            url: "https://github.com/argmaxinc/WhisperKit",
            revision: "dcf3a00f0ae4d5b57bc0aad92063b102b70d5fd1"
        ),
    ],
    targets: [
        .target(
            name: "MLXServe",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "MLXServeBench",
            dependencies: [
                "MLXServe",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "MLXServeHTTP",
            dependencies: []
        ),
        .target(
            name: "MLXServeSpeech",
            dependencies: []
        ),
        .target(
            name: "MLXServeSpeechWhisperKit",
            dependencies: [
                "MLXServeSpeech",
                .product(name: "WhisperKit", package: "WhisperKit"),
            ]
        ),
        .executableTarget(
            name: "MLXServeHTTPServer",
            dependencies: [
                "MLXServe",
                "MLXServeHTTP",
                "MLXServeSpeech",
                "MLXServeSpeechWhisperKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "MLXServeTests",
            dependencies: [
                "MLXServe",
                "MLXServeHTTP",
                "MLXServeHTTPServer",
                "MLXServeSpeech",
                "MLXServeSpeechWhisperKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            resources: [
                .process("Fixtures")
            ]
        ),
    ]
)
