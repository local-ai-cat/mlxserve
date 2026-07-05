// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "mlxserve",
    platforms: [
        .macOS(.v14)
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
        .executableTarget(
            name: "MLXServeHTTPServer",
            dependencies: [
                "MLXServe",
                "MLXServeHTTP",
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
