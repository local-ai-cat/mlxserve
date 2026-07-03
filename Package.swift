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
        )
    ],
    dependencies: [
        .package(path: "/Users/timapple/Documents/Guest/mlx-swift-lm"),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MLXServe",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ]
        ),
        .testTarget(
            name: "MLXServeTests",
            dependencies: [
                "MLXServe",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            resources: [
                .process("Fixtures")
            ]
        ),
    ]
)
