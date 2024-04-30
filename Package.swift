// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

#if os(watchOS)
let products: [PackageDescription.Product] = [
    .library(
        name: "WhisperKit",
        targets: ["WhisperKit"]
    ),
    .executable(
        name: "whisperkit-cli",
        targets: ["WhisperKitCLI"]
    ),
]

let targets: [PackageDescription.Target] = [
    .target(
        name: "WhisperKit",
        dependencies: [
            .product(name: "Transformers", package: "swift-transformers"),
        ],
        path: "Sources/WhisperKit/Core"
    ),
    .testTarget(
        name: "WhisperKitTests",
        dependencies: [
            "WhisperKit",
            .product(name: "Transformers", package: "swift-transformers"),
        ],
        path: ".",
        exclude: [
            "Examples",
            "Sources",
            "Makefile",
            "README.md",
            "LICENSE",
            "CONTRIBUTING.md",
            "Tests/WhisperKitMLXTests"
        ],
        resources: [
            .process("Tests/WhisperKitTests/Resources"),
            .copy("Models/whisperkit-coreml"),
        ]
    ),
]
#else
let products: [PackageDescription.Product] = [
    .library(
        name: "WhisperKit",
        targets: ["WhisperKit"]
    ),
    .library(
        name: "WhisperKitMLX",
        targets: ["WhisperKitMLX"]
    ),
    .executable(
        name: "whisperkit-cli",
        targets: ["WhisperKitCLI"]
    ),
]

let targets: [PackageDescription.Target] = [
    .target(
        name: "WhisperKit",
        dependencies: [
            .product(name: "Transformers", package: "swift-transformers"),
        ],
        path: "Sources/WhisperKit/Core"
    ),
    .executableTarget(
        name: "WhisperKitCLI",
        dependencies: [
            "WhisperKit",
            "WhisperKitMLX",
            .product(name: "ArgumentParser", package: "swift-argument-parser"),
        ]
    ),
    .testTarget(
        name: "WhisperKitTests",
        dependencies: [
            "WhisperKit",
            .product(name: "Transformers", package: "swift-transformers"),
        ],
        path: ".",
        exclude: [
            "Examples",
            "Sources",
            "Makefile",
            "README.md",
            "LICENSE",
            "CONTRIBUTING.md",
            "Tests/WhisperKitMLXTests"
        ],
        resources: [
            .process("Tests/WhisperKitTests/Resources"),
            .copy("Models/whisperkit-coreml"),
        ]
    ),
    .target(
        name: "WhisperKitMLX",
        dependencies: [
            "WhisperKit",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFFT", package: "mlx-swift")
        ],
        path: "Sources/WhisperKit/MLX",
        resources: [
            .copy("Resources/mel_filters_80.npy"),
            .copy("Resources/mel_filters_128.npy")
        ]
    ),
    .testTarget(
        name: "WhisperKitMLXTests",
        dependencies: [
            "WhisperKit",
            "WhisperKitMLX",
            .product(name: "Transformers", package: "swift-transformers"),
        ]
    )
]
#endif

let package = Package(
    name: "whisperkit",
    platforms: [
        .iOS(.v16),
        .macOS("13.3"),
    ],
    products: products,
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "0.1.7"),
        .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", exact: "1.3.0"),
    ],
    targets: targets
)
