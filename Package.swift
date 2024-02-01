// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "whisperkit",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(
            name: "WhisperKit",
            targets: ["WhisperKit"]
        ),
        .executable(
            name: "transcribe",
            targets: ["WhisperKitCLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", "0.1.1"..<"0.2.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "WhisperKit",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "WhisperKitCLI",
            dependencies: [
                "WhisperKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
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
                "CONTRIBUTING.md"
            ],
            resources: [
                .process("Tests/WhisperKitTests/Resources"),
                .copy("Models/whisperkit-coreml")
            ]
        ),
    ]
)
