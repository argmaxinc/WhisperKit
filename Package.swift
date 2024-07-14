// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import Foundation

// NOTE: `MLX` doesn't support `watchOS` yet, that's why we control the build using the `MLX_DISABLED` environment variable.
// To manualy build for `watchOS` use:
// `export MLX_DISABLED=1 && xcodebuild clean build-for-testing -scheme whisperkit -sdk watchos10.4 -destination 'platform=watchOS Simulator' -skipPackagePluginValidation`
let package = Package(
    name: "whisperkit",
    platforms: [
        .iOS(.v16),
        .macOS("13.3")
    ],
    products: products() + mlxProducts(),
    dependencies: dependencies() + mlxDependencies(),
    targets: targets() + mlxTargets()
)

func products() -> [PackageDescription.Product] {
    return [
        .library(
            name: "WhisperKit",
            targets: ["WhisperKit"]
        )
    ]
}

func mlxProducts() -> [PackageDescription.Product] {
    let isMLXDisabled = ProcessInfo.processInfo.environment["MLX_DISABLED"] == "1"
    if isMLXDisabled {
        return []
    } else {
        return [
            .library(
                name: "WhisperKitMLX",
                targets: ["WhisperKitMLX"]
            ),
            .executable(
                name: "whisperkit-cli",
                targets: ["WhisperKitCLI"]
            ),
        ]
    }
}

func dependencies() -> [PackageDescription.Package.Dependency] {
    return [
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "0.1.7"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", exact: "1.3.0"),
    ]
}

func mlxDependencies() -> [PackageDescription.Package.Dependency] {
    let isMLXDisabled = ProcessInfo.processInfo.environment["MLX_DISABLED"] == "1"
    if isMLXDisabled {
        return []
    } else {
        return [
            .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
        ]
    }
}

func targets() -> [PackageDescription.Target] {
    return [
        .target(
            name: "WhisperKit",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/WhisperKit/Core"
        ),
        .target(
            name: "WhisperKitTestsUtils",
            dependencies: [
                "WhisperKit",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: ".",
            exclude: [
                "Examples",
                "Sources/WhisperKit",
                "Sources/WhisperKitCLI",
                "Tests",
                "Makefile",
                "README.md",
                "LICENSE",
                "CONTRIBUTING.md",
            ],
            resources: [
                .copy("Models/whisperkit-coreml"),
                .copy("Models/whisperkit-mlx"),
                .process("Sources/WhisperKitTestsUtils/Resources")
            ]
        ),
        .testTarget(
            name: "WhisperKitTests",
            dependencies: [
                "WhisperKit",
                "WhisperKitTestsUtils",
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        )
    ]
}

func mlxTargets() -> [PackageDescription.Target] {
    let isMLXDisabled = ProcessInfo.processInfo.environment["MLX_DISABLED"] == "1"
    if isMLXDisabled {
        return []
    } else {
        return [
            .executableTarget(
                name: "WhisperKitCLI",
                dependencies: [
                    "WhisperKit",
                    "WhisperKitMLX",
                    .product(name: "ArgumentParser", package: "swift-argument-parser"),
                ]
            ),
            .target(
                name: "WhisperKitMLX",
                dependencies: [
                    "WhisperKit",
                    .product(name: "MLX", package: "mlx-swift"),
                    .product(name: "MLXFFT", package: "mlx-swift"),
                    .product(name: "MLXNN", package: "mlx-swift")
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
                    "WhisperKitTestsUtils",
                    .product(name: "Transformers", package: "swift-transformers"),
                ]
            )
        ]
    }
}
