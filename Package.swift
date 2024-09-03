// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import Foundation

let package = Package(
    name: "whisperkit",
    platforms: [
        .iOS(.v16),
        .macOS("13.3"),
        .watchOS(.v10)
    ],
    products: [
        .library(
            name: "WhisperKit",
            targets: ["WhisperKit"]
        ),
    ] 
    + cliProducts()
    + mlxProducts(),
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "0.1.7"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", exact: "1.3.0"),
    ] + mlxDependencies(),
    targets: [
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
    + cliTargets()
    + mlxTargets()
)

// MARK: - MLX Helper Functions

// CLI
func cliProducts() -> [Product] {
    guard !isMLXDisabled() else { return [] }
    return [
        .executable(
            name: "whisperkit-cli",
            targets: ["WhisperKitCLI"]
        ),
    ]
}

func cliTargets() -> [Target] {
    guard !isMLXDisabled() else { return [] }
    return [
        .executableTarget(
            name: "WhisperKitCLI",
            dependencies: [
                "WhisperKit",
                "WhisperKitMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
}

// MLX 
func mlxProducts() -> [Product] {
    guard !isMLXDisabled() else { return [] }
    return [
        .library(
            name: "WhisperKitMLX",
            targets: ["WhisperKitMLX"]
        ),
    ]
}

func mlxDependencies() -> [Package.Dependency] {
    guard !isMLXDisabled() else { return [] }
    return [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.16.0"),
    ]
}

func mlxTargets() -> [Target] {
    guard !isMLXDisabled() else { return [] }
    return [
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

// NOTE: `MLX` doesn't support `watchOS` yet, that's why we control the build using the `MLX_DISABLED` environment variable.
// To manualy build for `watchOS` use:
// `export MLX_DISABLED=1 && xcodebuild clean build-for-testing -scheme whisperkit -sdk watchos10.4 -destination 'platform=watchOS Simulator,OS=10.5,name=Apple Watch Ultra 2 (49mm)' -skipPackagePluginValidation`

func isMLXDisabled() -> Bool {
    ProcessInfo.processInfo.environment["MLX_DISABLED"] == "1"
}
