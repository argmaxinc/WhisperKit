// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import Foundation
import PackageDescription

let approachableConcurrencySettings: [SwiftSetting] = [
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
]

let macroPlugin = Target.macro(
    name: "ArgmaxCoreMacroPlugin",
    dependencies: [
        .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
        .product(name: "SwiftSyntax", package: "swift-syntax"),
        .product(name: "SwiftSyntaxMacros", package: "swift-syntax")
    ]
)

let macroTarget = Target.target(
    name: "ArgmaxCoreMacros",
    dependencies: [
        "ArgmaxCore",
        "ArgmaxCoreMacroPlugin"
    ],
    swiftSettings: approachableConcurrencySettings
)

let macroTestTarget = Target.testTarget(
    name: "ArgmaxCoreMacrosTests",
    dependencies: [
        "ArgmaxCoreMacros",
        "ArgmaxCoreMacroPlugin",
        .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
        .product(name: "SwiftSyntaxMacrosTestSupport", package: "swift-syntax")
    ],
    exclude: ["MacroTestsPlan.xctestplan"]
)

let package = Package(
    name: "whisperkit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .watchOS(.v10),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "WhisperKit",
            targets: ["WhisperKit"]
        ),
        .library(
            name: "TTSKit",
            targets: ["TTSKit"]
        ),
        .executable(
            name: "whisperkit-cli",
            targets: ["WhisperKitCLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMinor(from: "1.1.6")),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.7.0"),
        .package(url: "https://github.com/swiftlang/swift-syntax", from: "602.0.0"),
    ] + (isServerEnabled() ? [
        .package(url: "https://github.com/vapor/vapor.git", from: "4.115.1"),
        .package(url: "https://github.com/apple/swift-openapi-generator", from: "1.10.2"),
        .package(url: "https://github.com/apple/swift-openapi-runtime", from: "1.8.2"),
        .package(url: "https://github.com/swift-server/swift-openapi-vapor", from: "1.0.1"),
    ] : []),
    targets: [
        .target(
            name: "ArgmaxCore",
            swiftSettings: approachableConcurrencySettings
        ),
        macroPlugin,
        macroTarget,
        .target(
            name: "WhisperKit",
            dependencies: [
                "ArgmaxCore",
                "ArgmaxCoreMacros",
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: approachableConcurrencySettings
        ),
        .target(
            name: "TTSKit",
            dependencies: [
                "ArgmaxCore",
                "ArgmaxCoreMacros",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: approachableConcurrencySettings
        ),
        macroTestTarget,
        .testTarget(
            name: "WhisperKitTests",
            dependencies: [
                "WhisperKit",
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            exclude: ["UnitTestsPlan.xctestplan"],
            resources: [
                .process("Resources"),
            ],
            swiftSettings: approachableConcurrencySettings
        ),
        .testTarget(
            name: "TTSKitTests",
            dependencies: [
                "TTSKit"
            ],
            swiftSettings: approachableConcurrencySettings
        ),
        .executableTarget(
            name: "WhisperKitCLI",
            dependencies: [
                "WhisperKit",
                "TTSKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ] + (isServerEnabled() ? [
                .product(name: "Vapor", package: "vapor"),
                .product(name: "OpenAPIRuntime", package: "swift-openapi-runtime"),
                .product(name: "OpenAPIVapor", package: "swift-openapi-vapor"),
            ] : []),
            exclude: (isServerEnabled() ? [] : ["Server"]),
            swiftSettings: approachableConcurrencySettings + (isServerEnabled() ? [.define("BUILD_SERVER_CLI")] : [])
        )
    ],
    swiftLanguageModes: [.v5]
)

func isServerEnabled() -> Bool {
    if let enabledValue = Context.environment["BUILD_ALL"] {
        return enabledValue.lowercased() == "true" || enabledValue == "1"
    }

    // Default disabled, change to true temporarily for local development
    return false
}
