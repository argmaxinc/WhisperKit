//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser
import Foundation

let VERSION: String = "development"

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
@main
struct WhisperKitCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "whisperkit-cli",
        abstract: "WhisperKit CLI",
        discussion: "Swift native speech recognition with Whisper for Apple Silicon",
        version: VERSION,
        subcommands: [TranscribeCLI.self]
    )
}
