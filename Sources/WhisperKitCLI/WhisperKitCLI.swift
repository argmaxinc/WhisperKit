//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser
import Foundation

@main
struct WhisperKitCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "whisper-kit",
        abstract: "WhisperKit CLI",
        discussion: "Swift native speech recognition with Whisper for Apple Silicon",
        subcommands: [Transcribe.self]
    )
}
