//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser
import Foundation

let VERSION: String = "v0.15.0"

var subcommands: [ParsableCommand.Type] {
#if BUILD_SERVER_CLI
    [TranscribeCLI.self, ServeCLI.self]
#else
    [TranscribeCLI.self]
#endif
}

@main
struct WhisperKitCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "whisperkit-cli",
        abstract: "WhisperKit CLI",
        discussion: "Swift native speech recognition with Whisper for Apple Silicon",
        version: VERSION,
        subcommands: subcommands
    )
}
