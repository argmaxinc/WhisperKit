//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser

struct VersionCommand: ParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "version",
        abstract: "Display the current version of WhisperKitCLI"
    )

    func run() throws {
        print("WhisperKitCLI version \(Version.current)")
    }
}
