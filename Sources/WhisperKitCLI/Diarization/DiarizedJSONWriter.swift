//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

struct DiarizedJSONWriter {
    let outputDir: String

    func write(payload: DiarizedTranscriptionPayload, to file: String) -> Result<String, Error> {
        let reportPathURL = URL(fileURLWithPath: outputDir)
        let reportURL = reportPathURL.appendingPathComponent("\(file).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        do {
            let data = try encoder.encode(payload)
            try data.write(to: reportURL, options: .atomic)
            return .success(reportURL.path)
        } catch {
            return .failure(error)
        }
    }
}
