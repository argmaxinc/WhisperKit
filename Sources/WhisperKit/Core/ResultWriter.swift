//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

public protocol ResultWriting {
    var outputDir: String { get }
    func write(result: TranscriptionResult, to file: String, options: [String: Any]?) -> Result<String, Error>
    func formatTime(seconds: Float, alwaysIncludeHours: Bool, decimalMarker: String) -> String
}

public extension ResultWriting {
    /// Format a time value as a string
    func formatTime(seconds: Float, alwaysIncludeHours: Bool, decimalMarker: String) -> String {
        let hrs = Int(seconds / 3600)
        let mins = Int((seconds.truncatingRemainder(dividingBy: 3600)) / 60)
        let secs = Int(seconds.truncatingRemainder(dividingBy: 60))
        let msec = Int((seconds - floor(seconds)) * 1000)

        if alwaysIncludeHours || hrs > 0 {
            return String(format: "%02d:%02d:%02d\(decimalMarker)%03d", hrs, mins, secs, msec)
        } else {
            return String(format: "%02d:%02d\(decimalMarker)%03d", mins, secs, msec)
        }
    }
}

public class WriteJSON: ResultWriting {
    public let outputDir: String

    public init(outputDir: String) {
        self.outputDir = outputDir
    }

    /// Write a transcription result to a JSON file
    /// - Parameters:
    ///   - result: Completed transcription result
    ///   - file: Name of the file to write, without the extension
    ///   - options: Not used
    /// - Returns: The URL of the written file, or a error if the write failed
    public func write(result: TranscriptionResult, to file: String, options: [String: Any]? = nil) -> Result<String, Error> {
        let reportPathURL = URL(fileURLWithPath: outputDir)
        let reportURL = reportPathURL.appendingPathComponent("\(file).json")
        let jsonEncoder = JSONEncoder()
        jsonEncoder.outputFormatting = .prettyPrinted
        do {
            let reportJson = try jsonEncoder.encode(result)
            try reportJson.write(to: reportURL)
        } catch {
            return .failure(error)
        }

        return .success(reportURL.absoluteString)
    }
}

public class WriteSRT: ResultWriting {
    public let outputDir: String

    public init(outputDir: String) {
        self.outputDir = outputDir
    }

    public func write(result: TranscriptionResult, to file: String, options: [String: Any]? = nil) -> Result<String, Error> {
        let outputPathURL = URL(fileURLWithPath: outputDir)
        let outputFileURL = outputPathURL.appendingPathComponent("\(file).srt")

        do {
            var srtContent = ""
            for (index, segment) in result.segments.enumerated() {
                let start = formatTime(seconds: segment.start, alwaysIncludeHours: true, decimalMarker: ",")
                let end = formatTime(seconds: segment.end, alwaysIncludeHours: true, decimalMarker: ",")
                srtContent += "\(index + 1)\n\(start) --> \(end)\n\(segment.text)\n\n"
            }

            try srtContent.write(to: outputFileURL, atomically: true, encoding: .utf8)
        } catch {
            return .failure(error)
        }

        return .success(outputFileURL.absoluteString)
    }
}

public class WriteVTT: ResultWriting {
    public let outputDir: String

    public init(outputDir: String) {
        self.outputDir = outputDir
    }

    public func write(result: TranscriptionResult, to file: String, options: [String: Any]? = nil) -> Result<String, Error> {
        let outputPathURL = URL(fileURLWithPath: outputDir)
        let outputFileURL = outputPathURL.appendingPathComponent("\(file).vtt")

        do {
            var vttContent = "WEBVTT\n\n"
            for segment in result.segments {
                let start = formatTime(seconds: segment.start, alwaysIncludeHours: false, decimalMarker: ".")
                let end = formatTime(seconds: segment.end, alwaysIncludeHours: false, decimalMarker: ".")
                vttContent += "\(start) --> \(end)\n\(segment.text)\n\n"
            }

            try vttContent.write(to: outputFileURL, atomically: true, encoding: .utf8)
        } catch {
            return .failure(error)
        }

        return .success(outputFileURL.absoluteString)
    }
}
