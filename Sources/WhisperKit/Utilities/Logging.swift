//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgmaxCore
import OSLog

// MARK: - WhisperKit signpost categories

extension Logging {
    enum AudioEncoding {
        static let logger = Logger(
            subsystem: Constants.Logging.subsystem,
            category: "AudioEncoding"
        )
        static let signposter = OSSignposter(logger: logger)
    }

    enum FeatureExtractor {
        static let logger = Logger(
            subsystem: Constants.Logging.subsystem,
            category: "FeatureExtractor"
        )
        static let signposter = OSSignposter(logger: logger)
    }

    enum TranscribeTask {
        static let logger = Logger(
            subsystem: Constants.Logging.subsystem,
            category: "TranscribeTask"
        )
        static let signposter = OSSignposter(logger: logger)
    }

    static func beginSignpost(
        _ intervalName: StaticString,
        signposter: OSSignposter
    ) -> OSSignpostIntervalState {
        let signpostId = signposter.makeSignpostID()
        return signposter.beginInterval(intervalName, id: signpostId)
    }

    static func endSignpost(
        _ intervalName: StaticString,
        interval: OSSignpostIntervalState,
        signposter: OSSignposter
    ) {
        signposter.endInterval(intervalName, interval)
    }

    static func formatTimestamp(_ timestamp: Float) -> String {
        return String(format: "%.2f", timestamp)
    }

    static func formatTimeWithPercentage(_ time: Double, _ runs: Double, _ fullPipelineDuration: Double) -> String {
        let percentage = (time * 1000 / fullPipelineDuration) * 100
        let runTime = runs > 0 ? time * 1000 / Double(runs) : 0
        let formattedString = String(format: "%8.2f ms / %6.0f runs (%8.2f ms/run) %5.2f%%", time * 1000, runs, runTime, percentage)
        return formattedString
    }
}
