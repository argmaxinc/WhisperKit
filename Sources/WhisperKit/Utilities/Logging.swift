//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import OSLog

open class Logging {
    public static let shared = Logging()
    public var logLevel: LogLevel = .none

    public typealias LoggingCallback = (_ message: String) -> Void
    public var loggingCallback: LoggingCallback?

    private let logger = OSLog(subsystem: Bundle.main.bundleIdentifier ?? "com.argmax.whisperkit", category: "WhisperKit")

    @frozen
    public enum LogLevel: Int {
        case debug = 1
        case info = 2
        case error = 3
        case none = 4

        func shouldLog(level: LogLevel) -> Bool {
            return self.rawValue <= level.rawValue
        }
    }

    private init() {}

    public func log(_ items: Any..., separator: String = " ", terminator: String = "\n", type: OSLogType) {
        let message = items.map { "\($0)" }.joined(separator: separator)
        if let logger = loggingCallback {
            logger(message)
        } else {
            os_log("%{public}@", log: logger, type: type, message)
        }
    }

    public static func debug(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        if shared.logLevel.shouldLog(level: .debug) {
            shared.log(items, separator: separator, terminator: terminator, type: .debug)
        }
    }

    public static func info(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        if shared.logLevel.shouldLog(level: .info) {
            shared.log(items, separator: separator, terminator: terminator, type: .info)
        }
    }

    public static func error(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        if shared.logLevel.shouldLog(level: .error) {
            shared.log(items, separator: separator, terminator: terminator, type: .error)
        }
    }
}

public extension Logging {
    static func logCurrentMemoryUsage(_ message: String) {
        let memoryUsage = getMemoryUsage()
        Logging.debug("\(message) - Memory usage: \(memoryUsage) MB")
    }

    static func getMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        guard kerr == KERN_SUCCESS else {
            return 0 // If the call fails, return 0
        }

        return info.resident_size / 1024 / 1024 // Convert to MB
    }
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `Logging.logCurrentMemoryUsage(_:)` instead.")
public func logCurrentMemoryUsage(_ message: String) {
    Logging.logCurrentMemoryUsage(message)
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `Logging.getMemoryUsage()` instead.")
public func getMemoryUsage() -> UInt64 {
    return Logging.getMemoryUsage()
}

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
        let percentage = (time * 1000 / fullPipelineDuration) * 100 // Convert to percentage
        let runTime = runs > 0 ? time * 1000 / Double(runs) : 0
        let formattedString = String(format: "%8.2f ms / %6.0f runs (%8.2f ms/run) %5.2f%%", time * 1000, runs, runTime, percentage)
        return formattedString
    }
}

