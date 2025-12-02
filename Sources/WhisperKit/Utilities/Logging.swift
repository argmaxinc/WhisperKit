//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import OSLog

@frozen
public enum Logging {

    // MARK: - Helper Types

    public typealias LoggingCallback = @Sendable (_ message: String) -> Void

    @frozen
    public enum LogLevel: Int, Comparable {
        case debug = 1
        case info = 2
        case error = 3
        case none = 4

        var osLogType: OSLogType {
            switch self {
            case .debug: return .debug
            case .info: return .info
            case .error: return .error
            case .none: return .default
            }
        }

        public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }

    private actor Engine {
        var level: LogLevel
        var callback: LoggingCallback?
        private let logger: Logger

        init(level: LogLevel = .none, callback: LoggingCallback? = nil) {
            self.level = level
            self.callback = callback
            self.logger = Logger(
                subsystem: Constants.Logging.subsystem,
                category: "WhisperKit"
            )
        }

        func updateLogLevel(_ level: LogLevel) {
            self.level = level
        }

        func updateCallback(_ callback: LoggingCallback?) {
            self.callback = callback
        }

        func log(level: LogLevel, message: String) {
            guard self.level != .none, self.level <= level else { return }

            if let callback {
                callback(message)
            } else {
                logger.log(level: level.osLogType, "\(message, privacy: .public)")
            }
        }
    }

    // MARK: - Properties

    private static let engine = Engine()

    public static func isLoggingEnabled() async -> Bool {
        let level = await engine.level
        return level != .none
    }

    public static func updateLogLevel(_ level: LogLevel) async {
        await engine.updateLogLevel(level)
    }

    public static func updateCallback(_ callback: LoggingCallback?) async {
        await engine.updateCallback(callback)
    }

    // MARK: - Convenience

    public static func debug(_ message: String) {
        dispatch(level: .debug, message)
    }

    public static func info(_ message: String) {
        dispatch(level: .info, message)
    }
    
    public static func error(_ message: String) {
        dispatch(level: .error, message)
    }

    private static func dispatch(level: LogLevel, _ message: String) {
        Task(priority: .utility) {
            await engine.log(level: level, message: message)
        }
    }
}

// MARK: - Memory Usage

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

// MARK: - Feature Specific Logging

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
        String(format: "%.2f", timestamp)
    }

    static func formatTimeWithPercentage(_ time: Double, _ runs: Double, _ fullPipelineDuration: Double) -> String {
        let percentage = (time * 1000 / fullPipelineDuration) * 100
        let runTime = runs > 0 ? time * 1000 / Double(runs) : 0
        let formattedString = String(format: "%8.2f ms / %6.0f runs (%8.2f ms/run) %5.2f%%", time * 1000, runs, runTime, percentage)
        return formattedString
    }
}
