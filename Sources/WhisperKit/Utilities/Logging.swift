//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import OSLog

@frozen
public enum Logging {

    // MARK: - Helper Types

    public typealias LoggingCallback = @Sendable (_ message: String) -> Void

    /// Represents the severity threshold for emitting log messages.
    ///
    /// The `LogLevel` controls which messages are allowed to be logged. Messages are
    /// emitted when their severity is greater than or equal to the globally configured
    /// log level. For example, if the global level is set to `.info`, then `.info` and
    /// `.error` messages will be logged, while `.debug` messages will be suppressed.
    ///
    /// Ordering (from least to most severe):
    /// - `.debug` (1): Verbose diagnostic information useful during development.
    /// - `.info`  (2): High-level informational messages about app flow or state.
    /// - `.error` (3): Errors and failures that require attention.
    /// - `.none`  (4): Disables all logging.
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

    private actor LoggingActor {
        var level: LogLevel
        var callback: LoggingCallback?
        private let logger: Logger

        var isLoggingEnabled: Bool {
            level != .none
        }

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
            guard isLoggingEnabled(for: level) else { return }

            if let callback {
                callback(message)
            } else {
                logger.log(level: level.osLogType, "\(message, privacy: .public)")
            }
        }

        func isLoggingEnabled(for level: LogLevel) -> Bool {
            self.level != .none && self.level <= level
        }
    }

    // MARK: - Properties

    private static let loggingActor = LoggingActor()

    /// The current global logging level.
    ///
    /// Accessing this property is asynchronous because it reads state managed by an internal
    /// actor to ensure thread-safe access across concurrency domains.
    ///
    /// - Returns: The currently configured `LogLevel`.
    public static var logLevel: LogLevel {
        get async {
            await loggingActor.level
        }
    }

    /// A Boolean value indicating whether logging is currently enabled.
    ///
    /// This property reflects the global logging state managed by the internal logging actor.
    /// When `true`, log messages at or above the configured `LogLevel` may be emitted either
    /// to the system logger or to a custom callback if one has been provided. When `false`
    /// (i.e., when the log level is `.none`), all logging is suppressed.
    ///
    /// Accessing this property is asynchronous because it queries state stored within an actor,
    /// ensuring thread-safe reads across concurrency domains.
    ///
    /// - Returns: `true` if logging is enabled for any level other than `.none`; otherwise, `false`.
    /// - Note: The effective behavior also depends on the current `LogLevel` (see `updateLogLevel(_:)`)
    ///   and any registered `LoggingCallback` (see `updateCallback(_:)`).
    public static var isLoggingEnabled: Bool {
        get async { await loggingActor.isLoggingEnabled }
    }

    /// Updates the global logging level used by the logging system.
    ///
    /// This method communicates with an internal actor to safely mutate the shared
    /// logging state across concurrency domains. The effective logging behavior is
    /// determined by the value you provide:
    /// - `.debug`: Enables all logs (debug, info, error).
    /// - `.info`: Enables info and error logs; suppresses debug logs.
    /// - `.error`: Enables only error logs.
    /// - `.none`: Disables all logging.
    ///
    /// - Important: Because this operation crosses an actor boundary, it is asynchronous
    ///   and must be awaited from async contexts.
    ///
    /// - Parameter level: The new `LogLevel` to apply globally.
    /// - SeeAlso: `updateCallback(_:)`, `isLoggingEnabled`, `isLoggingEnabled(for:)`
    public static func updateLogLevel(_ level: LogLevel) async {
        await loggingActor.updateLogLevel(level)
    }

    /// Updates the global logging level without requiring an async context (fire-and-forget).
    ///
    /// This convenience overload schedules the level update on the internal logging actor
    /// without blocking the caller. Use this when you need to change the log level from
    /// synchronous code paths. If you need to guarantee the update has completed before
    /// proceeding, prefer the async variant `updateLogLevel(_:)`.
    ///
    /// - Parameter level: The new `LogLevel` to apply globally.
    public static func updateLogLevel(_ level: LogLevel) {
        Task(priority: .high) {
            await loggingActor.updateLogLevel(level)
        }
    }

    /// Updates the global logging callback used to handle emitted log messages.
    ///
    /// By default, messages are sent to the system logger (OSLog). Providing a custom
    /// callback allows you to intercept and route log output elsewhere (e.g., to a UI,
    /// file, or analytics pipeline). Pass `nil` to remove any previously set callback
    /// and revert to system logging.
    ///
    /// This method is asynchronous because it safely updates shared logging state
    /// managed by an internal actor.
    ///
    /// - Parameter callback: A closure that receives each log message as a `String`,
    ///   or `nil` to clear the custom callback.
    /// - Important: The callback is invoked on an actor-isolated context; ensure the
    ///   closure is `@Sendable` and avoid long-running or blocking work inside it.
    /// - SeeAlso: `updateCallback(_:)` (non-async), `updateLogLevel(_:)`, `isLoggingEnabled`
    public static func updateCallback(_ callback: LoggingCallback?) async {
        await loggingActor.updateCallback(callback)
    }

    /// Updates the logging callback without requiring an async context (fire-and-forget).
    ///
    /// This convenience overload schedules the callback update on the internal logging actor
    /// without blocking the caller. Use this from synchronous code paths. If you need to
    /// guarantee the update has completed before proceeding, prefer the async variant
    /// `updateCallback(_:)`.
    ///
    /// - Parameter callback: The new `LoggingCallback` to apply globally, or `nil` to remove it.
    public static func updateCallback(_ callback: LoggingCallback?) {
        Task(priority: .high) {
            await loggingActor.updateCallback(callback)
        }
    }

    // MARK: - Convenience

    /// Logs a debug message asynchronously (fire-and-forget).
    /// - Warning: This method does not await logging completion.
    /// Log messages may be dropped if the program terminates before logging completes.
    public static func debug(_ message: String) {
        dispatch(level: .debug, message)
    }

    /// Logs an info message asynchronously (fire-and-forget).
    /// - Warning: This method does not await logging completion.
    /// Log messages may be dropped if the program terminates before logging completes.
    public static func info(_ message: String) {
        dispatch(level: .info, message)
    }

    /// Logs an error message asynchronously (fire-and-forget).
    /// - Warning: This method does not await logging completion.
    /// Log messages may be dropped if the program terminates before logging completes.
    public static func error(_ message: String) {
        dispatch(level: .error, message)
    }

    /// Returns a Boolean value that indicates whether logging is currently enabled for a specific log level.
    /// 
    /// This method safely queries the internal logging actor to determine if messages at the given
    /// `level` would be emitted under the current configuration. Logging is considered enabled for
    /// the provided level when:
    /// - The global log level is not `.none`, and
    /// - The global log level is less than or equal to the provided `level` (i.e., the message’s
    ///   severity meets or exceeds the configured threshold).
    ///
    /// Because this check crosses an actor boundary, the API is asynchronous and must be awaited.
    ///
    /// - Parameter level: The `LogLevel` to evaluate (e.g., `.debug`, `.info`, `.error`).
    /// - Returns: `true` if messages at the specified level would be logged; otherwise, `false`.
    /// - SeeAlso: `updateLogLevel(_:)`, `isLoggingEnabled`, `debug(_:)`, `info(_:)`, `error(_:)`
    public static func isLoggingEnabled(for level: LogLevel) async -> Bool {
        await loggingActor.isLoggingEnabled(for: level)
    }

    private static func dispatch(level: LogLevel, _ message: String) {
        Task(priority: .high) {
            await loggingActor.log(level: level, message: message)
        }
    }
}

// MARK: - Convenience Extension

public extension Logging {
    static func debug(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        let message = items.map { "\($0)" }.joined(separator: separator) + terminator
        debug(message)
    }

    static func info(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        let message = items.map { "\($0)" }.joined(separator: separator) + terminator
        info(message)
    }

    static func error(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        let message = items.map { "\($0)" }.joined(separator: separator) + terminator
        error(message)
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
