//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import OSLog

/// Shared logger for all Argmax frameworks (WhisperKit, TTSKit, etc.).
///
/// Configure the log level once at startup:
/// ```swift
/// Logging.shared.logLevel = .debug
/// ```
/// or via a config object, following the WhisperKit pattern:
/// ```swift
/// Logging.shared.logLevel = config.verbose ? config.logLevel : .none
/// ```
open class Logging {
    public static let shared = Logging()
    public var logLevel: LogLevel = .none

    public typealias LoggingCallback = (_ message: String) -> Void
    public var loggingCallback: LoggingCallback?

    private let logger = OSLog(
        subsystem: Bundle.main.bundleIdentifier ?? "com.argmax.argmaxcore",
        category: "Argmax"
    )

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
            return 0
        }

        return info.resident_size / 1024 / 1024
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
