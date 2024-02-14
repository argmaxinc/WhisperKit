//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import CoreML
import Foundation
import Tokenizers
#if canImport(UIKit)
    import UIKit
#elseif canImport(AppKit)
    import AppKit
#endif

// MARK: - Helpers

extension MLMultiArray {
    /// Calculate the linear offset by summing the products of each dimension’s index with the dimension’s stride.
    /// More info [here](https://developer.apple.com/documentation/coreml/mlmultiarray/2879231-subscript)
    /// - Parameters:
    ///  - index: The index of the element
    ///  - strides: The precomputed strides of the multi-array, if not provided, it will be computed. It's a performance optimization to avoid recomputing the strides every time when accessing the multi-array with multiple indexes.
    @inline(__always)
    func linearOffset(for index: [NSNumber], strides strideInts: [Int]? = nil) -> Int {
        var linearOffset = 0
        let strideInts = strideInts ?? strides.map { $0.intValue }
        for (dimension, stride) in zip(index, strideInts) {
            linearOffset += dimension.intValue * stride
        }
        return linearOffset
    }

    func fill<Value>(indexes: [[NSNumber]], with value: Value) {
        let pointer = UnsafeMutablePointer<Value>(OpaquePointer(dataPointer))
        let strideInts = strides.map { $0.intValue }
        for index in indexes {
            let linearOffset = linearOffset(for: index, strides: strideInts)
            pointer[linearOffset] = value
        }
    }
}

func initMLMultiArray(shape: [NSNumber], dataType: MLMultiArrayDataType, initialValue: Any) -> MLMultiArray {
    let multiArray = try! MLMultiArray(shape: shape, dataType: dataType)

    let count = multiArray.count
    let pointer = multiArray.dataPointer
    switch dataType {
    case .double:
        if let value = initialValue as? Double {
            let typedPointer = pointer.bindMemory(to: Double.self, capacity: count)
            typedPointer.initialize(repeating: value, count: count)
        }
    case .float32:
        if let value = initialValue as? Float {
            let typedPointer = pointer.bindMemory(to: Float.self, capacity: count)
            typedPointer.initialize(repeating: value, count: count)
        }
    case .float16:
        if let value = initialValue as? FloatType {
            let typedPointer = pointer.bindMemory(to: FloatType.self, capacity: count)
            typedPointer.initialize(repeating: value, count: count)
        }
    case .int32:
        if let value = initialValue as? Int32 {
            let typedPointer = pointer.bindMemory(to: Int32.self, capacity: count)
            typedPointer.initialize(repeating: value, count: count)
        }
    @unknown default:
        fatalError("Unsupported data type")
    }

    return multiArray
}

func getModelInputDimention(_ model: MLModel?, named: String, position: Int) -> Int? {
    guard let inputDescription = model?.modelDescription.inputDescriptionsByName[named] else { return nil }
    guard inputDescription.type == .multiArray else { return nil }
    guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
    let shape = shapeConstraint.shape.map { $0.intValue }
    return shape[position]
}

func getModelOutputDimention(_ model: MLModel?, named: String, position: Int) -> Int? {
    guard let inputDescription = model?.modelDescription.outputDescriptionsByName[named] else { return nil }
    guard inputDescription.type == .multiArray else { return nil }
    guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
    let shape = shapeConstraint.shape.map { $0.intValue }
    return shape[position]
}

func tokenizerNameForVariant(_ variant: ModelVariant) -> String {
    var tokenizerName: String
    switch variant {
    case .tiny:
        tokenizerName = "openai/whisper-tiny"
    case .tinyEn:
        tokenizerName = "openai/whisper-tiny.en"
    case .base:
        tokenizerName = "openai/whisper-base"
    case .baseEn:
        tokenizerName = "openai/whisper-base.en"
    case .small:
        tokenizerName = "openai/whisper-small"
    case .smallEn:
        tokenizerName = "openai/whisper-small.en"
    case .medium:
        tokenizerName = "openai/whisper-medium"
    case .mediumEn:
        tokenizerName = "openai/whisper-medium.en"
    case .large:
        tokenizerName = "openai/whisper-large"
    case .largev2:
        tokenizerName = "openai/whisper-large-v2"
    case .largev3:
        tokenizerName = "openai/whisper-large-v3"
    }

    return tokenizerName
}

func detectVariant(logitsDim: Int, encoderDim: Int) -> ModelVariant {
    // Defaults
    var modelVariant: ModelVariant = .base

    // Determine model size
    if logitsDim == 51865 {
        // Muiltilingual
        switch encoderDim {
        case 384:
            modelVariant = .tiny
        case 512:
            modelVariant = .base
        case 768:
            modelVariant = .small
        case 1024:
            modelVariant = .medium
        case 1280:
            modelVariant = .largev2 // same for v1
        default:
            modelVariant = .base
        }
    } else if logitsDim == 51864 {
        // English only
        switch encoderDim {
        case 384:
            modelVariant = .tinyEn
        case 512:
            modelVariant = .baseEn
        case 768:
            modelVariant = .smallEn
        case 1024:
            modelVariant = .mediumEn
        default:
            modelVariant = .baseEn
        }

    } else if logitsDim == 51866 {
        // Large v3 has 1 additional language token
        modelVariant = .largev3
    } else {
        Logging.error("Unrecognized vocabulary size: \(logitsDim), defaulting to base variant")
        modelVariant = .base
    }

    return modelVariant
}

public func modelSupport(for deviceName: String) -> (default: String, disabled: [String]) {
    switch deviceName {
    case let model where model.hasPrefix("iPhone11"), // A12
         let model where model.hasPrefix("iPhone12"), // A13
         let model where model.hasPrefix("Watch7"): // Series 9 and Ultra 2
        return ("base", ["small", "small.en", "large-v3_turbo", "large-v3", "large-v3_turbo_1307MB", "large-v3_turbo_1049MB", "large-v2", "large-v2_turbo", "large-v2_turbo_1116MB", "large-v2_turbo_1430MB", "large-v2_1161MB", "large-v2_1400MB", "large-v3_1053MB", "large-v2_1382MB"])

    case let model where model.hasPrefix("iPhone13"): // A14
        return ("base", ["large-v3_turbo", "large-v3", "large-v3_turbo_1307MB", "large-v3_turbo_1049MB", "large-v2", "large-v2_turbo", "large-v2_turbo_1116MB", "large-v2_turbo_1430MB"])

    case let model where model.hasPrefix("iPhone14"), // A15
         let model where model.hasPrefix("iPhone15"), // A16
         let model where model.hasPrefix("iPhone16"): // A17
        return ("base", ["large-v3_turbo", "large-v3", "large-v2_turbo", "large-v2"])

    // Fall through to macOS checks
    default:
        break
    }

    #if os(macOS)
        if deviceName.hasPrefix("arm64") {
            if Process.processor.contains("Apple M1") {
                // Disable turbo variants for M1
                return ("base", ["large-v3_turbo", "large-v3_turbo_1049MB", "large-v3_turbo_1307MB", "large-v2_turbo", "large-v2_turbo_1116MB", "large-v2_turbo_1430MB"])
            } else {
                // Enable all variants for M2 or M3, none disabled
                return ("base", [])
            }
        }
    #endif

    // Unhandled device, default to base variant
    return ("base", [""])
}

#if os(macOS)
    // From: https://stackoverflow.com/a/71726663
    extension Process {
        static func stringFromTerminal(command: String) -> String {
            let task = Process()
            let pipe = Pipe()
            task.standardOutput = pipe
            task.launchPath = "/bin/bash"
            task.arguments = ["-c", "sysctl -n " + command]
            task.launch()
            return String(bytes: pipe.fileHandleForReading.availableData, encoding: .utf8) ?? ""
        }

        static let processor = stringFromTerminal(command: "machdep.cpu.brand_string")
        static let cores = stringFromTerminal(command: "machdep.cpu.core_count")
        static let threads = stringFromTerminal(command: "machdep.cpu.thread_count")
        static let vendor = stringFromTerminal(command: "machdep.cpu.vendor")
        static let family = stringFromTerminal(command: "machdep.cpu.family")
    }
#endif

public func resolveAbsolutePath(_ inputPath: String) -> String {
    let fileManager = FileManager.default

    // Expanding tilde if present
    let pathWithTildeExpanded = NSString(string: inputPath).expandingTildeInPath

    // If the path is already absolute, return it
    if pathWithTildeExpanded.hasPrefix("/") {
        return pathWithTildeExpanded
    }

    // Resolving relative path based on the current working directory
    if let cwd = fileManager.currentDirectoryPath as String? {
        let resolvedPath = URL(fileURLWithPath: cwd).appendingPathComponent(pathWithTildeExpanded).path
        return resolvedPath
    }

    return inputPath
}

func loadTokenizer(for pretrained: ModelVariant) async throws -> Tokenizer {
    let tokenizerName = tokenizerNameForVariant(pretrained)
    return try await AutoTokenizer.from(pretrained: tokenizerName)
}

func formatTimestamp(_ timestamp: Float) -> String {
    return String(format: "%.2f", timestamp)
}

func formatTimeWithPercentage(_ time: Double, _ runs: Double, _ fullPipelineDuration: Double) -> String {
    let percentage = (time * 1000 / fullPipelineDuration) * 100 // Convert to percentage
    let runTime = runs > 0 ? time * 1000 / Double(runs) : 0
    let formattedString = String(format: "%8.2f ms / %6.0f runs (%8.2f ms/run) %5.2f%%", time * 1000, runs, runTime, percentage)
    return formattedString
}

public func formatSegments(_ segments: [TranscriptionSegment], withTimestamps: Bool = true) -> [String] {
    var lines = [String]()
    for segment in segments {
        let start = segment.start
        let end = segment.end
        let text = segment.text
        let timestamps = withTimestamps ? "[\(formatTimestamp(start)) --> \(formatTimestamp(end))] " : ""
        let line = "\(timestamps)\(text)"
        lines.append(line)
    }
    return lines
}

func timeit(operation: () -> Void) -> TimeInterval {
    let startTime = CFAbsoluteTimeGetCurrent()
    operation()
    let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
    return TimeInterval(timeElapsed)
}

func getDownloadsDirectory() -> URL {
    let paths = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)
    return paths[0]
}

func saveBuffer(_ buffer: AVAudioPCMBuffer, to url: URL) throws {
    // create folder
    let folderURL = url.deletingLastPathComponent()
    if !FileManager.default.fileExists(atPath: folderURL.path) {
        try FileManager.default.createDirectory(at: folderURL, withIntermediateDirectories: true, attributes: nil)
    }
    let audioFile = try AVAudioFile(forWriting: url, settings: buffer.format.settings)
    try audioFile.write(from: buffer)
}

func rescale(value: Float, min: Float, max: Float) -> Float {
    return (value - min) / (max - min)
}

public func compressionRatio(of array: [Int]) -> Float {
    // Convert the integer array to a byte array (Data)
    let dataBuffer = array.compactMap { Int32($0) }
    let data = dataBuffer.withUnsafeBufferPointer { Data(buffer: $0) }

    // Compress the data using NSData compression
    do {
        let compressedData = try (data as NSData).compressed(using: .zlib)
        // Calculate and return the compression ratio
        return Float(data.count) / Float(compressedData.length)
    } catch {
        Logging.debug("Compression error: \(error.localizedDescription)")
        return Float.infinity
    }
}

public func compressionRatio(of text: String) -> Float {
    if text.isEmpty {
        return Float.infinity // TODO: throw to caller instead of return infinity
    }

    // Encode the string as UTF-8
    guard let data = text.data(using: .utf8) else {
        Logging.debug("String encoding error")
        return Float.infinity
    }

    // Compress the data using NSData compression
    do {
        let compressedData = try (data as NSData).compressed(using: .zlib)
        // Calculate and return the compression ratio
        return Float(data.count) / Float(compressedData.length)
    } catch {
        Logging.debug("Compression error: \(error.localizedDescription)")
        return Float.infinity
    }
}

// MARK: - Singletons

public class Logging {
    static let shared = Logging()
    var logLevel: LogLevel = .none

    public typealias LoggingCallback = (_ message: String) -> Void
    var loggingCallback: LoggingCallback?

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

    public func log(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        let message = items.map { "\($0)" }.joined(separator: separator)
        if let logger = loggingCallback {
            logger(message)
        } else {
            print("[WhisperKit] \(message)", terminator: terminator)
        }
    }

    public static func debug(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        if shared.logLevel.shouldLog(level: .debug) {
            shared.log(items, separator: separator, terminator: terminator)
        }
    }

    public static func info(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        if shared.logLevel.shouldLog(level: .info) {
            shared.log(items, separator: separator, terminator: terminator)
        }
    }

    public static func error(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        if shared.logLevel.shouldLog(level: .error) {
            shared.log(items, separator: separator, terminator: terminator)
        }
    }
}
