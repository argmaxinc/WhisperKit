//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import CoreML

public extension Array where Element == TranscriptionSegment {
    func contains(segment: TranscriptionSegment) -> Bool {
        return self.contains { $0.start == segment.start }
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public extension WhisperKit {
    static var isRunningOnSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }
}

public extension Float {
    func rounded(_ decimalPlaces: Int) -> Float {
        let divisor = pow(10.0, Float(decimalPlaces))
        return (self * divisor).rounded() / divisor
    }
}

public extension String {
    var normalized: String {
        // Convert to lowercase
        let lowercaseString = self.lowercased()

        // Replace dashes with spaces
        let noDashesString = lowercaseString.replacingOccurrences(of: "-", with: " ")

        // Remove punctuation
        let noPunctuationString = noDashesString.components(separatedBy: .punctuationCharacters).joined()

        // Replace multiple spaces with a single space
        let singleSpacedString = noPunctuationString.replacingOccurrences(of: " +", with: " ", options: .regularExpression)

        // Trim whitespace and newlines
        let trimmedString = singleSpacedString.trimmingCharacters(in: .whitespacesAndNewlines)

        return trimmedString
    }

    func trimmingSpecialTokenCharacters() -> String {
        trimmingCharacters(in: Constants.specialTokenCharacters)
    }
}

// MARK: CoreML

public extension MLMultiArray {
    @available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
    convenience init(shape: [NSNumber], dataType: MLMultiArrayDataType, initialValue: Any) throws {
        switch dataType {
        case .float16:
            // IOSurface-backed arrays are implicitly float16. They can
            // reduce buffer copies for some OS:compute unit combinations.
            guard let pixelBuffer = Self.pixelBuffer(for: shape) else {
                throw WhisperError.initializationError("MLMultiArray: Failed to initialize PixelBuffer")
            }
            self.init(pixelBuffer: pixelBuffer, shape: shape)
        default:
            try self.init(shape: shape, dataType: dataType)
        }

        switch dataType {
            case .double:
                if let value = initialValue as? Double {
                    let typedPointer = dataPointer.bindMemory(to: Double.self, capacity: count)
                    typedPointer.initialize(repeating: value, count: count)
                }
            case .float32:
                if let value = initialValue as? Float {
                    let typedPointer = dataPointer.bindMemory(to: Float.self, capacity: count)
                    typedPointer.initialize(repeating: value, count: count)
                }
            case .float16:
                if let value = initialValue as? FloatType {
                    let typedPointer = dataPointer.bindMemory(to: FloatType.self, capacity: count)
                    typedPointer.initialize(repeating: value, count: count)
                }
            case .int32:
                if let value = initialValue as? Int32 {
                    let typedPointer = dataPointer.bindMemory(to: Int32.self, capacity: count)
                    typedPointer.initialize(repeating: value, count: count)
                }
            @unknown default:
                fatalError("Unsupported data type")
        }
    }

    /// Calculate the linear offset by summing the products of each dimension's index with the dimension's stride.
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

    func fillLastDimension(indexes: Range<Int>, with value: FloatType) {
        precondition(shape.count == 3 && shape[0] == 1 && shape[1] == 1, "Must have [1, 1, n] shape")
        withUnsafeMutableBufferPointer(ofType: FloatType.self) { ptr, strides in
            for index in indexes {
                ptr[index * strides[2]] = value
            }
        }
    }

    func fill<Value>(indexes: [[NSNumber]], with value: Value) {
        let pointer = UnsafeMutablePointer<Value>(OpaquePointer(dataPointer))
        let strideInts = strides.map { $0.intValue }
        for index in indexes {
            let linearOffset = linearOffset(for: index, strides: strideInts)
            pointer[linearOffset] = value
        }
    }

    private class func pixelBuffer(for shape: [NSNumber]) -> CVPixelBuffer? {
        guard let width = shape.last?.intValue else { return nil }
        let height = shape[0..<shape.count - 1].reduce(1) { $0 * $1.intValue }

        var pixelBuffer: CVPixelBuffer?
        let createReturn = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_OneComponent16Half,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary,
            &pixelBuffer
        )
        guard createReturn == kCVReturnSuccess else { return nil }
        return pixelBuffer
    }
}

#if canImport(CoreML.MLState)
@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
public extension MLTensor {
    func asIntArray() -> [Int] {
        let semaphore = DispatchSemaphore(value: 0)
        var result: [Int] = []

        Task(priority: .high) {
            result = await self.shapedArray(of: Int32.self).scalars.map { Int($0) }
            semaphore.signal()
        }

        semaphore.wait()
        return result
    }

    func asFloatArray() -> [Float] {
        let semaphore = DispatchSemaphore(value: 0)
        let tensorType = self.scalarType

        var result: [Float] = []

        Task(priority: .high) {
            switch tensorType {
                case is Float32.Type:
                    result = await self.shapedArray(of: Float32.self).scalars.map { Float($0) }
                case is FloatType.Type:
                    result = await self.shapedArray(of: FloatType.self).scalars.map { Float($0) }
                case is Float.Type:
                    result = await self.shapedArray(of: Float.self).scalars.map { Float($0) }
                case is Int32.Type:
                    result = await self.shapedArray(of: Int32.self).scalars.map { Float($0) }
                default:
                    fatalError("Unsupported data type")
            }
            semaphore.signal()
        }

        semaphore.wait()
        return result
    }

    func asMLMultiArray() -> MLMultiArray {
        let semaphore = DispatchSemaphore(value: 0)
        let tensorType = self.scalarType

        var result = try! MLMultiArray(shape: [1], dataType: .float16, initialValue: 0.0)

        Task(priority: .high) {
            switch tensorType {
                case is Float32.Type:
                    result = MLMultiArray(await self.shapedArray(of: Float32.self))
                case is FloatType.Type:
                    result = MLMultiArray(await self.shapedArray(of: FloatType.self))
                case is Float.Type:
                    result = MLMultiArray(await self.shapedArray(of: Float.self))
                case is Int32.Type:
                    result = MLMultiArray(await self.shapedArray(of: Int32.self))
                default:
                    fatalError("Unsupported data type")
            }
            semaphore.signal()
        }

        semaphore.wait()
        return result
    }
}
#endif

public extension MLModel {
    func asyncPrediction(
        from input: MLFeatureProvider,
        options: MLPredictionOptions
    ) async throws -> MLFeatureProvider {
        if #available(macOS 14, iOS 17, watchOS 10, visionOS 1, *) {
            return try await prediction(from: input, options: options)
        } else {
            return try await Task {
                try prediction(from: input, options: options)
            }.value
        }
    }
}

public extension MLComputeUnits {
    var description: String {
        switch self {
            case .cpuOnly:
                return "cpuOnly"
            case .cpuAndGPU:
                return "cpuAndGPU"
            case .all:
                return "all"
            case .cpuAndNeuralEngine:
                return "cpuAndNeuralEngine"
            @unknown default:
                return "unknown"
        }
    }
}

#if os(macOS) || targetEnvironment(simulator)
// From: https://stackoverflow.com/a/71726663
public extension ProcessInfo {
    static func stringFromSysctl(named name: String) -> String {
        var size: size_t = 0
        sysctlbyname(name, nil, &size, nil, 0)
        var machineModel = [CChar](repeating: 0, count: Int(size))
        sysctlbyname(name, &machineModel, &size, nil, 0)
        return String(cString: machineModel)
    }

    static let processor = stringFromSysctl(named: "machdep.cpu.brand_string")
    static let cores = stringFromSysctl(named: "machdep.cpu.core_count")
    static let threads = stringFromSysctl(named: "machdep.cpu.thread_count")
    static let vendor = stringFromSysctl(named: "machdep.cpu.vendor")
    static let family = stringFromSysctl(named: "machdep.cpu.family")
    static let hwModel = stringFromSysctl(named: "hw.model")
}
#endif

// MARK: FileManager

public extension FileManager {
    static func resolveAbsolutePath(_ inputPath: String) -> String {
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
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `FileManager.resolveAbsolutePath(_:)` instead.")
public func resolveAbsolutePath(_ inputPath: String) -> String {
    return FileManager.resolveAbsolutePath(inputPath)
}
