//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import CoreML
import Foundation
import Hub
import os.signpost
import Tokenizers
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif
#if canImport(Darwin)
import Darwin
#endif

// MARK: - Extensions

extension Array {
    func batched(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}

extension Array where Element == Result<[TranscriptionResult], Swift.Error> {
    /// Convenience method to convert the `Result` object into an array of optional arrays of `TranscriptionResult`.
    /// - Returns: An array of optional arrays containing `TranscriptionResult`.
    func toOptionalArrays() -> [[TranscriptionResult]?] {
        return self.map { try? $0.get() }
    }
}

public extension Array where Element == TranscriptionSegment {
    func contains(segment: TranscriptionSegment) -> Bool {
        return self.contains { $0.start == segment.start }
    }
}

extension Array where Element: Hashable {
    /// Returns an array with duplicates removed, preserving the original order.
    var orderedSet: [Element] {
        var seen = Set<Element>()
        return self.filter { element in
            if seen.contains(element) {
                return false
            } else {
                seen.insert(element)
                return true
            }
        }
    }
}

extension MLMultiArray {
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

    class func uninitializedIOSurfaceArray(shape: [NSNumber]) -> MLMultiArray? {
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
        guard let pixelBuffer = pixelBuffer else { return nil }

        return MLMultiArray(pixelBuffer: pixelBuffer, shape: shape)
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

        var result: MLMultiArray = initMLMultiArray(shape: [1], dataType: .float16, initialValue: 0.0)

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

extension AVAudioPCMBuffer {
    /// Converts the buffer to a float array
    func asFloatArray() throws -> [Float] {
        guard let data = floatChannelData?.pointee else {
            throw WhisperError.audioProcessingFailed("Error converting audio, missing floatChannelData")
        }
        return Array(UnsafeBufferPointer(start: data, count: Int(frameLength)))
    }

    /// Appends the contents of another buffer to the current buffer
    func appendContents(of buffer: AVAudioPCMBuffer) -> Bool {
        return appendContents(of: buffer, startingFrame: 0, frameCount: buffer.frameLength)
    }

    /// Appends a specific range of frames from another buffer to the current buffer
    func appendContents(of buffer: AVAudioPCMBuffer, startingFrame: AVAudioFramePosition, frameCount: AVAudioFrameCount) -> Bool {
        guard format == buffer.format else {
            Logging.debug("Format mismatch")
            return false
        }

        guard startingFrame + AVAudioFramePosition(frameCount) <= AVAudioFramePosition(buffer.frameLength) else {
            Logging.error("Insufficient audio in buffer")
            return false
        }

        guard let destination = floatChannelData, let source = buffer.floatChannelData else {
            Logging.error("Failed to access float channel data")
            return false
        }

        var calculatedFrameCount = frameCount
        if frameLength + frameCount > frameCapacity {
            Logging.debug("Insufficient space in buffer, reducing frame count to fit")
            calculatedFrameCount = frameCapacity - frameLength
        }

        let calculatedStride = stride
        let destinationPointer = destination.pointee.advanced(by: calculatedStride * Int(frameLength))
        let sourcePointer = source.pointee.advanced(by: calculatedStride * Int(startingFrame))

        memcpy(destinationPointer, sourcePointer, Int(calculatedFrameCount) * calculatedStride * MemoryLayout<Float>.size)

        frameLength += calculatedFrameCount
        return true
    }

    /// Convenience initializer to concatenate multiple buffers into one
    convenience init?(concatenating buffers: [AVAudioPCMBuffer]) {
        guard !buffers.isEmpty else {
            Logging.debug("Buffers array should not be empty")
            return nil
        }

        let totalFrames = buffers.reduce(0) { $0 + $1.frameLength }

        guard let firstBuffer = buffers.first else {
            Logging.debug("Failed to get the first buffer")
            return nil
        }

        self.init(pcmFormat: firstBuffer.format, frameCapacity: totalFrames)

        for buffer in buffers {
            if !appendContents(of: buffer) {
                Logging.debug("Failed to append buffer")
                return nil
            }
        }
    }

    /// Computed property to determine the stride for float channel data
    private var stride: Int {
        return Int(format.streamDescription.pointee.mBytesPerFrame) / MemoryLayout<Float>.size
    }
}

// MARK: - Helpers

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
func prepareSeekClips(contentFrames: Int, decodeOptions: DecodingOptions?) -> [(start: Int, end: Int)] {
    let options = decodeOptions ?? DecodingOptions()
    var seekPoints: [Int] = options.clipTimestamps.map { Int(round($0 * Float(WhisperKit.sampleRate))) }
    if seekPoints.count == 0 {
        seekPoints.append(0)
    }

    if seekPoints.count % 2 == 1 {
        seekPoints.append(contentFrames)
    }

    var seekClips: [(start: Int, end: Int)] = []
    for i in stride(from: 0, to: seekPoints.count, by: 2) {
        let start = seekPoints[i]
        let end = i + 1 < seekPoints.count ? seekPoints[i + 1] : contentFrames
        seekClips.append((start, end))
    }

    return seekClips
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
func initMLMultiArray(shape: [NSNumber], dataType: MLMultiArrayDataType, initialValue: Any) -> MLMultiArray {
    var multiArray: MLMultiArray
    switch dataType {
        case .float16:
            // IOSurface-backed arrays are implicitly float16. They can
            // reduce buffer copies for some OS:compute unit combinations.
            multiArray = MLMultiArray.uninitializedIOSurfaceArray(shape: shape)!
        default:
            multiArray = try! MLMultiArray(shape: shape, dataType: dataType)
    }

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

func isModelMultilingual(logitsDim: Int?) -> Bool {
    logitsDim != 51864
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

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public func modelSupport(for deviceName: String, from config: ModelSupportConfig? = nil) -> ModelSupport {
    let config = config ?? Constants.fallbackModelSupportConfig
    let modelSupport = config.modelSupport(for: deviceName)
    return modelSupport
}

/// Deprecated
@available(*, deprecated, message: "Subject to removal in a future version. Use modelSupport(for:from:) -> ModelSupport instead.")
@_disfavoredOverload
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public func modelSupport(for deviceName: String, from config: ModelSupportConfig? = nil) -> (default: String, disabled: [String]) {
    let modelSupport: ModelSupport = modelSupport(for: deviceName, from: config)
    return (modelSupport.default, modelSupport.disabled)
}

public func detectModelURL(inFolder path: URL, named modelName: String) -> URL {
    let compiledUrl = path.appending(path: "\(modelName).mlmodelc")
    let packageUrl = path.appending(path: "\(modelName).mlpackage/Data/com.apple.CoreML/model.mlmodel")

    let compiledModelExists: Bool = FileManager.default.fileExists(atPath: compiledUrl.path)
    let packageModelExists: Bool = FileManager.default.fileExists(atPath: packageUrl.path)

    // Swap to mlpackage only if the following is true: we found the mlmodel within the mlpackage, and we did not find a .mlmodelc
    var modelURL = compiledUrl
    if packageModelExists && !compiledModelExists {
        modelURL = packageUrl
    }

    return modelURL
}

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

public func loadTokenizer(
    for pretrained: ModelVariant,
    tokenizerFolder: URL? = nil,
    useBackgroundSession: Bool = false
) async throws -> WhisperTokenizer {
    let tokenizerName = tokenizerNameForVariant(pretrained)
    let hubApi = HubApi(downloadBase: tokenizerFolder, useBackgroundSession: useBackgroundSession)

    // Attempt to load tokenizer from local folder if specified
    let resolvedTokenizerFolder = hubApi.localRepoLocation(HubApi.Repo(id: tokenizerName))
    let tokenizerConfigPath = resolvedTokenizerFolder.appendingPathComponent("tokenizer.json")

    // Check if 'tokenizer.json' exists in the folder
    if FileManager.default.fileExists(atPath: tokenizerConfigPath.path) {
        do {
            let localConfig = LanguageModelConfigurationFromHub(modelFolder: resolvedTokenizerFolder, hubApi: hubApi)
            if let tokenizerConfig = try await localConfig.tokenizerConfig {
                let tokenizerData = try await localConfig.tokenizerData
                let whisperTokenizer = try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
                Logging.debug("Loading tokenizer from local folder")
                return WhisperTokenizerWrapper(tokenizer: whisperTokenizer)
            } else {
                // tokenizerConfig is nil, fall through to load from Hub
                Logging.debug("Tokenizer configuration not found in local config")
            }
        } catch {
            // Error during the local loading process and fall through to load from Hub
            Logging.debug("Error loading local tokenizer: \(error)")
        }
    }

    // Fallback to loading from the Hub if local loading is not possible or fails
    Logging.debug("Loading tokenizer from Hub")
    return try await WhisperTokenizerWrapper(
        tokenizer: AutoTokenizer.from(
            pretrained: tokenizerName,
            hubApi: hubApi
        )
    )
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

public func findLongestCommonPrefix(_ words1: [WordTiming], _ words2: [WordTiming]) -> [WordTiming] {
    let commonPrefix = zip(words1, words2).prefix(while: { $0.word.normalized == $1.word.normalized })
    return commonPrefix.map { $0.1 }
}

public func findLongestDifferentSuffix(_ words1: [WordTiming], _ words2: [WordTiming]) -> [WordTiming] {
    let commonPrefix = findLongestCommonPrefix(words1, words2)
    let remainingWords = words2[commonPrefix.count...]
    return Array(remainingWords)
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public func mergeTranscriptionResults(_ results: [TranscriptionResult?], confirmedWords: [WordTiming]? = nil) -> TranscriptionResult {
    var mergedText = ""
    if let words = confirmedWords {
        mergedText = words.map { $0.word }.joined()
    } else {
        mergedText = results.map { $0?.text ?? "" }.joined(separator: " ")
    }

    // Merge segments
    let validResults = results.compactMap { $0 }
    var mergedSegments = [TranscriptionSegment]()
    var previousSeek: Float = 0.0
    for (resultIndex, result) in validResults.enumerated() {
        let seekTime = result.seekTime ?? previousSeek
        for (segmentIndex, segment) in result.segments.enumerated() {
            var updatedSegment = segment
            updatedSegment.id = resultIndex + segmentIndex
            mergedSegments.append(updatedSegment)
        }
        // Update previousSeek only if seekTime is nil
        if result.seekTime == nil {
            previousSeek += Float(result.timings.inputAudioSeconds)
        } else {
            previousSeek = seekTime + Float(result.timings.inputAudioSeconds)
        }
    }

    let language = validResults.first?.language ?? Constants.defaultLanguageCode

    // Calculate the earliest start and latest end times
    let earliestPipelineStart = validResults.map { $0.timings.pipelineStart }.min() ?? 0
    let earliestTokenTime = validResults.map { $0.timings.firstTokenTime }.min() ?? 0
    let latestPipelineEnd = validResults.map { $0.timings.pipelineStart + $0.timings.fullPipeline }.max() ?? 0

    // Calculate the "user" pipeline time, excluding the time spent in concurrent pipelines
    let userPipelineDuration = latestPipelineEnd - earliestPipelineStart
    let systemPipelineDuration = validResults.map { $0.timings.fullPipeline }.reduce(0, +)
    let fullPipelineDuration = min(userPipelineDuration, systemPipelineDuration)

    // Update the merged timings with non-overlapping time values
    var mergedTimings = TranscriptionTimings(
        modelLoading: validResults.map { $0.timings.modelLoading }.max() ?? 0,
        prewarmLoadTime: validResults.map { $0.timings.prewarmLoadTime }.max() ?? 0,
        encoderLoadTime: validResults.map { $0.timings.encoderLoadTime }.max() ?? 0,
        decoderLoadTime: validResults.map { $0.timings.decoderLoadTime }.max() ?? 0,
        tokenizerLoadTime: validResults.map { $0.timings.tokenizerLoadTime }.max() ?? 0,
        audioLoading: validResults.map { $0.timings.audioLoading }.reduce(0, +),
        audioProcessing: validResults.map { $0.timings.audioProcessing }.reduce(0, +),
        logmels: validResults.map { $0.timings.logmels }.reduce(0, +),
        encoding: validResults.map { $0.timings.encoding }.reduce(0, +),
        prefill: validResults.map { $0.timings.prefill }.reduce(0, +),
        decodingInit: validResults.map { $0.timings.decodingInit }.reduce(0, +),
        decodingLoop: validResults.map { $0.timings.decodingLoop }.reduce(0, +),
        decodingPredictions: validResults.map { $0.timings.decodingPredictions }.reduce(0, +),
        decodingFiltering: validResults.map { $0.timings.decodingFiltering }.reduce(0, +),
        decodingSampling: validResults.map { $0.timings.decodingSampling }.reduce(0, +),
        decodingFallback: validResults.map { $0.timings.decodingFallback }.reduce(0, +),
        decodingWindowing: validResults.map { $0.timings.decodingWindowing }.reduce(0, +),
        decodingKvCaching: validResults.map { $0.timings.decodingKvCaching }.reduce(0, +),
        decodingTimestampAlignment: validResults.map { $0.timings.decodingWordTimestamps }.reduce(0, +),
        decodingNonPrediction: validResults.map { $0.timings.decodingNonPrediction }.reduce(0, +),
        totalAudioProcessingRuns: validResults.map { $0.timings.totalAudioProcessingRuns }.reduce(0, +),
        totalLogmelRuns: validResults.map { $0.timings.totalLogmelRuns }.reduce(0, +),
        totalEncodingRuns: validResults.map { $0.timings.totalEncodingRuns }.reduce(0, +),
        totalDecodingLoops: validResults.map { $0.timings.totalDecodingLoops }.reduce(0, +),
        totalKVUpdateRuns: validResults.map { $0.timings.totalKVUpdateRuns }.reduce(0, +),
        totalTimestampAlignmentRuns: validResults.map { $0.timings.totalTimestampAlignmentRuns }.reduce(0, +),
        totalDecodingFallbacks: validResults.map { $0.timings.totalDecodingFallbacks }.reduce(0, +),
        totalDecodingWindows: validResults.map { $0.timings.totalDecodingWindows }.reduce(0, +),
        fullPipeline: fullPipelineDuration
    )

    mergedTimings.pipelineStart = earliestPipelineStart
    mergedTimings.firstTokenTime = earliestTokenTime
    mergedTimings.inputAudioSeconds = validResults.map { $0.timings.inputAudioSeconds }.reduce(0, +)

    return TranscriptionResult(
        text: mergedText,
        segments: mergedSegments,
        language: language,
        timings: mergedTimings
    )
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public func updateSegmentTimings(segment: TranscriptionSegment, seekTime: Float) -> TranscriptionSegment {
    var updatedSegment = segment
    let seekOffsetIndex = Int(seekTime * Float(WhisperKit.sampleRate))
    updatedSegment.seek += seekOffsetIndex
    updatedSegment.start += seekTime
    updatedSegment.end += seekTime
    if var words = updatedSegment.words {
        for wordIndex in 0..<words.count {
            words[wordIndex].start += seekTime
            words[wordIndex].end += seekTime
        }
        updatedSegment.words = words
    }
    return updatedSegment
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

public func logCurrentMemoryUsage(_ message: String) {
    let memoryUsage = getMemoryUsage()
    Logging.debug("\(message) - Memory usage: \(memoryUsage) MB")
}

public func getMemoryUsage() -> UInt64 {
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

// MARK: - Singletons

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

extension Logging {
    enum AudioEncoding {
        static let logger = Logger(
            subsystem: Constants.Logging.subsystem,
            category: "AudioEncoding"
        )
        static let signposter = OSSignposter(logger: logger)
    }
}

extension Logging {
    enum FeatureExtractor {
        static let logger = Logger(
            subsystem: Constants.Logging.subsystem,
            category: "FeatureExtractor"
        )
        static let signposter = OSSignposter(logger: logger)
    }
}

extension Logging {
    enum TranscribeTask {
        static let logger = Logger(
            subsystem: Constants.Logging.subsystem,
            category: "TranscribeTask"
        )
        static let signposter = OSSignposter(logger: logger)
    }
}

extension Logging {
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
}
