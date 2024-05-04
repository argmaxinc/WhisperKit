//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import CoreML
import Hub
import NaturalLanguage
import Tokenizers

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
public typealias FloatType = Float16
#else
public typealias FloatType = Float
#endif

#if (os(macOS) || targetEnvironment(macCatalyst)) && arch(arm64)
extension Float16: BNNSScalar {}
extension Float16: MLShapedArrayScalar {}
#endif

// MARK: - CoreML

public protocol WhisperMLModel {
    var model: MLModel? { get set }
    mutating func loadModel(at modelPath: URL, computeUnits: MLComputeUnits, prewarmMode: Bool) async throws
    mutating func unloadModel()
}

public extension WhisperMLModel {
    mutating func loadModel(at modelPath: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let loadedModel = try await Task {
            let modelConfig = MLModelConfiguration()
            modelConfig.computeUnits = computeUnits
            return try await MLModel.load(contentsOf: modelPath, configuration: modelConfig)
        }.value

        model = prewarmMode ? nil : loadedModel
    }

    mutating func unloadModel() {
        model = nil
    }
}

// MARK: - Whisper Models

public enum ModelVariant: CustomStringConvertible, CaseIterable {
    case tiny
    case tinyEn
    case base
    case baseEn
    case small
    case smallEn
    case medium
    case mediumEn
    case large
    case largev2
    case largev3

    // TODO: implement config.json and generation_config.json parsing for models
    public var isMultilingual: Bool {
        switch self {
            case .tiny, .base, .small, .medium, .large, .largev2, .largev3:
                return true
            case .tinyEn, .baseEn, .smallEn, .mediumEn:
                return false
        }
    }

    public var description: String {
        switch self {
            case .tiny:
                return "tiny"
            case .tinyEn:
                return "tiny.en"
            case .base:
                return "base"
            case .baseEn:
                return "base.en"
            case .small:
                return "small"
            case .smallEn:
                return "small.en"
            case .medium:
                return "medium"
            case .mediumEn:
                return "medium.en"
            case .large:
                return "large"
            case .largev2:
                return "large-v2"
            case .largev3:
                return "large-v3"
        }
    }
}

public enum ModelState: CustomStringConvertible {
    case unloading
    case unloaded
    case loading
    case loaded
    case prewarming
    case prewarmed
    case downloading
    case downloaded

    public var description: String {
        switch self {
            case .unloading:
                return "Unloading"
            case .unloaded:
                return "Unloaded"
            case .loading:
                return "Loading"
            case .loaded:
                return "Loaded"
            case .prewarming:
                return "Specializing"
            case .prewarmed:
                return "Specialized"
            case .downloading:
                return "Downloading"
            case .downloaded:
                return "Downloading"
        }
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public struct ModelComputeOptions {
    public var melCompute: MLComputeUnits
    public var audioEncoderCompute: MLComputeUnits
    public var textDecoderCompute: MLComputeUnits
    public var prefillCompute: MLComputeUnits

    public init(
        melCompute: MLComputeUnits = .cpuAndGPU,
        audioEncoderCompute: MLComputeUnits? = nil,
        textDecoderCompute: MLComputeUnits = .cpuAndNeuralEngine,
        prefillCompute: MLComputeUnits = .cpuOnly
    ) {
        if WhisperKit.isRunningOnSimulator {
            self.melCompute = .cpuOnly
            self.audioEncoderCompute = .cpuOnly
            self.textDecoderCompute = .cpuOnly
            self.prefillCompute = .cpuOnly
            return
        }

        self.melCompute = melCompute
        self.prefillCompute = prefillCompute
        self.textDecoderCompute = textDecoderCompute

        if #available(macOS 14.0, iOS 17.0, *) {
            self.audioEncoderCompute = audioEncoderCompute ?? .cpuAndNeuralEngine
        } else {
            self.audioEncoderCompute = audioEncoderCompute ?? .cpuAndGPU
        }
    }
}

// MARK: - Decoding

public enum DecodingTask: CustomStringConvertible, CaseIterable {
    case transcribe
    case translate

    public var description: String {
        switch self {
            case .transcribe:
                return "transcribe"
            case .translate:
                return "translate"
        }
    }
}

public struct DecodingInputs {
    var initialPrompt: [Int]
    var inputIds: MLMultiArray
    var cacheLength: MLMultiArray
    var keyCache: MLMultiArray
    var valueCache: MLMultiArray
    var alignmentWeights: MLMultiArray
    var kvCacheUpdateMask: MLMultiArray
    var decoderKeyPaddingMask: MLMultiArray
    var prefillKeyCache: MLMultiArray
    var prefillValueCache: MLMultiArray

    func reset(prefilledCacheSize: Int, maxTokenContext: Int) {
        // NOTE: Because we have a mask on the kvcache,
        // we can simply shift the masks without touching the data,
        // it will be overwritten by the new data without impact on the output
        cacheLength[0] = NSNumber(value: prefilledCacheSize - 1)

        // Store token history and
        // Reset masks to prepare for next window
        for i in 0..<maxTokenContext {
            if i <= prefilledCacheSize - 1 {
                // Inside overlap window
                decoderKeyPaddingMask[i] = 0
                kvCacheUpdateMask[i - 1] = 0
                kvCacheUpdateMask[i] = 1
            } else {
                // Padding
                decoderKeyPaddingMask[i] = -10000
                kvCacheUpdateMask[i] = 0
            }
        }
    }
}

public struct DecodingCache {
    var keyCache: MLMultiArray?
    var valueCache: MLMultiArray?
    var alignmentWeights: MLMultiArray?
}

/// Options for how to transcribe an audio file using WhisperKit.
///
/// - Parameters:
///   - verbose: Whether to display the text being decoded to the console.
///              If true, displays all details; if false, displays minimal details;
///   - task: Whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')
///   - language: Language spoken in the audio
///   - temperature: Temperature to use for sampling.
///   - temperatureIncrementOnFallback: Increment which will be
///                  successively added to temperature upon failures according to either `compressionRatioThreshold`
///                  or `logProbThreshold`.
///   - temperatureFallbackCount: Number of times to increment temperature on fallback.
///   - sampleLength: The maximum number of tokens to sample.
///   - topK: Number of candidates when sampling with non-zero temperature.
///   - usePrefillPrompt: If true, the prefill tokens will be forced according to task and language settings.
///   - usePrefillCache: If true, the kv cache will be prefilled based on the prefill data mlmodel.
///   - detectLanguage: Use this in conjuntion with `usePrefillPrompt: true` to detect the language of the input audio.
///   - skipSpecialTokens: Whether to skip special tokens in the output.
///   - withoutTimestamps: Whether to include timestamps in the transcription result.
///   - wordTimestamps: Whether to include word-level timestamps in the transcription result.
///   - maxInitialTimestamp: Maximal initial timestamp.
///   - clipTimestamps: Array of timestamps (in seconds) to split the audio into segments for transcription.
///   - promptTokens: Array of token IDs to use as the conditioning prompt for the decoder. These are prepended to the prefill tokens.
///   - prefixTokens: Array of token IDs to use as the initial prefix for the decoder. These are appended to the prefill tokens.
///   - suppressBlank: If true, blank tokens will be suppressed during decoding.
///   - supressTokens: List of token IDs to suppress during decoding.
///   - compressionRatioThreshold: If the compression ratio of the transcription text is above this value, it is too repetitive and treated as failed.
///   - logProbThreshold: If the average log probability over sampled tokens is below this value, treat as failed.
///   - firstTokenLogProbThreshold: If the log probability over the first sampled token is below this value, treat as failed.
///   - noSpeechThreshold: If the no speech probability is higher than this value AND the average log
///                        probability over sampled tokens is below `logProbThreshold`, consider the segment as silent.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public struct DecodingOptions {
    public var verbose: Bool
    public var task: DecodingTask
    public var language: String?
    public var temperature: Float
    public var temperatureIncrementOnFallback: Float
    public var temperatureFallbackCount: Int
    public var sampleLength: Int
    public var topK: Int
    public var usePrefillPrompt: Bool
    public var usePrefillCache: Bool
    public var detectLanguage: Bool
    public var skipSpecialTokens: Bool
    public var withoutTimestamps: Bool
    public var wordTimestamps: Bool
    public var maxInitialTimestamp: Float?
    public var clipTimestamps: [Float]
    public var promptTokens: [Int]?
    public var prefixTokens: [Int]?
    public var suppressBlank: Bool
    public var supressTokens: [Int]
    public var compressionRatioThreshold: Float?
    public var logProbThreshold: Float?
    public var firstTokenLogProbThreshold: Float?
    public var noSpeechThreshold: Float?
    public var concurrentWorkerCount: Int

    public init(verbose: Bool = false,
                task: DecodingTask = .transcribe,
                language: String? = nil,
                temperature: Float = 0.0,
                temperatureIncrementOnFallback: Float = 0.2,
                temperatureFallbackCount: Int = 5,
                sampleLength: Int = Constants.maxTokenContext,
                topK: Int = 5,
                usePrefillPrompt: Bool = true,
                usePrefillCache: Bool = true,
                detectLanguage: Bool? = nil,
                skipSpecialTokens: Bool = false,
                withoutTimestamps: Bool = false,
                wordTimestamps: Bool = false,
                maxInitialTimestamp: Float? = nil,
                clipTimestamps: [Float] = [],
                promptTokens: [Int]? = nil,
                prefixTokens: [Int]? = nil,
                suppressBlank: Bool = false,
                supressTokens: [Int]? = nil,
                compressionRatioThreshold: Float? = 2.4,
                logProbThreshold: Float? = -1.0,
                firstTokenLogProbThreshold: Float? = -1.5,
                noSpeechThreshold: Float? = 0.6,
                concurrentWorkerCount: Int = 0)
    {
        self.verbose = verbose
        self.task = task
        self.language = language
        self.temperature = temperature
        self.temperatureIncrementOnFallback = temperatureIncrementOnFallback
        self.temperatureFallbackCount = temperatureFallbackCount
        self.sampleLength = sampleLength
        self.topK = topK
        self.usePrefillPrompt = usePrefillPrompt
        self.usePrefillCache = usePrefillCache
        self.detectLanguage = detectLanguage ?? !usePrefillPrompt // If prefill is false, detect language by default
        self.skipSpecialTokens = skipSpecialTokens
        self.withoutTimestamps = withoutTimestamps
        self.wordTimestamps = wordTimestamps
        self.maxInitialTimestamp = maxInitialTimestamp
        self.clipTimestamps = clipTimestamps
        self.promptTokens = promptTokens
        self.prefixTokens = prefixTokens
        self.suppressBlank = suppressBlank
        self.supressTokens = supressTokens ?? [] // nonSpeechTokens() // TODO: implement these as default
        self.compressionRatioThreshold = compressionRatioThreshold
        self.logProbThreshold = logProbThreshold
        self.firstTokenLogProbThreshold = firstTokenLogProbThreshold
        self.noSpeechThreshold = noSpeechThreshold
        self.concurrentWorkerCount = concurrentWorkerCount
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public struct DecodingFallback {
    public var needsFallback: Bool
    public var fallbackReason: String

    public init(needsFallback: Bool, fallbackReason: String) {
        self.needsFallback = needsFallback
        self.fallbackReason = fallbackReason
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public extension DecodingFallback {
    init?(
        options: DecodingOptions,
        isFirstTokenLogProbTooLow: Bool,
        noSpeechProb: Float,
        compressionRatio: Float,
        avgLogProb: Float
    ) {
        // NOTE: order matters here
        if isFirstTokenLogProbTooLow {
            self.init(needsFallback: true, fallbackReason: "firstTokenLogProbThreshold")
        } else if let threshold = options.noSpeechThreshold, noSpeechProb > threshold {
            // silence detected
            self.init(needsFallback: false, fallbackReason: "silence")
        } else if let threshold = options.compressionRatioThreshold, compressionRatio > threshold {
            // too repetitive
            self.init(needsFallback: true, fallbackReason: "compressionRatioThreshold")
        } else if let threshold = options.logProbThreshold, avgLogProb < threshold {
            // average log probablity too low (model is not confident enough)
            self.init(needsFallback: true, fallbackReason: "logProbThreshold")
        } else {
            return nil
        }
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public struct DecodingResult {
    public var language: String
    public var languageProbs: [String: Float]
    public var tokens: [Int]
    public var tokenLogProbs: [[Int: Float]]
    public var text: String
    public var avgLogProb: Float
    public var noSpeechProb: Float
    public var temperature: Float
    public var compressionRatio: Float
    public var cache: DecodingCache?
    public var timings: TranscriptionTimings?
    public var fallback: DecodingFallback?

    public static var emptyResults: DecodingResult {
        return DecodingResult(language: "",
                              languageProbs: [:],
                              tokens: [],
                              tokenLogProbs: [],
                              text: "",
                              avgLogProb: 0.0,
                              noSpeechProb: 0.0,
                              temperature: 0.0,
                              compressionRatio: 0.0,
                              cache: nil,
                              timings: nil,
                              fallback: nil)
    }
}

public enum WhisperError: Error, LocalizedError, Equatable {
    case tokenizerUnavailable(String = "Tokenizer is unavailable")
    case modelsUnavailable(String = "Models are unavailable")
    case prefillFailed(String = "Prefill failed")
    case audioProcessingFailed(String = "Audio processing failed")
    case decodingLogitsFailed(String = "Unable to decode logits from the model output")
    case segmentingFailed(String = "Creating segments failed")
    case loadAudioFailed(String = "Load audio failed")
    case prepareDecoderInputsFailed(String = "Prepare decoder inputs failed")
    case transcriptionFailed(String = "Transcription failed")
    case decodingFailed(String = "Decoding failed")
    case microphoneUnavailable(String = "No available microphone to record or stream")

    public var errorDescription: String? {
        switch self {
            case let .tokenizerUnavailable(message):
                Logging.error(message)
                return message
            case let .modelsUnavailable(message):
                Logging.error(message)
                return message
            case let .prefillFailed(message):
                Logging.error(message)
                return message
            case let .audioProcessingFailed(message):
                Logging.error(message)
                return message
            case let .decodingLogitsFailed(message):
                Logging.error(message)
                return message
            case let .segmentingFailed(message):
                Logging.error(message)
                return message
            case let .loadAudioFailed(message):
                Logging.error(message)
                return message
            case let .prepareDecoderInputsFailed(message):
                Logging.error(message)
                return message
            case let .transcriptionFailed(message):
                Logging.error(message)
                return message
            case let .decodingFailed(message):
                Logging.error(message)
                return message
            case let .microphoneUnavailable(message):
                Logging.error(message)
                return message
        }
    }
}

// Structs

public struct TranscriptionResult: Codable {
    public var text: String
    public var segments: [TranscriptionSegment]
    public var language: String
    public var timings: TranscriptionTimings

    func logSegments() {
        for segment in segments {
            let start = segment.start
            let end = segment.end
            let text = segment.text
            let line = "[\(formatTimestamp(start)) --> \(formatTimestamp(end))] \(text)"
            Logging.debug(line)
        }
    }

    func logTimings() {
        let decodeLoopTime = timings.decodingLoop
        let totalLoops = timings.totalDecodingLoops
        let timeToFirstToken = timings.firstTokenTime - timings.pipelineStart
        let tokensPerSecond = timings.tokensPerSecond
        let rtf = timings.realTimeFactor
        let totalTokens = segments.reduce(0) { $0 + $1.tokens.count }

        let fullPipelineDuration = timings.fullPipeline * 1000 // Convert to milliseconds

        let audioLoadTime = formatTimeWithPercentage(timings.audioLoading, 1, fullPipelineDuration)
        let audioProcTime = formatTimeWithPercentage(timings.audioProcessing, timings.totalAudioProcessingRuns, fullPipelineDuration)
        let logmelsTime = formatTimeWithPercentage(timings.logmels, timings.totalLogmelRuns, fullPipelineDuration)
        let encodingTime = formatTimeWithPercentage(timings.encoding, timings.totalEncodingRuns, fullPipelineDuration)
        let decodingInitTime = formatTimeWithPercentage(timings.decodingInit, 1, fullPipelineDuration)
        let prefillInfo = formatTimeWithPercentage(timings.prefill, 1, fullPipelineDuration)
        let predictionsInfo = formatTimeWithPercentage(timings.decodingPredictions, totalLoops, fullPipelineDuration)
        let filteringInfo = formatTimeWithPercentage(timings.decodingFiltering, totalLoops, fullPipelineDuration)
        let samplingInfo = formatTimeWithPercentage(timings.decodingSampling, totalLoops, fullPipelineDuration)
        let kvCachingInfo = formatTimeWithPercentage(timings.decodingKvCaching, timings.totalKVUpdateRuns, fullPipelineDuration)
        let wordTimestampInfo = formatTimeWithPercentage(timings.decodingWordTimestamps, timings.totalTimestampAlignmentRuns, fullPipelineDuration)
        let nonPredTimeInfo = formatTimeWithPercentage(timings.decodingNonPrediction, totalLoops, fullPipelineDuration)
        let windowingInfo = formatTimeWithPercentage(timings.decodingWindowing - timings.decodingWordTimestamps, timings.totalDecodingWindows, fullPipelineDuration)
        let fallbackInfo = formatTimeWithPercentage(timings.decodingFallback, timings.totalDecodingFallbacks, fullPipelineDuration)
        let decodingLoopInfo = formatTimeWithPercentage(timings.decodingLoop, totalLoops, fullPipelineDuration)

        // Logging
        Logging.info("---- Transcription Timings ----")

        Logging.info("Audio Load:          \(audioLoadTime)")
        Logging.info("Audio Processing:    \(audioProcTime)")
        Logging.info("Mels:                \(logmelsTime)")
        Logging.info("Encoding:            \(encodingTime)")
        Logging.info("Matrices Init:       \(decodingInitTime)")
        Logging.info("Prefill:             \(prefillInfo)")
        Logging.info("Decoding:            \(predictionsInfo)")
        Logging.info("Non-inference:       \(nonPredTimeInfo)")
        Logging.info("- Logit Filtering:   \(filteringInfo)")
        Logging.info("- Sampling:          \(samplingInfo)")
        Logging.info("- Kv Caching:        \(kvCachingInfo)")
        Logging.info("- Word Timestamps:   \(wordTimestampInfo)")
        Logging.info("- Windowing:         \(windowingInfo)")
        Logging.info("Fallbacks:           \(fallbackInfo)")
        Logging.info("Decoding Full Loop:  \(decodingLoopInfo)")
        Logging.info("-------------------------------")

        // Summary statistics
        Logging.info("Model Load Time:     \(String(format: "%.2f", timings.modelLoading)) seconds")
        Logging.info("Inference Duration:  \(String(format: "%.2f", timings.fullPipeline)) seconds")
        Logging.info("- Decoding Loop:     \(String(format: "%.2f", decodeLoopTime)) seconds")
        Logging.info("Time to first token: \(String(format: "%.2f", timeToFirstToken)) seconds")
        Logging.info("Total Tokens:        \(totalTokens)")
        Logging.info("Tokens per Second:   \(String(format: "%.2f", tokensPerSecond)) tok/s")
        Logging.info("Real Time Factor:    \(String(format: "%.2f", rtf))")
        Logging.info("Fallbacks:           \(timings.totalDecodingFallbacks)")
    }
}

public extension TranscriptionResult {
    var allWords: [WordTiming] {
        return segments.compactMap { $0.words }.flatMap { $0 }
    }
}

public struct TranscriptionSegment: Hashable, Codable {
    public var id: Int = 0
    public var seek: Int = 0
    public var start: Float = 0.0
    public var end: Float = 0.0
    public var text: String = ""
    public var tokens: [Int] = []
    public var tokenLogProbs: [[Int: Float]] = [[:]]
    public var temperature: Float = 1.0
    public var avgLogprob: Float = 0.0
    public var compressionRatio: Float = 1.0
    public var noSpeechProb: Float = 0.0
    public var words: [WordTiming]? = nil
}

public struct WordTiming: Hashable, Codable {
    public var word: String
    public var tokens: [Int]
    public var start: Float
    public var end: Float
    public var probability: Float
}

public struct TranscriptionProgress {
    public var timings: TranscriptionTimings
    public var text: String
    public var tokens: [Int]
    public var temperature: Float?
    public var avgLogprob: Float?
    public var compressionRatio: Float?
}

/// Callback to receive progress updates during transcription.
/// Return `false` to force the transcription to stop early.
public typealias TranscriptionCallback = ((TranscriptionProgress) -> Bool?)?

public struct TranscriptionTimings: Codable {
    public var pipelineStart: CFAbsoluteTime
    public var firstTokenTime: CFAbsoluteTime
    public var inputAudioSeconds: TimeInterval
    public var modelLoading: TimeInterval
    public var audioLoading: TimeInterval
    public var audioProcessing: TimeInterval
    public var logmels: TimeInterval
    public var encoding: TimeInterval
    public var prefill: TimeInterval
    public var decodingInit: TimeInterval
    public var decodingLoop: TimeInterval
    public var decodingPredictions: TimeInterval
    public var decodingFiltering: TimeInterval
    public var decodingSampling: TimeInterval
    public var decodingFallback: TimeInterval
    public var decodingWindowing: TimeInterval
    public var decodingKvCaching: TimeInterval
    public var decodingWordTimestamps: TimeInterval
    public var decodingNonPrediction: TimeInterval
    public var totalAudioProcessingRuns: Double
    public var totalLogmelRuns: Double
    public var totalEncodingRuns: Double
    public var totalDecodingLoops: Double
    public var totalKVUpdateRuns: Double
    public var totalTimestampAlignmentRuns: Double
    public var totalDecodingFallbacks: Double
    public var totalDecodingWindows: Double
    public var fullPipeline: TimeInterval

    /// Computed properties
    public var tokensPerSecond: Double {
        Double(totalDecodingLoops) / Double(decodingLoop)
    }

    public var realTimeFactor: Double {
        decodingLoop / inputAudioSeconds
    }

    /// Initialize with all time intervals set to zero.
    public init(modelLoading: TimeInterval = 0,
                audioLoading: TimeInterval = 0,
                audioProcessing: TimeInterval = 0,
                logmels: TimeInterval = 0,
                encoding: TimeInterval = 0,
                prefill: TimeInterval = 0,
                decodingInit: TimeInterval = 0,
                decodingLoop: TimeInterval = 0,
                decodingPredictions: TimeInterval = 0,
                decodingFiltering: TimeInterval = 0,
                decodingSampling: TimeInterval = 0,
                decodingFallback: TimeInterval = 0,
                decodingWindowing: TimeInterval = 0,
                decodingKvCaching: TimeInterval = 0,
                decodingTimestampAlignment: TimeInterval = 0,
                decodingNonPrediction: TimeInterval = 0,
                totalAudioProcessingRuns: Double = 0,
                totalLogmelRuns: Double = 0,
                totalEncodingRuns: Double = 0,
                totalDecodingLoops: Double = 0,
                totalKVUpdateRuns: Double = 0,
                totalTimestampAlignmentRuns: Double = 0,
                totalDecodingFallbacks: Double = 0,
                totalDecodingWindows: Double = 0,
                fullPipeline: TimeInterval = 0)
    {
        self.pipelineStart = CFAbsoluteTimeGetCurrent()
        self.firstTokenTime = 0
        self.inputAudioSeconds = 0.001
        self.modelLoading = modelLoading
        self.audioLoading = audioLoading
        self.audioProcessing = audioProcessing
        self.logmels = logmels
        self.encoding = encoding
        self.prefill = prefill
        self.decodingInit = decodingInit
        self.decodingLoop = decodingLoop
        self.decodingPredictions = decodingPredictions
        self.decodingFiltering = decodingFiltering
        self.decodingSampling = decodingSampling
        self.decodingFallback = decodingFallback
        self.decodingWindowing = decodingWindowing
        self.decodingKvCaching = decodingKvCaching
        self.decodingWordTimestamps = decodingTimestampAlignment
        self.decodingNonPrediction = decodingNonPrediction
        self.totalAudioProcessingRuns = totalAudioProcessingRuns
        self.totalLogmelRuns = totalLogmelRuns
        self.totalEncodingRuns = totalEncodingRuns
        self.totalDecodingLoops = totalDecodingLoops
        self.totalKVUpdateRuns = totalKVUpdateRuns
        self.totalTimestampAlignmentRuns = totalTimestampAlignmentRuns
        self.totalDecodingFallbacks = totalDecodingFallbacks
        self.totalDecodingWindows = totalDecodingWindows
        self.fullPipeline = fullPipeline
    }
}

// MARK: MelSpectrogram

public class MelSpectrogramInput: MLFeatureProvider {
    /// audio as 480000 element vector of floats
    public var audio: MLMultiArray

    public var featureNames: Set<String> {
        return ["audio"]
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "audio" {
            return MLFeatureValue(multiArray: self.audio)
        }
        return nil
    }

    public init(audio: MLMultiArray) {
        self.audio = audio
    }

    public convenience init(audio: MLShapedArray<Float>) {
        self.init(audio: MLMultiArray(audio))
    }
}

/// Model Prediction Output Type
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class MelSpectrogramOutput: MLFeatureProvider {
    /// Source provided by CoreML
    private let provider: MLFeatureProvider

    /// melspectrogram_features as 1 × 80 × 1 × 3000 4-dimensional array of 16-bit floats
    public var melspectrogramFeatures: MLMultiArray {
        return self.provider.featureValue(for: "melspectrogram_features")!.multiArrayValue!
    }

    /// melspectrogram_features as 1 × 80 × 1 × 3000 4-dimensional array of 16-bit floats
    @available(macOS, unavailable)
    @available(macCatalyst, unavailable)
    public var melspectrogram_featuresShapedArray: MLShapedArray<Float16> {
        return MLShapedArray<Float16>(self.melspectrogramFeatures)
    }

    public var featureNames: Set<String> {
        return self.provider.featureNames
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    public init(melspectrogram_features: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["melspectrogram_features": MLFeatureValue(multiArray: melspectrogram_features)])
    }

    public init(features: MLFeatureProvider) {
        self.provider = features
    }
}

// MARK: AudioEncoder

/// Model Prediction Input Type
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class AudioEncoderInput: MLFeatureProvider {
    /// melspectrogram_features as 1 × {80,128} × 1 × 3000 4-dimensional array of floats
    public var melspectrogram_features: MLMultiArray

    public var featureNames: Set<String> {
        return ["melspectrogram_features"]
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "melspectrogram_features" {
            return MLFeatureValue(multiArray: self.melspectrogram_features)
        }
        return nil
    }

    public init(melspectrogram_features: MLMultiArray) {
        self.melspectrogram_features = melspectrogram_features
    }

    public convenience init(melspectrogram_features: MLShapedArray<Float>) {
        self.init(melspectrogram_features: MLMultiArray(melspectrogram_features))
    }
}

/// Model Prediction Output Type
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class AudioEncoderOutput: MLFeatureProvider {
    /// Source provided by CoreML
    private let provider: MLFeatureProvider

    /// encoder_output_embeds as 1 × embedDim × 1 × sequenceLength 4-dimensional array of 16-bit floats
    public var encoder_output_embeds: MLMultiArray {
        return self.provider.featureValue(for: "encoder_output_embeds")!.multiArrayValue!
    }

    /// encoder_output_embeds as 1 × embedDim × 1 × sequenceLength 4-dimensional array of 16-bit floats
    @available(macOS, unavailable)
    @available(macCatalyst, unavailable)
    public var encoder_output_embedsShapedArray: MLShapedArray<Float16> {
        return MLShapedArray<Float16>(self.encoder_output_embeds)
    }

    public var featureNames: Set<String> {
        return self.provider.featureNames
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    public init(encoder_output_embeds: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["encoder_output_embeds": MLFeatureValue(multiArray: encoder_output_embeds)])
    }

    public init(features: MLFeatureProvider) {
        self.provider = features
    }
}

// MARK: TextDecoder

/// Model Prediction Input Type
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class TextDecoderInput: MLFeatureProvider {
    /// input_ids as 1 element vector of 32-bit integers
    public var input_ids: MLMultiArray

    /// cache_length as 1 element vector of 32-bit integers
    public var cache_length: MLMultiArray

    /// key_cache as 1 × kvCacheEmbedDim × 1 × kvCacheMaxSequenceLength 4-dimensional array of floats
    public var key_cache: MLMultiArray

    /// value_cache as 1 × kvCacheEmbedDim × 1 × kvCacheMaxSequenceLength 4-dimensional array of floats
    public var value_cache: MLMultiArray

    /// kv_cache_update_mask as 1 by kvCacheMaxSequenceLength matrix of floats
    public var kv_cache_update_mask: MLMultiArray

    /// encoder_output_embeds as 1 × embedSize × 1 × sequenceLength 4-dimensional array of floats
    public var encoder_output_embeds: MLMultiArray

    /// decoder_key_padding_mask as 1 by kvCacheMaxSequenceLength matrix of floats
    public var decoder_key_padding_mask: MLMultiArray

    public var featureNames: Set<String> {
        return ["input_ids", "cache_length", "key_cache", "value_cache", "kv_cache_update_mask", "encoder_output_embeds", "decoder_key_padding_mask"]
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "input_ids" {
            return MLFeatureValue(multiArray: self.input_ids)
        }
        if featureName == "cache_length" {
            return MLFeatureValue(multiArray: self.cache_length)
        }
        if featureName == "key_cache" {
            return MLFeatureValue(multiArray: self.key_cache)
        }
        if featureName == "value_cache" {
            return MLFeatureValue(multiArray: self.value_cache)
        }
        if featureName == "kv_cache_update_mask" {
            return MLFeatureValue(multiArray: self.kv_cache_update_mask)
        }
        if featureName == "encoder_output_embeds" {
            return MLFeatureValue(multiArray: self.encoder_output_embeds)
        }
        if featureName == "decoder_key_padding_mask" {
            return MLFeatureValue(multiArray: self.decoder_key_padding_mask)
        }
        return nil
    }

    public init(input_ids: MLMultiArray, cache_length: MLMultiArray, key_cache: MLMultiArray, value_cache: MLMultiArray, kv_cache_update_mask: MLMultiArray, encoder_output_embeds: MLMultiArray, decoder_key_padding_mask: MLMultiArray) {
        self.input_ids = input_ids
        self.cache_length = cache_length
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.kv_cache_update_mask = kv_cache_update_mask
        self.encoder_output_embeds = encoder_output_embeds
        self.decoder_key_padding_mask = decoder_key_padding_mask
    }

    public convenience init(input_ids: MLShapedArray<Int32>, cache_length: MLShapedArray<Int32>, key_cache: MLShapedArray<Float>, value_cache: MLShapedArray<Float>, kv_cache_update_mask: MLShapedArray<Float>, encoder_output_embeds: MLShapedArray<Float>, decoder_key_padding_mask: MLShapedArray<Float>) {
        self.init(input_ids: MLMultiArray(input_ids), cache_length: MLMultiArray(cache_length), key_cache: MLMultiArray(key_cache), value_cache: MLMultiArray(value_cache), kv_cache_update_mask: MLMultiArray(kv_cache_update_mask), encoder_output_embeds: MLMultiArray(encoder_output_embeds), decoder_key_padding_mask: MLMultiArray(decoder_key_padding_mask))
    }
}

/// Model Prediction Output Type
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class TextDecoderOutput: MLFeatureProvider {
    /// Source provided by CoreML
    private let provider: MLFeatureProvider

    /// logits as 1 × vocab size × 1 × 1 4-dimensional array of 16-bit floats
    public var logits: MLMultiArray {
        return self.provider.featureValue(for: "logits")!.multiArrayValue!
    }

    /// logits as 1 × vocab size × 1 × 1 4-dimensional array of 16-bit floats
    @available(macOS, unavailable)
    @available(macCatalyst, unavailable)
    public var logitsShapedArray: MLShapedArray<Float16> {
        return MLShapedArray<Float16>(self.logits)
    }

    /// key_cache_updates as 1 × embed_dim * num_layers × 1 × 1 4-dimensional array of 16-bit floats
    public var key_cache_updates: MLMultiArray {
        return self.provider.featureValue(for: "key_cache_updates")!.multiArrayValue!
    }

    /// key_cache_updates as 1 × embed_dim * num_layers × 1 × 1 4-dimensional array of 16-bit floats
    @available(macOS, unavailable)
    @available(macCatalyst, unavailable)
    public var key_cache_updatesShapedArray: MLShapedArray<Float16> {
        return MLShapedArray<Float16>(self.key_cache_updates)
    }

    /// value_cache_updates as 1 × kvCacheEmbedDim × 1 × 1 4-dimensional array of 16-bit floats
    public var value_cache_updates: MLMultiArray {
        return self.provider.featureValue(for: "value_cache_updates")!.multiArrayValue!
    }

    /// value_cache_updates as 1 × kvCacheEmbedDim × 1 × 1 4-dimensional array of 16-bit floats
    @available(macOS, unavailable)
    @available(macCatalyst, unavailable)
    public var value_cache_updatesShapedArray: MLShapedArray<Float16> {
        return MLShapedArray<Float16>(self.value_cache_updates)
    }

    /// alignment_heads_weights as 1 × 1500 2-dimensional array of 16-bit floats
    public var alignment_heads_weights: MLMultiArray? {
        return self.provider.featureValue(for: "alignment_heads_weights")?.multiArrayValue
    }

    /// alignment_heads_weights as 1 × 1500 2-dimensional array of 16-bit floats
    @available(macOS, unavailable)
    @available(macCatalyst, unavailable)
    public var alignment_heads_weightsShapedArray: MLShapedArray<Float16>? {
        guard let alignment_heads_weights = self.alignment_heads_weights else {
            return nil
        }
        return MLShapedArray<Float16>(alignment_heads_weights)
    }

    public var featureNames: Set<String> {
        return self.provider.featureNames
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    public init(logits: MLMultiArray, key_cache_updates: MLMultiArray, value_cache_updates: MLMultiArray, logits_argmax _: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["logits": MLFeatureValue(multiArray: logits), "key_cache_updates": MLFeatureValue(multiArray: key_cache_updates), "value_cache_updates": MLFeatureValue(multiArray: value_cache_updates)])
    }

    public init(features: MLFeatureProvider) {
        self.provider = features
    }
}

// MARK: TextDecoderCachePrefill

public class TextDecoderCachePrefillInput: MLFeatureProvider {
    /// task as 1 element vector of 32-bit integers
    public var task: MLMultiArray

    /// language as 1 element vector of 32-bit integers
    public var language: MLMultiArray

    public var featureNames: Set<String> {
        return ["task", "language"]
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "task" {
            return MLFeatureValue(multiArray: self.task)
        }
        if featureName == "language" {
            return MLFeatureValue(multiArray: self.language)
        }
        return nil
    }

    public init(task: MLMultiArray, language: MLMultiArray) {
        self.task = task
        self.language = language
    }

    public convenience init(task: MLShapedArray<Int32>, language: MLShapedArray<Int32>) {
        self.init(task: MLMultiArray(task), language: MLMultiArray(language))
    }
}

/// Model Prediction Output Type
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class TextDecoderCachePrefillOutput: MLFeatureProvider {
    /// Source provided by CoreML
    private let provider: MLFeatureProvider

    /// key_cache_prefill as 1 × embed_dim * num_layers × 1 × 3 4-dimensional array of 16-bit floats
    public var key_cache_prefill: MLMultiArray {
        return self.provider.featureValue(for: "key_cache_prefill")!.multiArrayValue!
    }

    /// key_cache_prefill as 1 × embed_dim * num_layers × 1 × 3 4-dimensional array of 16-bit floats
    @available(macOS, unavailable)
    @available(macCatalyst, unavailable)
    public var key_cache_prefillShapedArray: MLShapedArray<Float16> {
        return MLShapedArray<Float16>(self.key_cache_prefill)
    }

    /// value_cache_prefill as 1 × embed_dim * num_layers × 1 × 3 4-dimensional array of 16-bit floats
    public var value_cache_prefill: MLMultiArray {
        return self.provider.featureValue(for: "value_cache_prefill")!.multiArrayValue!
    }

    /// value_cache_prefill as 1 × embed_dim * num_layers × 1 × 3 4-dimensional array of 16-bit floats
    @available(macOS, unavailable)
    @available(macCatalyst, unavailable)
    public var value_cache_prefillShapedArray: MLShapedArray<Float16> {
        return MLShapedArray<Float16>(self.value_cache_prefill)
    }

    public var featureNames: Set<String> {
        return self.provider.featureNames
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    public init(key_cache_prefill: MLMultiArray, value_cache_prefill: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["key_cache_prefill": MLFeatureValue(multiArray: key_cache_prefill), "value_cache_prefill": MLFeatureValue(multiArray: value_cache_prefill)])
    }

    public init(features: MLFeatureProvider) {
        self.provider = features
    }
}

// MARK: SpecialTokens

public struct SpecialTokens {
    public let endToken: Int
    public let englishToken: Int
    public let noSpeechToken: Int
    public let noTimestampsToken: Int
    public let specialTokenBegin: Int
    public let startOfPreviousToken: Int
    public let startOfTranscriptToken: Int
    public let timeTokenBegin: Int
    public let transcribeToken: Int
    public let translateToken: Int
    public let whitespaceToken: Int

    public init(
        endToken: Int,
        englishToken: Int,
        noSpeechToken: Int,
        noTimestampsToken: Int,
        specialTokenBegin: Int,
        startOfPreviousToken: Int,
        startOfTranscriptToken: Int,
        timeTokenBegin: Int,
        transcribeToken: Int,
        translateToken: Int,
        whitespaceToken: Int
    ) {
        self.endToken = endToken
        self.englishToken = englishToken
        self.noSpeechToken = noSpeechToken
        self.noTimestampsToken = noTimestampsToken
        self.specialTokenBegin = specialTokenBegin
        self.startOfPreviousToken = startOfPreviousToken
        self.startOfTranscriptToken = startOfTranscriptToken
        self.timeTokenBegin = timeTokenBegin
        self.transcribeToken = transcribeToken
        self.translateToken = translateToken
        self.whitespaceToken = whitespaceToken
    }
}

public protocol WhisperTokenizer: Tokenizer {
    var specialTokens: SpecialTokens { get }
    var allLanguageTokens: Set<Int> { get }

    func splitToWordTokens(tokenIds: [Int]) -> (words: [String], wordTokens: [[Int]])
}

struct WhisperTokenizerWrapper: WhisperTokenizer {
    let tokenizer: any Tokenizer
    let specialTokens: SpecialTokens
    let allLanguageTokens: Set<Int>

    init(tokenizer: any Tokenizer) {
        let specialTokens = SpecialTokens(
            endToken: tokenizer.convertTokenToId("<|endoftext|>") ?? Self.defaultEndToken,
            englishToken: tokenizer.convertTokenToId("<|en|>") ?? Self.defaultEnglishToken,
            noSpeechToken: tokenizer.convertTokenToId("<|nospeech|>") ?? Self.defaultNoSpeechToken,
            noTimestampsToken: tokenizer.convertTokenToId("<|notimestamps|>") ?? Self.defaultNoTimestampsToken,
            specialTokenBegin: tokenizer.convertTokenToId("<|endoftext|>") ?? Self.defaultSpecialTokenBegin,
            startOfPreviousToken: tokenizer.convertTokenToId("<|startofprev|>") ?? Self.defaultStartOfPreviousToken,
            startOfTranscriptToken: tokenizer.convertTokenToId("<|startoftranscript|>") ?? Self.defaultStartOfTranscriptToken,
            timeTokenBegin: tokenizer.convertTokenToId("<|0.00|>") ?? Self.defaultTimeTokenBegin,
            transcribeToken: tokenizer.convertTokenToId("<|transcribe|>") ?? Self.defaultTranscribeToken,
            translateToken: tokenizer.convertTokenToId("<|translate|>") ?? Self.defaultTranslateToken,
            whitespaceToken: tokenizer.convertTokenToId(" ") ?? Self.defaultWhitespaceToken
        )
        self.tokenizer = tokenizer
        self.specialTokens = specialTokens
        self.allLanguageTokens = Set(
            Constants.languages
                .compactMap { tokenizer.convertTokenToId("<|\($0.value)|>") }
                .filter { $0 > specialTokens.specialTokenBegin }
        )
    }

    private func splitTokensOnUnicode(tokens: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        let decodedFull = tokenizer.decode(tokens: tokens)
        let replacementString = "\u{fffd}"

        var words: [String] = []
        var wordTokens: [[Int]] = []
        var currentTokens: [Int] = []
        var unicodeOffset = 0

        for token in tokens {
            currentTokens.append(token)
            let decoded = tokenizer.decode(tokens: currentTokens)

            var hasUnicodeInFullString = false
            if let range = decoded.range(of: replacementString) {
                hasUnicodeInFullString = decodedFull[range] == replacementString
            }

            if !decoded.contains(replacementString) || hasUnicodeInFullString {
                words.append(decoded)
                wordTokens.append(currentTokens)
                currentTokens = []
                unicodeOffset += decoded.count
            }
        }

        return (words, wordTokens)
    }

    private func splitTokensOnSpaces(tokens: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        let (subwords, subwordTokensList) = splitTokensOnUnicode(tokens: tokens)
        var words: [String] = []
        var wordTokens: [[Int]] = []

        for (subword, subwordTokens) in zip(subwords, subwordTokensList) {
            let special = subwordTokens.first! >= specialTokens.specialTokenBegin
            let withSpace = subword.hasPrefix(" ")
            var punctuation = false
            if let strippedSubword = UnicodeScalar(subword.trimmingCharacters(in: .whitespaces)) {
                punctuation = CharacterSet.punctuationCharacters.contains(strippedSubword)
            }
            if special || withSpace || punctuation || words.isEmpty {
                words.append(subword)
                wordTokens.append(subwordTokens)
            } else {
                words[words.count - 1] += subword
                wordTokens[words.count - 1].append(contentsOf: subwordTokens)
            }
        }

        return (words, wordTokens)
    }

    private func isPunctuation(_ text: String, tokenRange: Range<String.Index>, tag: NLTag?) -> Bool {
        let punctuationCharacters = CharacterSet.punctuationCharacters
        let token = String(text[tokenRange])
        if let tag = tag, tag == .punctuation {
            return true
        } else if token.unicodeScalars.allSatisfy({ punctuationCharacters.contains($0) }) {
            return true
        }
        return false
    }

    /// Decodes token ids into individual words and per-word subtokens
    /// - Parameter tokenIds: Array of tokens to decode and then split
    /// - Returns: Tuple containing and array of the split words and all tokens for each word
    func splitToWordTokens(tokenIds: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        let decodedWords = tokenizer.decode(tokens: tokenIds.filter { $0 < specialTokens.specialTokenBegin })

        // Detect language of input text
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(decodedWords)
        let languageCode = recognizer.dominantLanguage?.rawValue

        if ["zh", "ja", "th", "lo", "my", "yue"].contains(languageCode) {
            return splitTokensOnUnicode(tokens: tokenIds)
        } else {
            return splitTokensOnSpaces(tokens: tokenIds)
        }
    }
}

extension WhisperTokenizerWrapper: Tokenizer {
    func tokenize(text: String) -> [String] {
        tokenizer.tokenize(text: text)
    }

    func encode(text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        tokenizer.convertIdToToken(id)
    }

    var bosToken: String? {
        tokenizer.bosToken
    }

    var bosTokenId: Int? {
        tokenizer.bosTokenId
    }

    var eosToken: String? {
        tokenizer.eosToken
    }

    var eosTokenId: Int? {
        tokenizer.eosTokenId
    }

    var unknownToken: String? {
        tokenizer.unknownToken
    }

    var unknownTokenId: Int? {
        tokenizer.unknownTokenId
    }
}

extension WhisperTokenizerWrapper {
    /// Default values for each token, using base vocab
    static var defaultWhitespaceToken: Int { 220 }
    static var defaultSpecialTokenBegin: Int { 50257 }
    static var defaultEndToken: Int { 50257 }
    static var defaultStartOfPreviousToken: Int { 50361 }
    static var defaultStartOfTranscriptToken: Int { 50258 }
    static var defaultEnglishToken: Int { 50259 }
    static var defaultTranscribeToken: Int { 50359 }
    static var defaultTranslateToken: Int { 50358 }
    static var defaultNoSpeechToken: Int { 50362 }
    static var defaultNoTimestampsToken: Int { 50363 }
    static var defaultTimeTokenBegin: Int { 50364 }
}

// MARK: Constants

public enum Constants {
    enum Logging {
        static let subsystem = "com.argmax.whisperkit"
    }

    static let specialTokenCharacters = CharacterSet(charactersIn: "<|>")

    public static let maxTokenContext = Int(448 / 2)
    public static let languages: [String: String] =
        [
            "english": "en",
            "chinese": "zh",
            "german": "de",
            "spanish": "es",
            "russian": "ru",
            "korean": "ko",
            "french": "fr",
            "japanese": "ja",
            "portuguese": "pt",
            "turkish": "tr",
            "polish": "pl",
            "catalan": "ca",
            "dutch": "nl",
            "arabic": "ar",
            "swedish": "sv",
            "italian": "it",
            "indonesian": "id",
            "hindi": "hi",
            "finnish": "fi",
            "vietnamese": "vi",
            "hebrew": "he",
            "ukrainian": "uk",
            "greek": "el",
            "malay": "ms",
            "czech": "cs",
            "romanian": "ro",
            "danish": "da",
            "hungarian": "hu",
            "tamil": "ta",
            "norwegian": "no",
            "thai": "th",
            "urdu": "ur",
            "croatian": "hr",
            "bulgarian": "bg",
            "lithuanian": "lt",
            "latin": "la",
            "maori": "mi",
            "malayalam": "ml",
            "welsh": "cy",
            "slovak": "sk",
            "telugu": "te",
            "persian": "fa",
            "latvian": "lv",
            "bengali": "bn",
            "serbian": "sr",
            "azerbaijani": "az",
            "slovenian": "sl",
            "kannada": "kn",
            "estonian": "et",
            "macedonian": "mk",
            "breton": "br",
            "basque": "eu",
            "icelandic": "is",
            "armenian": "hy",
            "nepali": "ne",
            "mongolian": "mn",
            "bosnian": "bs",
            "kazakh": "kk",
            "albanian": "sq",
            "swahili": "sw",
            "galician": "gl",
            "marathi": "mr",
            "punjabi": "pa",
            "sinhala": "si",
            "khmer": "km",
            "shona": "sn",
            "yoruba": "yo",
            "somali": "so",
            "afrikaans": "af",
            "occitan": "oc",
            "georgian": "ka",
            "belarusian": "be",
            "tajik": "tg",
            "sindhi": "sd",
            "gujarati": "gu",
            "amharic": "am",
            "yiddish": "yi",
            "lao": "lo",
            "uzbek": "uz",
            "faroese": "fo",
            "haitian creole": "ht",
            "pashto": "ps",
            "turkmen": "tk",
            "nynorsk": "nn",
            "maltese": "mt",
            "sanskrit": "sa",
            "luxembourgish": "lb",
            "myanmar": "my",
            "tibetan": "bo",
            "tagalog": "tl",
            "malagasy": "mg",
            "assamese": "as",
            "tatar": "tt",
            "hawaiian": "haw",
            "lingala": "ln",
            "hausa": "ha",
            "bashkir": "ba",
            "javanese": "jw",
            "sundanese": "su",
            "cantonese": "yue",
            "burmese": "my",
            "valencian": "ca",
            "flemish": "nl",
            "haitian": "ht",
            "letzeburgesch": "lb",
            "pushto": "ps",
            "panjabi": "pa",
            "moldavian": "ro",
            "moldovan": "ro",
            "sinhalese": "si",
            "castilian": "es",
            "mandarin": "zh",
        ]

    public static let languageCodes: Set<String> = Set(languages.values)

    public static let defaultLanguageCode: String = "en"
}
