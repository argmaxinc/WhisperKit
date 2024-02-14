//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Hub
import Tokenizers

#if os(watchOS) || arch(arm64)
    @available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
    public typealias FloatType = Float16
#else
    public typealias FloatType = Float
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

public struct ModelComputeOptions {
    public var melCompute: MLComputeUnits
    public var audioEncoderCompute: MLComputeUnits
    public var textDecoderCompute: MLComputeUnits
    public var prefillCompute: MLComputeUnits

    public init(
        melCompute: MLComputeUnits = .cpuAndGPU,
        audioEncoderCompute: MLComputeUnits = .cpuAndNeuralEngine,
        textDecoderCompute: MLComputeUnits = .cpuAndNeuralEngine,
        prefillCompute: MLComputeUnits = .cpuOnly
    ) {
        self.melCompute = melCompute
        self.audioEncoderCompute = audioEncoderCompute
        self.textDecoderCompute = textDecoderCompute
        self.prefillCompute = prefillCompute
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
    var kvCacheUpdateMask: MLMultiArray
    var decoderKeyPaddingMask: MLMultiArray
    var prefillKeyCache: MLMultiArray
    var prefillValueCache: MLMultiArray
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
///   - usePrefillPrompt: If true, the kv cache will be prefilled based on the prefill data mlmodel.
///   - skipSpecialTokens: Whether to skip special tokens in the output.
///   - withoutTimestamps: Whether to include timestamps in the transcription result.
///   - suppressBlank: If true, blank tokens will be suppressed during decoding.
///   - supressTokens: List of token IDs to suppress during decoding.
///   - compressionRatioThreshold: If the compression ratio of the transcription text is above this value, it is too repetitive and treated as failed.
///   - logProbThreshold: If the average log probability over sampled tokens is below this value, treat as failed.
///   - noSpeechThreshold: If the no speech probability is higher than this value AND the average log
///                        probability over sampled tokens is below `logProbThreshold`, consider the segment as silent.

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
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
    public var skipSpecialTokens: Bool
    public var withoutTimestamps: Bool
    public var clipTimestamps: [Float]
    public var suppressBlank: Bool
    public var supressTokens: [Int]
    public var compressionRatioThreshold: Float?
    public var logProbThreshold: Float?
    public var noSpeechThreshold: Float?

    public init(verbose: Bool = false,
                task: DecodingTask = .transcribe,
                language: String? = nil,
                temperature: Float = 0.0,
                temperatureIncrementOnFallback: Float = 0.2,
                temperatureFallbackCount: Int = 5,
                sampleLength: Int = WhisperKit.maxTokenContext,
                topK: Int = 5,
                usePrefillPrompt: Bool = true,
                usePrefillCache: Bool = true,
                skipSpecialTokens: Bool = false,
                withoutTimestamps: Bool = false,
                clipTimestamps: [Float] = [],
                suppressBlank: Bool = false,
                supressTokens: [Int]? = nil,
                compressionRatioThreshold: Float? = 2.4,
                logProbThreshold: Float? = -1.0,
                noSpeechThreshold: Float? = 0.6)
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
        self.skipSpecialTokens = skipSpecialTokens
        self.withoutTimestamps = withoutTimestamps
        self.clipTimestamps = clipTimestamps
        self.suppressBlank = suppressBlank
        self.supressTokens = supressTokens ?? [] // nonSpeechTokens() // TODO: implement these as default
        self.compressionRatioThreshold = compressionRatioThreshold
        self.logProbThreshold = logProbThreshold
        self.noSpeechThreshold = noSpeechThreshold
    }
}

public struct DecodingResult {
    public var language: String
    public var languageProbs: [String: Float]
    public var tokens: [Int]
    public var text: String
    public var avgLogProb: Float
    public var noSpeechProb: Float
    public var temperature: Float
    public var compressionRatio: Float
    public var timings: TranscriptionTimings?

    public static var emptyResults: DecodingResult {
        return DecodingResult(language: "",
                              languageProbs: [:],
                              tokens: [],
                              text: "",
                              avgLogProb: 0.0,
                              noSpeechProb: 0.0,
                              temperature: 0.0,
                              compressionRatio: 0.0,
                              timings: nil)
    }
}

enum WhisperError: Error, LocalizedError {
    case tokenizerUnavailable(String = "Tokenizer is unavailable")
    case modelsUnavailable(String = "Models are unavailable")
    case prefillFailed(String = "Prefill failed")
    case audioProcessingFailed(String = "Audio processing failed")
    case decodingLogitsFailed(String = "Unable to decode logits from the model output")

    var errorDescription: String? {
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
        }
    }
}

// Structs
public struct TranscriptionResult: Codable {
    public var text: String
    public var segments: [TranscriptionSegment]
    public var language: String
    public var timings: TranscriptionTimings?
}

public struct TranscriptionSegment: Hashable, Codable {
    public var id: Int
    public var seek: Int
    public var start: Float
    public var end: Float
    public var text: String
    public var tokens: [Int]
    public var temperature: Float
    public var avgLogprob: Float
    public var compressionRatio: Float
    public var noSpeechProb: Float
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
    public var decodingSampling: TimeInterval
    public var decodingFallback: TimeInterval
    public var decodingWindowing: TimeInterval
    public var decodingKvCaching: TimeInterval
    public var decodingNonPrediction: TimeInterval
    public var totalAudioProcessingRuns: Double
    public var totalLogmelRuns: Double
    public var totalEncodingRuns: Double
    public var totalDecodingLoops: Double
    public var totalKVUpdateRuns: Double
    public var totalDecodingFallbacks: Double
    public var totalDecodingWindows: Double
    public var fullPipeline: TimeInterval

    // Computed properties
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
                decodingSampling: TimeInterval = 0,
                decodingFallback: TimeInterval = 0,
                decodingWindowing: TimeInterval = 0,
                decodingKvCaching: TimeInterval = 0,
                decodingNonPrediction: TimeInterval = 0,
                totalAudioProcessingRuns: Double = 0,
                totalLogmelRuns: Double = 0,
                totalEncodingRuns: Double = 0,
                totalDecodingLoops: Double = 0,
                totalKVUpdateRuns: Double = 0,
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
        self.decodingSampling = decodingSampling
        self.decodingFallback = decodingFallback
        self.decodingWindowing = decodingWindowing
        self.decodingKvCaching = decodingKvCaching
        self.decodingNonPrediction = decodingNonPrediction
        self.totalAudioProcessingRuns = totalAudioProcessingRuns
        self.totalLogmelRuns = totalLogmelRuns
        self.totalEncodingRuns = totalEncodingRuns
        self.totalDecodingLoops = totalDecodingLoops
        self.totalKVUpdateRuns = totalKVUpdateRuns
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
@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
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
@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
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
@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
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
@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
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
@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
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
@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
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

// MARK: Tokenizer

public extension Tokenizer {
    var whitespaceToken: Int { convertTokenToId(" ") ?? Self.defaultWhitespaceToken }
    var specialTokenBegin: Int { convertTokenToId("<|endoftext|>") ?? Self.defaultSpecialTokenBegin }
    var endToken: Int { convertTokenToId("<|endoftext|>") ?? Self.defaultEndToken }
    var startOfTranscriptToken: Int { convertTokenToId("<|startoftranscript|>") ?? Self.defaultStartOfTranscriptToken }
    var englishToken: Int { convertTokenToId("<|en|>") ?? Self.defaultEnglishToken }
    var transcribeToken: Int { convertTokenToId("<|transcribe|>") ?? Self.defaultTranscribeToken }
    var translateToken: Int { convertTokenToId("<|translate|>") ?? Self.defaultTranslateToken }
    var noSpeechToken: Int { convertTokenToId("<|nospeech|>") ?? Self.defaultNoSpeechToken }
    var noTimestampsToken: Int { convertTokenToId("<|notimestamps|>") ?? Self.defaultNoTimestampsToken }
    var timeTokenBegin: Int { convertTokenToId("<|0.00|>") ?? Self.defaultTimeTokenBegin }

    // Default values for each token, using base vocab
    internal static var defaultWhitespaceToken: Int { 50257 }
    internal static var defaultSpecialTokenBegin: Int { 50257 }
    internal static var defaultEndToken: Int { 50257 }
    internal static var defaultStartOfTranscriptToken: Int { 50258 }
    internal static var defaultEnglishToken: Int { 50259 }
    internal static var defaultTranscribeToken: Int { 50359 }
    internal static var defaultTranslateToken: Int { 50358 }
    internal static var defaultNoSpeechToken: Int { 50362 }
    internal static var defaultNoTimestampsToken: Int { 50363 }
    internal static var defaultTimeTokenBegin: Int { 50364 }

    var langauges: [String: String] { [
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
    ] }
}
