//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// Configuration to initialize WhisperKit
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class WhisperKitConfig {
    /// Name for whisper model to use
    public var model: String?
    /// Base URL for downloading models
    public var downloadBase: URL?
    /// Repository for downloading models
    public var modelRepo: String?
    /// Token for downloading models from repo (if required)
    public var modelToken: String?

    /// Folder to store models
    public var modelFolder: String?
    /// Folder to store tokenizers
    public var tokenizerFolder: URL?

    /// Model compute options, see `ModelComputeOptions`
    public var computeOptions: ModelComputeOptions?
    /// Audio input config to define how to process audio input
    public var audioInputConfig: AudioInputConfig?
    /// Audio processor for the model
    public var audioProcessor: (any AudioProcessing)?
    public var featureExtractor: (any FeatureExtracting)?
    public var audioEncoder: (any AudioEncoding)?
    public var textDecoder: (any TextDecoding)?
    public var logitsFilters: [any LogitsFiltering]?
    public var segmentSeeker: (any SegmentSeeking)?
    public var voiceActivityDetector: VoiceActivityDetector?

    /// Enable extra verbosity for logging
    public var verbose: Bool
    /// Maximum log level
    public var logLevel: Logging.LogLevel

    /// Enable model prewarming
    public var prewarm: Bool?
    /// Load models if available
    public var load: Bool?
    /// Download models if not available
    public var download: Bool
    /// Use background download session
    public var useBackgroundDownloadSession: Bool

    public init(model: String? = nil,
                downloadBase: URL? = nil,
                modelRepo: String? = nil,
                modelToken: String? = nil,
                modelFolder: String? = nil,
                tokenizerFolder: URL? = nil,
                computeOptions: ModelComputeOptions? = nil,
                audioInputConfig: AudioInputConfig? = nil,
                audioProcessor: (any AudioProcessing)? = nil,
                featureExtractor: (any FeatureExtracting)? = nil,
                audioEncoder: (any AudioEncoding)? = nil,
                textDecoder: (any TextDecoding)? = nil,
                logitsFilters: [any LogitsFiltering]? = nil,
                segmentSeeker: (any SegmentSeeking)? = nil,
                voiceActivityDetector: VoiceActivityDetector? = nil,
                verbose: Bool = true,
                logLevel: Logging.LogLevel = .info,
                prewarm: Bool? = nil,
                load: Bool? = nil,
                download: Bool = true,
                useBackgroundDownloadSession: Bool = false)
    {
        self.model = model
        self.downloadBase = downloadBase
        self.modelRepo = modelRepo
        self.modelToken = modelToken
        self.modelFolder = modelFolder
        self.tokenizerFolder = tokenizerFolder
        self.computeOptions = computeOptions
        self.audioInputConfig = audioInputConfig
        self.audioProcessor = audioProcessor
        self.featureExtractor = featureExtractor
        self.audioEncoder = audioEncoder
        self.textDecoder = textDecoder
        self.logitsFilters = logitsFilters
        self.segmentSeeker = segmentSeeker
        self.voiceActivityDetector = voiceActivityDetector
        self.verbose = verbose
        self.logLevel = logLevel
        self.prewarm = prewarm
        self.load = load
        self.download = download
        self.useBackgroundDownloadSession = useBackgroundDownloadSession
    }
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
public struct DecodingOptions: Codable {
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
    public var chunkingStrategy: ChunkingStrategy?

    public init(
        verbose: Bool = false,
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
        concurrentWorkerCount: Int? = nil,
        chunkingStrategy: ChunkingStrategy? = nil
    ) {
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
        // Set platform-specific default worker count if not explicitly provided
        // Non-macOS devices have shown regressions with >4 workers, default to 4 for safety
        #if os(macOS)
        self.concurrentWorkerCount = concurrentWorkerCount ?? 16
        #else
        self.concurrentWorkerCount = concurrentWorkerCount ?? 4
        #endif
        self.chunkingStrategy = chunkingStrategy
    }
}
