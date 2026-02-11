//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
@_exported import ArgmaxCore
import CoreML
import Foundation

// MARK: - Generation progress

/// Per-step progress information delivered to a `TTSCallback` during generation.
///
/// A single callback receives both streaming audio
/// and timing metadata on every generation step
///
/// **Typical usage:**
/// ```swift
/// let result = try await tts.generateSpeech(text: "Hello!") { progress in
///     player.enqueue(progress.audio)                                  // stream audio
///     if let t = progress.stepTime { configureBufferingStrategy(t) }  // first-step setup
///     return nil                                                      // nil or true = continue, false = cancel generation
/// }
/// ```
public struct TTSProgress: Sendable {
    /// Audio samples produced by this generation step (~80ms at 24 kHz).
    public var audio: [Float]

    /// Timings accumulated so far in the current chunk.
    public var timings: TTSTimings

    /// Wall-clock duration of this generation step (seconds).
    ///
    /// Non-`nil` **only on the first step** - use it to configure adaptive playback
    /// buffers (see `TTSPlaybackStrategy.auto`) before the first audio frame is played.
    public var stepTime: TimeInterval?

    public init(audio: [Float], timings: TTSTimings, stepTime: TimeInterval? = nil) {
        self.audio = audio
        self.timings = timings
        self.stepTime = stepTime
    }
}

/// A closure invoked on each generation step with decoded audio and timing info.
///
/// Return `false` to cancel generation early; return `nil` or `true` to continue.
/// Matches the `TranscriptionCallback` pattern in WhisperKit.
public typealias TTSCallback = (@Sendable (TTSProgress) -> Bool?)?

// MARK: - TTSModel Protocol

/// The stable public contract that every TTS model family must satisfy.
///
/// Conform to this protocol to add a new TTS model family to TTSKit.
///
/// **Minimum implementation:**
/// ```swift
/// public class MyTTSModel: TTSModel {
///     public var sampleRate: Int { 24000 }
///
///     public func generate(text: String, voice: String, language: String,
///                          options: TTSGenerationOptions, callback: TTSCallback) async throws -> TTSResult { ... }
///
///     public func playSpeech(text: String, voice: String, language: String,
///                            options: TTSGenerationOptions, playbackStrategy: TTSPlaybackStrategy,
///                            callback: TTSCallback) async throws -> TTSResult { ... }
/// }
/// ```
public protocol TTSModel: AnyObject, Sendable {
    /// Output sample rate in Hz for all audio produced by this model.
    var sampleRate: Int { get }

    /// Generate speech from text and return the complete audio result.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice/speaker identifier. Format is model-specific (e.g. `"ryan"` for Qwen3 TTS).
    ///   - language: Language identifier. Format is model-specific (e.g. `"english"` for Qwen3 TTS).
    ///   - options: Sampling and generation options. Model-specific fields are ignored by models
    ///     that do not support them.
    ///   - callback: Optional per-step callback receiving decoded audio chunks.
    ///               Return `false` to cancel; `nil` or `true` to continue.
    func generate(
        text: String,
        voice: String,
        language: String,
        options: TTSGenerationOptions,
        callback: TTSCallback
    ) async throws -> TTSResult

    /// Generate speech and stream it through the device audio output in real time.
    ///
    /// Default implementation calls `generate` and plays via `TTSAudioOutput`. Override
    /// if the model has its own streaming playback path.
    func playSpeech(
        text: String,
        voice: String,
        language: String,
        options: TTSGenerationOptions,
        playbackStrategy: TTSPlaybackStrategy,
        callback: TTSCallback
    ) async throws -> TTSResult
}

// MARK: - Playback Strategy

/// Controls how `playSpeech()` buffers audio before starting playback.
///
/// The TTS generation loop produces one RVQ frame (~80ms of audio) per step.
/// On fast devices, steps complete well under 80ms and audio can stream immediately.
/// On slower devices (or in debug builds), steps may exceed 80ms, causing the
/// playback buffer to drain and produce choppy audio.
///
/// `.auto` (default) measures the first generation step - which runs the full
/// pipeline (MultiCodeDecoder + SpeechDecoder + CodeDecoder) before any audio
/// is emitted - and uses that exact timing to compute the pre-buffer needed so
/// playback is less likely to underrun. The buffer is re-assessed at the start of each chunk.
public enum TTSPlaybackStrategy: Sendable {
    /// Automatically determine buffer duration from the first step's measured
    /// wall-clock time. On fast devices this resolves to a small minimum (~240ms)
    /// to absorb per-step variance. On slower devices it pre-buffers just enough
    /// to avoid underruns for the full generation. Re-assessed per chunk.
    case auto

    /// Always stream frame-by-frame with no pre-buffer.
    /// May produce choppy audio if the device can't generate faster than real-time.
    case stream

    /// Pre-buffer a fixed duration of audio before starting playback.
    case buffered(seconds: Double)

    /// Generate all audio for each chunk first, then play.
    /// Highest latency but guaranteed smooth playback.
    case generateFirst

    // MARK: - Buffer calculation

    /// Minimum buffer duration (seconds) always applied in `.auto` mode.
    /// 3 frames ≈ 240ms - provides headroom for occasional scheduling jitter
    /// on fast devices where the deficit formula yields zero.
    // TODO: this should be model specific
    public static let minimumBufferDuration: TimeInterval = 0.24

    /// Audio duration produced by a single generation step (seconds).
    ///
    /// - Parameters:
    ///   - samplesPerFrame: PCM samples produced per decode step (model-specific).
    ///   - sampleRate: Output sample rate in Hz (model-specific).
    public static func audioPerStep(samplesPerFrame: Int, sampleRate: Int) -> Double {
        Double(samplesPerFrame) / Double(sampleRate)
    }

    /// Compute the required pre-buffer duration (seconds) given a measured step time.
    ///
    /// Uses the first step's wall-clock time directly. No additional safety margin
    /// is applied because:
    /// - Step 0 typically overestimates
    /// - `maxNewTokens` is an upper bound on actual generation length, adding
    ///   further conservatism.
    ///
    /// - Parameters:
    ///   - stepTime: Measured wall-clock time of one full generation step (seconds).
    ///   - maxNewTokens: Upper bound on remaining generation steps.
    ///   - samplesPerFrame: PCM samples produced per decode step (model-specific).
    ///   - sampleRate: Output sample rate in Hz (model-specific).
    /// - Returns: Buffer duration in seconds (≥ minimumBufferDuration).
    public static func requiredBuffer(
        stepTime: TimeInterval,
        maxNewTokens: Int,
        samplesPerFrame: Int,
        sampleRate: Int
    ) -> TimeInterval {
        let perStep = audioPerStep(samplesPerFrame: samplesPerFrame, sampleRate: sampleRate)
        let speedRatio = perStep / stepTime
        let deficit = max(0.0, 1.0 - speedRatio)
        let maxAudioDuration = Double(maxNewTokens) * perStep
        let deficitBuffer = maxAudioDuration * deficit
        return max(minimumBufferDuration, deficitBuffer)
    }
}

// MARK: - Generation Options

public struct TTSGenerationOptions: Sendable {
    // MARK: - Defaults

    public static let defaultTemperature: Float = 0.9
    public static let defaultTopK: Int = 50
    public static let defaultRepetitionPenalty: Float = 1.05
    public static let defaultMaxNewTokens: Int = 245

    public var temperature: Float
    public var topK: Int
    public var repetitionPenalty: Float
    public var maxNewTokens: Int

    /// Number of concurrent workers for multi-chunk generation.
    /// - `nil`: unlimited - all chunks run concurrently (fastest for non-streaming use cases).
    /// - `1`: sequential - one chunk at a time; required for real-time `playSpeech` streaming.
    /// - `N`: at most N chunks run concurrently.
    public var concurrentWorkerCount: Int?

    /// How to split long text into chunks. Defaults to `.sentence`.
    /// Set to `.none` to force a single-pass generation without sentence splitting.
    public var chunkingStrategy: TTSChunkingStrategy?

    /// Target chunk size in tokens for sentence chunking.
    /// `nil` resolves to `TTSTextChunker.defaultTargetChunkSize` at the call site in TTSKit.
    public var targetChunkSize: Int?

    /// Minimum chunk size in tokens.
    /// `nil` resolves to `TTSTextChunker.defaultMinChunkSize` at the call site in TTSKit.
    public var minChunkSize: Int?

    /// Optional style instruction for controlling speech characteristics
    /// (e.g., "Very happy"). Prepended as a text-only user prompt
    /// before the main TTS segment. For Qwen, this is only supported by the 1.7B model variant.
    public var instruction: String?

    public init(
        temperature: Float = TTSGenerationOptions.defaultTemperature,
        topK: Int = TTSGenerationOptions.defaultTopK,
        repetitionPenalty: Float = TTSGenerationOptions.defaultRepetitionPenalty,
        maxNewTokens: Int = TTSGenerationOptions.defaultMaxNewTokens,
        concurrentWorkerCount: Int? = nil,
        chunkingStrategy: TTSChunkingStrategy? = nil,
        targetChunkSize: Int? = nil,
        minChunkSize: Int? = nil,
        instruction: String? = nil
    ) {
        self.temperature = temperature
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.maxNewTokens = maxNewTokens
        // TODO: enable chunking
        self.concurrentWorkerCount = concurrentWorkerCount ?? 1
        // #if os(macOS)
        // self.concurrentWorkerCount = concurrentWorkerCount ?? 16
        // #else
        // self.concurrentWorkerCount = concurrentWorkerCount ?? 4
        // #endif
        self.chunkingStrategy = chunkingStrategy
        self.targetChunkSize = targetChunkSize
        self.minChunkSize = minChunkSize
        self.instruction = instruction
    }
}

// MARK: - Timings

/// All timing values are stored in seconds (matching WhisperKit's TranscriptionTimings convention).
public struct TTSTimings: Sendable {
    /// Model loading
    public var modelLoading: TimeInterval = 0
    public var tokenizerLoading: TimeInterval = 0

    /// Pipeline phases
    public var tokenize: TimeInterval = 0
    public var prefill: TimeInterval = 0
    public var timeToFirstBuffer: TimeInterval = 0
    public var fullPipeline: TimeInterval = 0

    /// Generation loop total
    public var generationLoop: TimeInterval = 0

    /// CodeDecoder (1 call/step, timed at call site)
    public var codeDecoder: TimeInterval = 0

    /// MultiCodeDecoder
    public var multiCodeDecoder: TimeInterval = 0 // total wall-clock time
    public var multiCodeDecoderPredictions: TimeInterval = 0 // sum of model.prediction() calls
    public var multiCodeDecoderSampling: TimeInterval = 0 // sampleMultiHead calls
    public var multiCodeDecoderEmbedding: TimeInterval = 0 // multi-code embedder lookups
    public var multiCodeDecoderKvCache: TimeInterval = 0 // KV cache update calls
    public var totalMultiCodeDecoderPredictions: Double = 0 // count of prediction calls

    /// SpeechDecoder
    public var speechDecoder: TimeInterval = 0 // total wall-clock time
    public var speechDecoderPredictions: TimeInterval = 0 // model.prediction() only

    /// Non-model operations (outer loop)
    public var kvCacheUpdate: TimeInterval = 0 // outer CodeDecoder cache update
    public var codeEmbed: TimeInterval = 0 // CodeEmbedder lookup
    public var codecHidden: TimeInterval = 0 // MultiCodeEmbedder + sum for next step
    public var textProjection: TimeInterval = 0 // TextProjector + combine embeddings
    public var sampling: TimeInterval = 0 // outer code0 sampling

    /// Counts
    public var prefillTokens: Double = 0
    public var totalDecodingLoops: Double = 0
    public var inputAudioSeconds: TimeInterval = 0

    public init() {}

    /// Total wall-clock time in model.prediction() calls across all three models
    /// Note: with async parallelism, this sum may exceed generationLoop (overlap)
    public var totalPredictions: TimeInterval {
        codeDecoder + multiCodeDecoderPredictions + speechDecoderPredictions
    }

    /// Parallel overlap: how much prediction time ran concurrently
    public var parallelOverlap: TimeInterval {
        max(0, totalPredictions - generationLoop)
    }

    /// Non-inference overhead on the main path (excludes overlapped SpeechDecoder time)
    public var totalNonPrediction: TimeInterval {
        // Main-path predictions: CodeDecoder + MultiCodeDecoder run sequentially
        let mainPathPredictions = codeDecoder + multiCodeDecoderPredictions
        return generationLoop - mainPathPredictions
    }

    public var tokensPerSecond: Double {
        fullPipeline > 0 ? totalDecodingLoops / fullPipeline : 0
    }

    public var realTimeFactor: Double {
        inputAudioSeconds > 0 ? fullPipeline / inputAudioSeconds : 0
    }

    public var speedFactor: Double {
        inputAudioSeconds > 0 ? inputAudioSeconds / fullPipeline : 0
    }
}

// MARK: - Result

public struct TTSResult: Sendable {
    /// Raw float audio samples (mono PCM).
    public let audio: [Float]

    /// Generation performance timings.
    public let timings: TTSTimings

    /// Sample rate of the audio in Hz.
    public let sampleRate: Int

    /// Audio duration in seconds.
    public var audioDuration: Double {
        Double(audio.count) / Double(sampleRate)
    }

    public init(audio: [Float], timings: TTSTimings, sampleRate: Int) {
        self.audio = audio
        self.timings = timings
        self.sampleRate = sampleRate
    }

    /// Log detailed timing breakdown to console, matching WhisperKit's format.
    public func logTimings() {
        let t = timings
        let totalLoops = t.totalDecodingLoops
        let mcdPredRuns = t.totalMultiCodeDecoderPredictions
        let fullPipelineDuration = max(t.generationLoop, t.fullPipeline) * 1000

        Logging.info("""
        ---- TTS Timings ----
        Tokenize:                \(fmt(t.tokenize, 1, fullPipelineDuration))
        Prefill:                 \(fmt(t.prefill, t.prefillTokens, fullPipelineDuration))
        All Predictions:         \(fmt(t.totalPredictions, totalLoops, fullPipelineDuration))
        Non-inference (main):    \(fmt(t.totalNonPrediction, totalLoops, fullPipelineDuration))
        Parallel Overlap:        \(fmt(t.parallelOverlap, totalLoops, fullPipelineDuration))
        CodeDecoder:             \(fmt(t.codeDecoder, totalLoops, fullPipelineDuration))
        MultiCodeDecoder:        \(fmt(t.multiCodeDecoder, totalLoops, fullPipelineDuration))
        - Predictions:           \(fmt(t.multiCodeDecoderPredictions, mcdPredRuns, fullPipelineDuration))
        - Sampling:              \(fmt(t.multiCodeDecoderSampling, totalLoops, fullPipelineDuration))
        - Embedding:             \(fmt(t.multiCodeDecoderEmbedding, totalLoops, fullPipelineDuration))
        - KV Caching:            \(fmt(t.multiCodeDecoderKvCache, totalLoops, fullPipelineDuration))
        SpeechDecoder:           \(fmt(t.speechDecoder, totalLoops, fullPipelineDuration))
        - Predictions:           \(fmt(t.speechDecoderPredictions, totalLoops, fullPipelineDuration))
        KV Cache (outer):        \(fmt(t.kvCacheUpdate, totalLoops, fullPipelineDuration))
        Code Embed (outer):      \(fmt(t.codeEmbed, totalLoops, fullPipelineDuration))
        Codec Hidden:            \(fmt(t.codecHidden, totalLoops, fullPipelineDuration))
        Text Projection:         \(fmt(t.textProjection, totalLoops, fullPipelineDuration))
        Sampling (outer):        \(fmt(t.sampling, totalLoops, fullPipelineDuration))
        Generation Loop:         \(fmt(t.generationLoop, totalLoops, fullPipelineDuration))
        -------------------------------
        Model Load Time:               \(String(format: "%.2f", t.modelLoading)) seconds
        - Tokenizer:                   \(String(format: "%.2f", t.tokenizerLoading)) seconds
        Inference Duration (Global):   \(String(format: "%.2f", t.fullPipeline)) seconds
        Time to first buffer:          \(String(format: "%.2f", t.timeToFirstBuffer)) seconds
        Total Steps:                   \(Int(totalLoops))
        Steps per Second:              \(String(format: "%.2f", t.tokensPerSecond)) steps/s
        Real Time Factor:              \(String(format: "%.3f", t.realTimeFactor))
        Speed Factor:                  \(String(format: "%.3f", t.speedFactor))
        Audio Duration:                \(String(format: "%.2f", audioDuration)) seconds
        """)
    }

    /// Format a timing value matching WhisperKit's `Logging.formatTimeWithPercentage`.
    /// Input `time` is in seconds; `fullPipelineDuration` is in milliseconds.
    // TODO: use shared formatter for timings
    private func fmt(_ time: TimeInterval, _ runs: Double, _ fullPipelineDuration: Double) -> String {
        let timeMs = time * 1000
        let percentage = fullPipelineDuration > 0 ? (timeMs / fullPipelineDuration) * 100 : 0
        let runTime = runs > 0 ? timeMs / runs : 0
        return String(format: "%8.2f ms / %6.0f runs (%8.2f ms/run) %5.2f%%", timeMs, runs, runTime, percentage)
    }
}

// MARK: - Decoder Result Types

/// Result from MultiCodeDecoder.generateMultiCodes
public struct MultiCodeGenerationResult {
    public let codes: [Int32]
    public let timings: TTSTimings
}

/// Result from SpeechDecoder.decodeFrameAsync
public struct SpeechDecoderTimedResult: Sendable {
    public let samples: [Float]
    public let timings: TTSTimings
}

// MARK: - Error

@frozen
public enum TTSError: Error, LocalizedError {
    case emptyText
    case modelNotFound(String)
    case generationFailed(String)
    case tokenizerUnavailable(String)
    case audioOutputFailed(String)
    /// A required component URL could not be resolved from the config.
    /// Thrown at `loadModels()` time so callers fail early with a clear message.
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
            case .emptyText: return "Input text is empty"
            case let .modelNotFound(path): return "Model directory not found: \(path)"
            case let .generationFailed(msg): return "Generation failed: \(msg)"
            case let .tokenizerUnavailable(msg): return "Tokenizer unavailable: \(msg)"
            case let .audioOutputFailed(msg): return "Audio output failed: \(msg)"
            case let .invalidConfiguration(msg): return "Invalid TTSKit configuration: \(msg)"
        }
    }
}

// MARK: - Embedding Buffer

/// Fixed-size buffer for a single embedding vector
public typealias EmbedBuffer = [FloatType]

// MARK: - Embedding Helpers

/// Element-wise addition of two equal-length embedding vectors via vDSP.
public func addEmbeddings(_ a: EmbedBuffer, _ b: EmbedBuffer) -> EmbedBuffer {
    precondition(a.count == b.count)
    var result = [Float](repeating: 0, count: a.count)
    a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            var aFloat = [Float](repeating: 0, count: a.count)
            var bFloat = [Float](repeating: 0, count: a.count)
            convertToFloat(aPtr, to: &aFloat)
            convertToFloat(bPtr, to: &bFloat)
            vDSP_vadd(aFloat, 1, bFloat, 1, &result, 1, vDSP_Length(a.count))
        }
    }
    return result.map { FloatType($0) }
}

/// Accumulates embeddings into a Float32 buffer to avoid repeated type conversions,
/// reusing a single intermediate buffer rather than allocating one per embed.
public func sumEmbeddings(_ embeddings: [EmbedBuffer]) -> EmbedBuffer {
    guard let first = embeddings.first else { return [] }
    let count = first.count
    var accum = [Float](repeating: 0, count: count)
    var floatBuf = [Float](repeating: 0, count: count) // allocated once, reused each iteration
    for embed in embeddings {
        embed.withUnsafeBufferPointer { ptr in
            convertToFloat(ptr, to: &floatBuf)
            vDSP_vadd(accum, 1, floatBuf, 1, &accum, 1, vDSP_Length(count))
        }
    }
    var result = EmbedBuffer(repeating: 0, count: count)
    result.withUnsafeMutableBufferPointer { dst in
        accum.withUnsafeBufferPointer { src in
            // vDSP Float16<->Float conversion is available on iOS 14+, macOS 11+, visionOS 1+
            // but not on watchOS. Fall back to scalar on watchOS and x86_64.
            #if arch(arm64) && !os(watchOS)
            vDSP.convertElements(of: src, to: &dst)
            #else
            for i in 0..<count {
                dst[i] = FloatType(src[i])
            }
            #endif
        }
    }
    return result
}

// TODO: move this section to a utility class
/// Platform-safe conversion from FloatType buffer to Float array.
/// On arm64 iOS/macOS/visionOS (Float16), uses vDSP for vectorized conversion.
@inline(__always)
func convertToFloat(_ source: UnsafeBufferPointer<FloatType>, to dest: inout [Float]) {
    #if arch(arm64) && !os(watchOS)
    vDSP.convertElements(of: source, to: &dest)
    #else
    for i in 0..<min(source.count, dest.count) {
        dest[i] = Float(source[i])
    }
    #endif
}

public func extractEmbed(from arr: MLMultiArray) -> EmbedBuffer {
    let dim: Int
    if arr.shape.count == 4 {
        dim = arr.shape[1].intValue
    } else {
        dim = arr.count
    }
    let ptr = arr.dataPointer.bindMemory(to: FloatType.self, capacity: arr.count)
    var result = EmbedBuffer(repeating: 0, count: dim)

    if arr.shape.count == 4 {
        let stride1 = arr.strides[1].intValue
        if stride1 == 1 {
            // Contiguous layout ([1, D, 1, 1] with unit stride) - direct buffer copy
            result.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: ptr, count: dim)
            }
        } else {
            for d in 0..<dim {
                result[d] = ptr[d * stride1]
            }
        }
    } else {
        for i in 0..<min(dim, arr.count) {
            result[i] = ptr[i]
        }
    }
    return result
}

public func createEmbedMLArray(_ values: EmbedBuffer) throws -> MLMultiArray {
    let dim = values.count
    let arr = try MLMultiArray(shape: [1, NSNumber(value: dim), 1, 1], dataType: .float16)
    let ptr = arr.dataPointer.bindMemory(to: FloatType.self, capacity: dim)
    values.withUnsafeBufferPointer { buf in
        ptr.update(from: buf.baseAddress!, count: dim)
    }
    return arr
}

/// Create a zero-filled embed buffer. Pass `embedDim` from the loaded model's introspected dimension.
/// - Parameter dim: Embedding dimension (default 1024; match the actual model's embed size).
public func zeroEmbed(dim: Int = 1024) -> EmbedBuffer {
    EmbedBuffer(repeating: FloatType(0), count: dim)
}

public func makeInt32Array(_ values: [Int32]) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: [NSNumber(value: values.count)], dataType: .int32)
    let ptr = arr.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
    for (i, v) in values.enumerated() {
        ptr[i] = v
    }
    return arr
}
