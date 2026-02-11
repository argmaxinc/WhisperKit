//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
import ArgmaxCore
import CoreML
import Foundation

// MARK: - Speech Decoder Protocol

// TODO: Move this and other protocols to generic file, keep this qwen specific
/// Decodes RVQ code frames into audio waveform samples.
public protocol SpeechDecoding: TTSModelLoading {
    var model: MLModel? { get }

    // MARK: - Audio format (model-specific)

    /// Output sample rate in Hz (e.g. 24000 for Qwen3 TTS).
    var sampleRate: Int { get }
    /// Number of PCM samples produced per decoded RVQ frame (e.g. 1920 for Qwen3 TTS).
    var samplesPerFrame: Int { get }

    // MARK: - Cache geometry (read after loadModel)

    var kvCacheEmbedDim: Int { get }
    var kvCacheMaxSequenceLength: Int { get }
    var hiddenDim: Int { get }
    var hiddenContextLen: Int { get }

    // MARK: - Decoding

    /// Synchronous decode of a single RVQ frame (16 codes) into audio samples.
    func decodeFrame(
        codes: [Int32],
        cache: TTSSpeechDecoderCache
    ) throws -> [Float]

    /// Async decode that returns audio samples with wall-clock timing.
    func decodeFrameAsync(
        codes: [Int32],
        cache: TTSSpeechDecoderCache
    ) async throws -> SpeechDecoderTimedResult
}

// MARK: - Implementation

/// RVQ-to-audio waveform decoder backed by a CoreML model.
///
/// Thread safety: mutable state (`model`, dimension properties) is set once during
/// `loadModel()` and read-only thereafter. `MLModel.prediction()` is thread-safe.
public class TTSSpeechDecoder: SpeechDecoding, @unchecked Sendable {
    public var model: MLModel?

    // MARK: - Audio format

    public let sampleRate: Int = Qwen3TTSConstants.sampleRate
    public let samplesPerFrame: Int = Qwen3TTSConstants.samplesPerFrame

    /// Detected from model metadata at load time
    public private(set) var hiddenContextLen: Int = Qwen3TTSConstants.sdHiddenContextLen
    /// KV cache embedding dimension
    public private(set) var kvCacheEmbedDim: Int = Qwen3TTSConstants.sdCacheDim
    /// KV cache max sequence length
    public private(set) var kvCacheMaxSequenceLength: Int = Qwen3TTSConstants.sdMaxSeq
    /// Hidden state dimension
    public private(set) var hiddenDim: Int = Qwen3TTSConstants.sdHiddenDim
    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        // In prewarm mode, compilation is complete - discard to free memory before next model compiles
        guard !prewarmMode else { return }

        self.model = loaded

        // Detect dimensions from model description
        // hidden_context input shape: [1, hiddenDim, 1, contextLen]
        if let dim = modelInputDim(model, named: "hidden_context", position: 1) {
            self.hiddenDim = dim
        }
        if let ctxLen = modelInputDim(model, named: "hidden_context", position: 3) {
            self.hiddenContextLen = ctxLen
        }
        // key_cache input shape: [1, cacheDim, 1, maxSeqLen]
        if let dim = modelInputDim(model, named: "key_cache", position: 1) {
            self.kvCacheEmbedDim = dim
        }
        if let seq = modelInputDim(model, named: "key_cache", position: 3) {
            self.kvCacheMaxSequenceLength = seq
        }
    }

    public func decodeFrame(
        codes: [Int32],
        cache: TTSSpeechDecoderCache
    ) throws -> [Float] {
        guard let model else {
            throw TTSError.generationFailed("SpeechDecoder model not loaded")
        }

        let codesArr = try MLMultiArray(shape: [1, 16, 1], dataType: .int32)
        let codesPtr = codesArr.dataPointer.bindMemory(to: Int32.self, capacity: 16)
        for i in 0..<16 {
            codesPtr[i] = codes[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio_codes": MLFeatureValue(multiArray: codesArr),
            "cache_length": MLFeatureValue(multiArray: cache.makeCacheLengthArray()),
            "key_cache": MLFeatureValue(multiArray: cache.keyCache!),
            "value_cache": MLFeatureValue(multiArray: cache.valueCache!),
            "kv_cache_update_mask": MLFeatureValue(multiArray: cache.kvCacheUpdateMask),
            "key_padding_mask": MLFeatureValue(multiArray: cache.keyPaddingMask),
            "hidden_context": MLFeatureValue(multiArray: cache.hiddenContext),
        ])

        let output = try model.prediction(from: input)
        cache.updateWithHiddenContext(output: output)

        let audioArr = output.featureValue(for: "audio")!.multiArrayValue!
        let sampleCount = audioArr.count
        let audioPtr = audioArr.dataPointer.bindMemory(to: FloatType.self, capacity: sampleCount)
        var samples = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            samples[i] = Float(audioPtr[i])
        }

        return samples
    }

    /// Async variant for concurrent execution via `async let`.
    /// Uses native async prediction (macOS 14+) for cooperative scheduling.
    /// Returns samples and a TTSTimings with speechDecoder fields populated.
    public func decodeFrameAsync(
        codes: [Int32],
        cache: TTSSpeechDecoderCache
    ) async throws -> SpeechDecoderTimedResult {
        guard let model else {
            throw TTSError.generationFailed("SpeechDecoder model not loaded")
        }

        var timings = TTSTimings()

        let codesArr = try MLMultiArray(shape: [1, 16, 1], dataType: .int32)
        let codesPtr = codesArr.dataPointer.bindMemory(to: Int32.self, capacity: 16)
        for i in 0..<16 {
            codesPtr[i] = codes[i]
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio_codes": MLFeatureValue(multiArray: codesArr),
            "cache_length": MLFeatureValue(multiArray: cache.makeCacheLengthArray()),
            "key_cache": MLFeatureValue(multiArray: cache.keyCache!),
            "value_cache": MLFeatureValue(multiArray: cache.valueCache!),
            "kv_cache_update_mask": MLFeatureValue(multiArray: cache.kvCacheUpdateMask),
            "key_padding_mask": MLFeatureValue(multiArray: cache.keyPaddingMask),
            "hidden_context": MLFeatureValue(multiArray: cache.hiddenContext),
        ])

        let t = CFAbsoluteTimeGetCurrent()
        let output = try await model.asyncPrediction(from: input, options: MLPredictionOptions())
        timings.speechDecoderPredictions += CFAbsoluteTimeGetCurrent() - t

        cache.updateWithHiddenContext(output: output)
        let audioArr = output.featureValue(for: "audio")!.multiArrayValue!
        let sampleCount = audioArr.count
        let audioPtr = audioArr.dataPointer.bindMemory(to: FloatType.self, capacity: sampleCount)
        var samples = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            samples[i] = Float(audioPtr[i])
        }

        return SpeechDecoderTimedResult(
            samples: samples,
            timings: timings
        )
    }

    public func unloadModel() {
        model = nil
    }
}
