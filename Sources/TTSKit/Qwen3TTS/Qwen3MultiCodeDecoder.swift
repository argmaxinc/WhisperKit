//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

// MARK: - Multi-Code Decoder Output

public struct MultiCodeDecoderOutput {
    public let allLogits: MLMultiArray // [1, 15, 2048]
    public let keyCacheUpdates: MLMultiArray
    public let valueCacheUpdates: MLMultiArray
}

// MARK: - Multi-Code Decoder Protocol

/// Generates codec-1..15 tokens for a single RVQ frame given hidden states and codec-0 embedding.
public protocol MultiCodeDecoding: TTSModelLoading {
    var model: MLModel? { get }
    var isStateful: Bool { get }

    // MARK: - Cache geometry (read after loadModel)

    var kvCacheEmbedDim: Int { get }
    var kvCacheMaxSequenceLength: Int { get }
    var codecVocabSize: Int { get }

    // MARK: - Decoding

    func decode(inputEmbeds: MLMultiArray, cache: TTSKVCache, state: Any?) throws -> MultiCodeDecoderOutput
    func makeState() -> Any?

    /// Generate codes 1–15 for one RVQ frame given the hidden states from the CodeDecoder
    /// and the embedding of code-0. Returns all 15 codes and their timings.
    func generateMultiCodes(
        hiddenStates: EmbedBuffer,
        code0Embed: EmbedBuffer,
        multiCodeEmbedder: any MultiCodeEmbedding,
        sampler: any TTSTokenSampling,
        options: TTSGenerationOptions
    ) throws -> MultiCodeGenerationResult
}

public extension MultiCodeDecoding {
    #if canImport(CoreML.MLState)
    /// Accepts a pre-composed MLTensor as input embeds, materializing it into an MLMultiArray
    /// before forwarding to the standard decode path.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func decode(inputEmbedsTensor: MLTensor, cache: TTSKVCache, state: Any?) throws -> MultiCodeDecoderOutput {
        let arr = inputEmbedsTensor.asMLMultiArray()
        return try decode(inputEmbeds: arr, cache: cache, state: state)
    }
    #endif
}

// MARK: - Implementation

/// Multi-code RVQ decoder backed by a CoreML model.
///
/// Thread safety: mutable state (`model`, dimension properties) is set once during
/// `loadModel()` and read-only thereafter. `MLModel.prediction()` is thread-safe.
/// Per-frame `MLState` is created via `makeState()` and managed locally within
/// `generateMultiCodes()`, never stored on this shared instance.
public class TTSMultiCodeDecoder: MultiCodeDecoding, @unchecked Sendable {
    public var model: MLModel?

    /// KV cache embedding dimension, detected from model at load time
    public private(set) var kvCacheEmbedDim: Int = Qwen3TTSConstants.mcdCacheDim
    /// KV cache max sequence length, detected from model at load time
    public private(set) var kvCacheMaxSequenceLength: Int = Qwen3TTSConstants.mcdMaxSeq
    /// Codec vocabulary size per head (codes 1-15), detected from model output
    public private(set) var codecVocabSize: Int = Qwen3TTSConstants.codecVocabSize

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        // In prewarm mode, compilation is complete - discard to free memory before next model compiles
        guard !prewarmMode else { return }

        self.model = loaded

        // Detect dimensions from model description
        if let dim = modelOutputDim(model, named: "key_cache_updates", position: 1) {
            self.kvCacheEmbedDim = dim
        }
        if let seq = modelInputDim(model, named: "key_padding_mask", position: 1) {
            self.kvCacheMaxSequenceLength = seq
        }
        // all_logits output shape: [1, 15, codecVocabSize]
        if let vocab = modelOutputDim(model, named: "all_logits", position: 2) {
            self.codecVocabSize = vocab
        }
    }

    public var isStateful: Bool {
        guard let model else { return false }
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            return !model.modelDescription.stateDescriptionsByName.isEmpty
        }
        return false
    }

    /// Create a fresh MLState for a new RVQ frame (stateful models only).
    /// Returns nil for non-stateful models. The caller owns the returned state.
    public func makeState() -> Any? {
        guard isStateful, let model else { return nil }
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            return model.makeState()
        }
        return nil
    }

    public func decode(inputEmbeds: MLMultiArray, cache: TTSKVCache, state: Any? = nil) throws -> MultiCodeDecoderOutput {
        guard let model else {
            throw TTSError.generationFailed("MultiCodeDecoder model not loaded")
        }

        var dict: [String: MLFeatureValue] = try [
            "input_embeds": MLFeatureValue(multiArray: inputEmbeds),
            "cache_length": MLFeatureValue(multiArray: cache.makeCacheLengthArray()),
            "kv_cache_update_mask": MLFeatureValue(multiArray: cache.kvCacheUpdateMask),
            "key_padding_mask": MLFeatureValue(multiArray: cache.keyPaddingMask),
        ]

        // Only pass external KV cache for non-stateful models
        if !isStateful, let keyCache = cache.keyCache, let valueCache = cache.valueCache {
            dict["key_cache"] = MLFeatureValue(multiArray: keyCache)
            dict["value_cache"] = MLFeatureValue(multiArray: valueCache)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: dict)

        let output: MLFeatureProvider
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = state as? MLState {
            // TODO: use direct mltensor input instead of features
            output = try model.prediction(from: input, using: mlState)
        } else {
            output = try model.prediction(from: input)
        }

        let keyCacheUpdates = output.featureValue(for: "key_cache_updates")!.multiArrayValue!
        let valueCacheUpdates = output.featureValue(for: "value_cache_updates")!.multiArrayValue!

        // Stateful cache update
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = state as? MLState, isStateful {
            let position = Int(cache.cacheLength)
            updateStateCache(
                state: mlState,
                keyCacheUpdates: keyCacheUpdates,
                valueCacheUpdates: valueCacheUpdates,
                position: position
            )
        }

        return MultiCodeDecoderOutput(
            allLogits: output.featureValue(for: "all_logits")!.multiArrayValue!, // TODO: dont force unwrap
            keyCacheUpdates: keyCacheUpdates,
            valueCacheUpdates: valueCacheUpdates
        )
    }

    /// Generate codes 1-15 for a single RVQ frame.
    ///
    /// Given hidden states and code0 embedding from the CodeDecoder, runs the MultiCodeDecoder
    /// autoregressively with its own fresh KV cache per frame.
    /// Returns a result struct containing the generated codes and accumulated timing data.
    public func generateMultiCodes(
        hiddenStates: EmbedBuffer,
        code0Embed: EmbedBuffer,
        multiCodeEmbedder: any MultiCodeEmbedding,
        sampler: any TTSTokenSampling,
        options: TTSGenerationOptions
    ) throws -> MultiCodeGenerationResult {
        var timings = TTSTimings()
        let mcdCache = try TTSKVCache(
            cacheDim: kvCacheEmbedDim,
            maxSeqLength: kvCacheMaxSequenceLength,
            isStateful: isStateful
        )

        // Create fresh state for this frame if needed
        let frameState = makeState()

        let embedDim = hiddenStates.count

        // Pre-allocate one embed MLMultiArray and reuse it for the first two (EmbedBuffer) decode calls,
        // to avoid allocation overhead there while keeping the inline pointer-mutation fast path.
        let reuseArr = try MLMultiArray(shape: [1, NSNumber(value: embedDim), 1, 1], dataType: .float16)
        let reusePtr = reuseArr.dataPointer.bindMemory(to: FloatType.self, capacity: embedDim)

        @inline(__always) func fillAndDecode(_ buf: EmbedBuffer) throws -> MultiCodeDecoderOutput {
            buf.withUnsafeBufferPointer { src in
                reusePtr.update(from: src.baseAddress!, count: embedDim)
            }
            return try decode(inputEmbeds: reuseArr, cache: mcdCache, state: frameState)
        }

        // Prefill step 1: hiddenStates from CodeDecoder
        var t = CFAbsoluteTimeGetCurrent()
        var mcdOutput = try fillAndDecode(hiddenStates)
        timings.multiCodeDecoderPredictions += CFAbsoluteTimeGetCurrent() - t
        timings.totalMultiCodeDecoderPredictions += 1

        t = CFAbsoluteTimeGetCurrent()
        mcdCache.update(keyCacheUpdates: mcdOutput.keyCacheUpdates, valueCacheUpdates: mcdOutput.valueCacheUpdates)
        timings.multiCodeDecoderKvCache += CFAbsoluteTimeGetCurrent() - t

        // Prefill step 2: code0 embedding
        t = CFAbsoluteTimeGetCurrent()
        mcdOutput = try fillAndDecode(code0Embed)
        timings.multiCodeDecoderPredictions += CFAbsoluteTimeGetCurrent() - t
        timings.totalMultiCodeDecoderPredictions += 1

        // After prefill: sample code 1 from head 0
        var stepIdx = 0
        t = CFAbsoluteTimeGetCurrent()
        let code1 = sampler.sampleMultiHead(
            allLogits: mcdOutput.allLogits,
            headIndex: stepIdx,
            temperature: options.temperature,
            topK: options.topK
        )
        timings.multiCodeDecoderSampling += CFAbsoluteTimeGetCurrent() - t
        var codes: [Int32] = [code1]

        // Autoregressively generate code 2 through code 15.
        // Use the pre-allocated reuseArr path for the inner embed calls: embed -> reuseArr (in-place copy)
        // avoids per-step allocations and is faster than MLTensor for these small 1024-element tensors.
        for _ in 0..<14 {
            t = CFAbsoluteTimeGetCurrent()
            mcdCache.update(keyCacheUpdates: mcdOutput.keyCacheUpdates, valueCacheUpdates: mcdOutput.valueCacheUpdates)
            timings.multiCodeDecoderKvCache += CFAbsoluteTimeGetCurrent() - t

            // Embed previous code with position-dependent offset, then decode (reusing arr)
            t = CFAbsoluteTimeGetCurrent()
            let offsetId = codes.last! + Int32(codecVocabSize * stepIdx)
            let codeEmbedBuf = try multiCodeEmbedder.embed(tokenId: offsetId)
            timings.multiCodeDecoderEmbedding += CFAbsoluteTimeGetCurrent() - t

            t = CFAbsoluteTimeGetCurrent()
            mcdOutput = try fillAndDecode(codeEmbedBuf)
            timings.multiCodeDecoderPredictions += CFAbsoluteTimeGetCurrent() - t

            timings.totalMultiCodeDecoderPredictions += 1
            stepIdx += 1

            t = CFAbsoluteTimeGetCurrent()
            let code = sampler.sampleMultiHead(
                allLogits: mcdOutput.allLogits,
                headIndex: stepIdx,
                temperature: options.temperature,
                topK: options.topK
            )
            timings.multiCodeDecoderSampling += CFAbsoluteTimeGetCurrent() - t
            codes.append(code)
        }

        return MultiCodeGenerationResult(
            codes: codes, // [code_1, code_2, ..., code_15]
            timings: timings
        )
    }

    public func unloadModel() {
        model = nil
    }
}
