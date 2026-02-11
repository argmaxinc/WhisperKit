//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - Code Decoder Output

public struct CodeDecoderOutput {
    public let logits: MLMultiArray // [1, 1, 3072]
    public let hiddenStates: EmbedBuffer // 1024 values
    public let keyCacheUpdates: MLMultiArray // [1, 28672, 1, 1]
    public let valueCacheUpdates: MLMultiArray
}

// MARK: - Code Decoder Protocol

/// Autoregressive decoder that generates codec-0 tokens from combined embeddings.
public protocol CodeDecoding: TTSModelLoading {
    var model: MLModel? { get }
    var isStateful: Bool { get }
    var kvCacheEmbedDim: Int { get }
    var kvCacheMaxSequenceLength: Int { get }
    var embedSize: Int { get }
    func decode(inputEmbeds: MLMultiArray, cache: TTSKVCache, state: Any?) throws -> CodeDecoderOutput
    func makeState() -> Any?
}

public extension CodeDecoding {
    #if canImport(CoreML.MLState)
    /// Accepts a pre-composed MLTensor as input embeds, materializing it into an MLMultiArray
    /// before forwarding to the standard decode path.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func decode(inputEmbedsTensor: MLTensor, cache: TTSKVCache, state: Any?) throws -> CodeDecoderOutput {
        let arr = inputEmbedsTensor.asMLMultiArray()
        return try decode(inputEmbeds: arr, cache: cache, state: state)
    }
    #endif
}

// MARK: - Implementation

/// Autoregressive codec-0 decoder backed by a CoreML model.
///
/// Thread safety: mutable state (`model`, dimension properties) is set once during
/// `loadModel()` and read-only thereafter. `MLModel.prediction()` is thread-safe.
/// Per-generation `MLState` is created via `makeState()` and owned by each task,
/// never stored on this shared instance.
public class TTSCodeDecoder: CodeDecoding, @unchecked Sendable {
    public var model: MLModel?

    /// KV cache embedding dimension, detected from model at load time
    public private(set) var kvCacheEmbedDim: Int = Qwen3TTSConstants.cdCacheDim
    /// KV cache max sequence length, detected from model at load time
    public private(set) var kvCacheMaxSequenceLength: Int = Qwen3TTSConstants.cdMaxSeq
    /// Input embedding dimension
    public private(set) var embedSize: Int = Qwen3TTSConstants.embedDim

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        // In prewarm mode, compilation is complete - discard to free memory before next model compiles
        guard !prewarmMode else { return }

        self.model = loaded

        // Detect dimensions from model description
        // key_cache_updates output shape: [1, cacheDim, 1, 1]
        if let dim = modelOutputDim(model, named: "key_cache_updates", position: 1) {
            self.kvCacheEmbedDim = dim
        }
        // key_padding_mask input shape: [1, maxSeqLen]
        if let seq = modelInputDim(model, named: "key_padding_mask", position: 1) {
            self.kvCacheMaxSequenceLength = seq
        }
        // input_embeds input shape: [1, embedDim, 1, 1]
        if let dim = modelInputDim(model, named: "input_embeds", position: 1) {
            self.embedSize = dim
        }
    }

    public var isStateful: Bool {
        guard let model else { return false }
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            return !model.modelDescription.stateDescriptionsByName.isEmpty
        }
        return false
    }

    /// Create a fresh MLState for a new generation session (stateful models only).
    /// Returns nil for non-stateful models. The caller owns the returned state.
    public func makeState() -> Any? {
        guard isStateful, let model else { return nil }
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            return model.makeState()
        }
        return nil
    }

    public func decode(inputEmbeds: MLMultiArray, cache: TTSKVCache, state: Any? = nil) throws -> CodeDecoderOutput {
        guard let model else {
            throw TTSError.generationFailed("CodeDecoder model not loaded")
        }

        var dict: [String: MLFeatureValue] = try [
            "input_embeds": MLFeatureValue(multiArray: inputEmbeds),
            "cache_length": MLFeatureValue(multiArray: cache.makeCacheLengthArray()),
            "kv_cache_update_mask": MLFeatureValue(multiArray: cache.kvCacheUpdateMask),
            "key_padding_mask": MLFeatureValue(multiArray: cache.keyPaddingMask),
        ]

        if !isStateful, let keyCache = cache.keyCache, let valueCache = cache.valueCache {
            dict["key_cache"] = MLFeatureValue(multiArray: keyCache)
            dict["value_cache"] = MLFeatureValue(multiArray: valueCache)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: dict)

        let output: MLFeatureProvider
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = state as? MLState {
            output = try model.prediction(from: input, using: mlState)
        } else {
            output = try model.prediction(from: input)
        }

        let keyCacheUpdates = output.featureValue(for: "key_cache_updates")!.multiArrayValue!
        let valueCacheUpdates = output.featureValue(for: "value_cache_updates")!.multiArrayValue!

        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = state as? MLState, isStateful {
            let position = Int(cache.cacheLength)
            updateStateCache(
                state: mlState,
                keyCacheUpdates: keyCacheUpdates,
                valueCacheUpdates: valueCacheUpdates,
                position: position
            )
        }

        return CodeDecoderOutput(
            logits: output.featureValue(for: "logits")!.multiArrayValue!,
            hiddenStates: extractEmbed(from: output.featureValue(for: "hidden_states")!.multiArrayValue!),
            keyCacheUpdates: keyCacheUpdates,
            valueCacheUpdates: valueCacheUpdates
        )
    }

    public func unloadModel() {
        model = nil
    }
}
