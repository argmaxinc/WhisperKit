//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

// MARK: - Code Embedder Protocol

/// Embeds codec-0 tokens into the shared embedding space.
public protocol CodeEmbedding: TTSModelLoading {
    var model: MLModel? { get }
    func embed(tokenId: Int32) throws -> EmbedBuffer
}

public extension CodeEmbedding {
    // Returns the embedding as an MLTensor for use in the MLTensor pipeline (macOS 15+ / iOS 18+).
    // Default implementation wraps the EmbedBuffer result; concrete classes may override
    // with an optimized path that goes MLMultiArray -> MLTensor directly (skipping EmbedBuffer).
    #if canImport(CoreML.MLState)
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func embedTensor(tokenId: Int32) throws -> MLTensor {
        let buf = try embed(tokenId: tokenId)
        return MLTensor(shape: [1, buf.count, 1, 1], scalars: buf, scalarType: FloatType.self)
    }
    #endif
}

// MARK: - Multi-Code Embedder Protocol

/// Embeds codec-1..15 tokens (with position offsets) into the shared embedding space.
public protocol MultiCodeEmbedding: TTSModelLoading {
    var model: MLModel? { get }
    func embed(tokenId: Int32) throws -> EmbedBuffer
}

public extension MultiCodeEmbedding {
    // Returns the embedding as an MLTensor. Default wraps EmbedBuffer; override for zero-intermediate path.
    #if canImport(CoreML.MLState)
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func embedTensor(tokenId: Int32) throws -> MLTensor {
        let buf = try embed(tokenId: tokenId)
        return MLTensor(shape: [1, buf.count, 1, 1], scalars: buf, scalarType: FloatType.self)
    }
    #endif
}

// MARK: - Code Embedder Implementation

/// Codec-0 token embedder backed by a CoreML model.
///
/// Thread safety: `model`, `embedSize`, and `inputProvider` are set once during
/// `loadModel()`. The `MLMultiArray` backing `inputProvider` is mutated in-place
/// before each prediction - safe because each `TTSGenerateTask` owns its own instances.
public class TTSCodeEmbedder: CodeEmbedding, @unchecked Sendable {
    public var model: MLModel?

    /// Embedding dimension, detected from model output shape at load time.
    public private(set) var embedSize: Int = Qwen3TTSConstants.embedDim

    /// Pre-created feature provider wrapping a pre-allocated input array.
    /// `MLDictionaryFeatureProvider` retains the `MLMultiArray` via its `MLFeatureValue`,
    /// so no separate `inputIds` property is needed. The backing array is mutated
    /// in-place via a pointer derived on each call - `assumingMemoryBound` is a no-op cast.
    private var inputProvider: MLDictionaryFeatureProvider?

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        guard !prewarmMode else { return }

        self.model = loaded

        if let dim = modelOutputDim(model, named: "input_embeds", position: 1) {
            self.embedSize = dim
        }

        let ids = try makeInt32Array([0])
        self.inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
        ])
    }

    public func embed(tokenId: Int32) throws -> EmbedBuffer {
        guard let model, let inputProvider,
              let ids = inputProvider.featureValue(for: "input_ids")?.multiArrayValue
        else { throw TTSError.generationFailed("CodeEmbedder model not loaded") }
        ids.dataPointer.assumingMemoryBound(to: Int32.self)[0] = tokenId
        let output = try model.prediction(from: inputProvider)
        return extractEmbed(from: output.featureValue(for: "input_embeds")!.multiArrayValue!)
    }

    #if canImport(CoreML.MLState)
    /// Optimized MLTensor variant: goes MLMultiArray -> MLTensor directly, skipping EmbedBuffer.
    /// One copy (MLShapedArray init) vs two copies in the default path (extractEmbed + embedTensor wrap).
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func embedTensor(tokenId: Int32) throws -> MLTensor {
        guard let model, let inputProvider,
              let ids = inputProvider.featureValue(for: "input_ids")?.multiArrayValue
        else { throw TTSError.generationFailed("CodeEmbedder model not loaded") }
        ids.dataPointer.assumingMemoryBound(to: Int32.self)[0] = tokenId
        let output = try model.prediction(from: inputProvider)
        guard let arr = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.generationFailed("CodeEmbedder missing 'input_embeds' output")
        }
        return MLTensor(MLShapedArray<FloatType>(arr))
    }
    #endif

    public func unloadModel() {
        model = nil
        inputProvider = nil
    }
}

// MARK: - Multi-Code Embedder Implementation

/// Codec-1..15 token embedder backed by a CoreML model.
///
/// Thread safety: see `TTSCodeEmbedder` - same ownership model applies.
public class TTSMultiCodeEmbedder: MultiCodeEmbedding, @unchecked Sendable {
    public var model: MLModel?

    /// Embedding dimension, detected from model output shape at load time.
    public private(set) var embedSize: Int = Qwen3TTSConstants.embedDim

    private var inputProvider: MLDictionaryFeatureProvider?

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        guard !prewarmMode else { return }

        self.model = loaded

        if let dim = modelOutputDim(model, named: "input_embeds", position: 1) {
            self.embedSize = dim
        }

        let ids = try makeInt32Array([0])
        self.inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
        ])
    }

    public func embed(tokenId: Int32) throws -> EmbedBuffer {
        guard let model, let inputProvider,
              let ids = inputProvider.featureValue(for: "input_ids")?.multiArrayValue
        else { throw TTSError.generationFailed("MultiCodeEmbedder model not loaded") }
        ids.dataPointer.assumingMemoryBound(to: Int32.self)[0] = tokenId
        let output = try model.prediction(from: inputProvider)
        return extractEmbed(from: output.featureValue(for: "input_embeds")!.multiArrayValue!)
    }

    #if canImport(CoreML.MLState)
    /// Optimized MLTensor variant: goes MLMultiArray -> MLTensor directly, skipping EmbedBuffer.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func embedTensor(tokenId: Int32) throws -> MLTensor {
        guard let model, let inputProvider,
              let ids = inputProvider.featureValue(for: "input_ids")?.multiArrayValue
        else { throw TTSError.generationFailed("MultiCodeEmbedder model not loaded") }
        ids.dataPointer.assumingMemoryBound(to: Int32.self)[0] = tokenId
        let output = try model.prediction(from: inputProvider)
        guard let arr = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.generationFailed("MultiCodeEmbedder missing 'input_embeds' output")
        }
        return MLTensor(MLShapedArray<FloatType>(arr))
    }
    #endif

    public func unloadModel() {
        model = nil
        inputProvider = nil
    }
}
