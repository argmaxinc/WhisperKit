//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

// MARK: - Text Projector Protocol

/// Projects text token IDs into the shared embedding space.
public protocol TextProjecting: TTSModelLoading {
    var model: MLModel? { get }
    func project(tokenId: Int32) throws -> EmbedBuffer
}

public extension TextProjecting {
    // Returns the projection as an MLTensor for use in the MLTensor pipeline (macOS 15+ / iOS 18+).
    // Default wraps EmbedBuffer; concrete classes may override for a zero-intermediate path.
    #if canImport(CoreML.MLState)
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func projectTensor(tokenId: Int32) throws -> MLTensor {
        let buf = try project(tokenId: tokenId)
        return MLTensor(shape: [1, buf.count, 1, 1], scalars: buf, scalarType: FloatType.self)
    }
    #endif
}

// MARK: - Implementation

/// Text token projector backed by a CoreML model.
///
/// Thread safety: see `TTSCodeEmbedder` - same ownership model applies.
public class TTSTextProjector: TextProjecting, @unchecked Sendable {
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

    public func project(tokenId: Int32) throws -> EmbedBuffer {
        guard let model, let inputProvider,
              let ids = inputProvider.featureValue(for: "input_ids")?.multiArrayValue
        else { throw TTSError.generationFailed("TextProjector model not loaded") }
        ids.dataPointer.assumingMemoryBound(to: Int32.self)[0] = tokenId
        let output = try model.prediction(from: inputProvider)
        return extractEmbed(from: output.featureValue(for: "input_embeds")!.multiArrayValue!)
    }

    #if canImport(CoreML.MLState)
    /// Optimized MLTensor variant: goes MLMultiArray -> MLTensor directly, skipping EmbedBuffer.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func projectTensor(tokenId: Int32) throws -> MLTensor {
        guard let model, let inputProvider,
              let ids = inputProvider.featureValue(for: "input_ids")?.multiArrayValue
        else { throw TTSError.generationFailed("TextProjector model not loaded") }
        ids.dataPointer.assumingMemoryBound(to: Int32.self)[0] = tokenId
        let output = try model.prediction(from: inputProvider)
        guard let arr = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.generationFailed("TextProjector missing 'input_embeds' output")
        }
        return MLTensor(MLShapedArray<FloatType>(arr))
    }
    #endif

    public func unloadModel() {
        model = nil
        inputProvider = nil
    }
}
