//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

// MARK: - Model loading protocol

/// Shared lifecycle contract for all CoreML-backed TTS model components.
///
/// Adding this to a component protocol (e.g. `TextProjecting: TTSModelLoading`) lets
/// `TTSKit` call `loadModel` and `unloadModel` through the protocol type without knowing
/// the concrete class.
// TODO: move section to ArgmaxCore with agnostic naming; generic for any CoreML model
public protocol TTSModelLoading {
    /// Load the CoreML model bundle at `url` using the specified compute units.
    ///
    /// When `prewarmMode` is `true` the model is compiled on-device and then immediately
    /// discarded. This serializes compilation to cap peak memory before the final
    /// concurrent load (see `TTSKit.prewarmModels()`).
    func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool) async throws

    /// Release the loaded model weights from memory.
    func unloadModel()
}

public extension TTSModelLoading {
    /// Convenience overload - loads with `prewarmMode: false`.
    func loadModel(at url: URL, computeUnits: MLComputeUnits) async throws {
        try await loadModel(at: url, computeUnits: computeUnits, prewarmMode: false)
    }
}

// MARK: - Compute Options

/// Per-component CoreML compute unit configuration.
///
/// Used by `TTSKitConfig` to specify which hardware accelerators each model component
/// should use. This struct is model-agnostic; any backend with multiple CoreML components
/// that need different compute targets can adopt it.
public struct TTSComputeOptions: Sendable {
    /// Compute units for embedding lookup models (TextProjector, CodeEmbedder, MultiCodeEmbedder).
    /// Defaults to CPU-only since these are simple table lookups with minimal compute.
    public var embedderComputeUnits: MLComputeUnits

    /// Compute units for CodeDecoder. Defaults to CPU + Neural Engine.
    public var codeDecoderComputeUnits: MLComputeUnits

    /// Compute units for MultiCodeDecoder. Defaults to CPU + Neural Engine.
    public var multiCodeDecoderComputeUnits: MLComputeUnits

    /// Compute units for SpeechDecoder. Defaults to CPU + Neural Engine.
    public var speechDecoderComputeUnits: MLComputeUnits

    public init(
        embedderComputeUnits: MLComputeUnits = .cpuOnly,
        codeDecoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        multiCodeDecoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        speechDecoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) {
        self.embedderComputeUnits = embedderComputeUnits
        self.codeDecoderComputeUnits = codeDecoderComputeUnits
        self.multiCodeDecoderComputeUnits = multiCodeDecoderComputeUnits
        self.speechDecoderComputeUnits = speechDecoderComputeUnits
    }
}
