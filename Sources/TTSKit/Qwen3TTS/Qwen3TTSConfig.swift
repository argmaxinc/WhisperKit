//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - Model Presets

/// Pre-configured Qwen3 TTS model sizes with matching variant defaults.
///
/// Pass a preset to `TTSKitConfig(model:)` to select the desired model size.
public enum TTSModelPreset: String, CaseIterable, Sendable {
    case qwen3TTS_0_6b = "0.6b"
    case qwen3TTS_1_7b = "1.7b"

    /// Whether this model supports the voice instruction prompt.
    /// Only the 1.7B variant has the capacity to follow style instructions.
    public var supportsVoiceDirection: Bool {
        switch self {
            case .qwen3TTS_0_6b: return false
            case .qwen3TTS_1_7b: return true
        }
    }

    /// Returns `true` when this preset can run on the current platform.
    ///
    /// The 1.7B model requires more peak memory during CoreML compilation than
    /// iOS/iPadOS devices can reliably provide, so it is restricted to macOS.
    public var isAvailableOnCurrentPlatform: Bool {
        #if os(macOS)
        return true
        #else
        switch self {
            case .qwen3TTS_0_6b: return true
            case .qwen3TTS_1_7b: return false
        }
        #endif
    }

    /// The best default preset for the current platform.
    public static var defaultForCurrentPlatform: TTSModelPreset {
        return .qwen3TTS_0_6b
    }

    public var versionDir: String {
        switch self {
            case .qwen3TTS_0_6b: return "12hz-0.6b-customvoice"
            case .qwen3TTS_1_7b: return "12hz-1.7b-customvoice"
        }
    }

    public var displayName: String {
        switch self {
            case .qwen3TTS_0_6b: return "Qwen3 TTS 0.6B"
            case .qwen3TTS_1_7b: return "Qwen3 TTS 1.7B"
        }
    }

    /// Recommended code decoder variant for this model size.
    public var codeDecoderVariant: String { TTSVariantDefaults.codeDecoder }
    /// Recommended multi-code decoder variant for this model size.
    public var multiCodeDecoderVariant: String { TTSVariantDefaults.multiCodeDecoder }
    /// Code embedder variant (same across model sizes).
    public var codeEmbedderVariant: String { TTSVariantDefaults.codeEmbedder }
    /// Multi-code embedder variant (same across model sizes).
    public var multiCodeEmbedderVariant: String { TTSVariantDefaults.multiCodeEmbedder }
    /// Text projector variant (same across model sizes).
    public var textProjectorVariant: String { TTSVariantDefaults.textProjector }
    /// Recommended speech decoder variant for this model size.
    public var speechDecoderVariant: String { TTSVariantDefaults.speechDecoder }
    /// HuggingFace repo used to load the tokenizer for this model size.
    public var tokenizerRepo: String { Qwen3TTSConstants.defaultTokenizerRepo }
}

// MARK: - Variant Defaults

/// Default quantization variant strings matching the standard model repository layout.
public enum TTSVariantDefaults {
    public static let codeDecoder = "W8A16-stateful"
    public static let multiCodeDecoder = "W8A16"
    public static let codeEmbedder = "W16A16"
    public static let multiCodeEmbedder = "W16A16"
    public static let textProjector = "W8A16"
    public static let speechDecoder = "W8A16"
}

// MARK: - TTSKit Configuration

/// Configuration for initializing a `TTSKit` instance with Qwen3 TTS models.
///
/// Models can be loaded from a local directory or automatically downloaded from HuggingFace Hub.
///
/// **Minimal usage** (auto-downloads models and tokenizer):
/// ```swift
/// let config = TTSKitConfig()
/// ```
///
/// **Local models** (skip download):
/// ```swift
/// let config = TTSKitConfig(modelsPath: "/path/to/models")
/// ```
///
/// **Expected local directory layout:**
/// ```
/// modelsPath/
/// └── qwen3_tts/
///     ├── code_decoder/<versionDir>/<variant>/*.mlmodelc
///     ├── multi_code_decoder/<versionDir>/<variant>/*.mlmodelc
///     ├── code_embedder/<versionDir>/<variant>/*.mlmodelc
///     ├── multi_code_embedder/<versionDir>/<variant>/*.mlmodelc
///     ├── text_projector/<versionDir>/<variant>/*.mlmodelc
///     └── speech_decoder/<versionDir>/<variant>/*.mlmodelc
/// ```
public struct TTSKitConfig: @unchecked Sendable {
    /// Model preset that determines default versionDir, variants, and tokenizer.
    public var model: TTSModelPreset

    /// Local path to the model repository directory.
    /// If `nil`, models are downloaded from `modelRepo` on HuggingFace Hub.
    public var modelsPath: String?

    /// HuggingFace repo ID for auto-downloading models.
    public var modelRepo: String

    /// Version directory shared across all components (resolved from `model` by default).
    public var versionDir: String

    /// HuggingFace repo ID or local folder path for the tokenizer
    /// (resolved from `model` by default).
    public var tokenizerSource: String

    /// Per-component quantization variant (resolved from `model` by default).
    public var codeDecoderVariant: String
    public var multiCodeDecoderVariant: String
    public var codeEmbedderVariant: String
    public var multiCodeEmbedderVariant: String
    public var textProjectorVariant: String
    public var speechDecoderVariant: String

    /// Compute unit configuration per model component.
    public var computeOptions: TTSComputeOptions

    /// Whether to emit diagnostic logs during loading and generation.
    public var verbose: Bool

    /// Logging level when `verbose` is `true`. Defaults to `.debug`.
    public var logLevel: Logging.LogLevel

    /// HuggingFace API token for private repos (or set the `HF_TOKEN` env var).
    public var token: String?

    /// Token IDs to suppress during sampling (model-specific).
    /// Defaults to `Qwen3TTSConstants.suppressTokenIds` for Qwen3 TTS.
    public var suppressTokenIds: Set<Int>

    // MARK: - Component overrides

    //
    // Set any of these to substitute a custom implementation for that model component.
    // `nil` means TTSKit will use the default Qwen3 TTS class for that component.
    // Example:
    //   let config = TTSKitConfig()
    //   config.codeDecoder = MyCodeDecoder()
    //   let tts = try await TTSKit(config)

    public var textProjector: (any TextProjecting)?
    public var codeEmbedder: (any CodeEmbedding)?
    public var multiCodeEmbedder: (any MultiCodeEmbedding)?
    public var codeDecoder: (any CodeDecoding)?
    public var multiCodeDecoder: (any MultiCodeDecoding)?
    public var speechDecoder: (any SpeechDecoding)?

    public init(
        model: TTSModelPreset = .qwen3TTS_0_6b,
        modelsPath: String? = nil,
        modelRepo: String = Qwen3TTSConstants.defaultModelRepo,
        versionDir: String? = nil,
        tokenizerSource: String? = nil,
        codeDecoderVariant: String? = nil,
        multiCodeDecoderVariant: String? = nil,
        codeEmbedderVariant: String? = nil,
        multiCodeEmbedderVariant: String? = nil,
        textProjectorVariant: String? = nil,
        speechDecoderVariant: String? = nil,
        computeOptions: TTSComputeOptions = TTSComputeOptions(),
        verbose: Bool = false,
        logLevel: Logging.LogLevel = .debug,
        token: String? = nil,
        suppressTokenIds: Set<Int> = Qwen3TTSConstants.suppressTokenIds
    ) {
        self.model = model
        self.modelsPath = modelsPath
        self.modelRepo = modelRepo
        self.versionDir = versionDir ?? model.versionDir
        self.tokenizerSource = tokenizerSource ?? model.tokenizerRepo
        self.codeDecoderVariant = codeDecoderVariant ?? model.codeDecoderVariant
        self.multiCodeDecoderVariant = multiCodeDecoderVariant ?? model.multiCodeDecoderVariant
        self.codeEmbedderVariant = codeEmbedderVariant ?? model.codeEmbedderVariant
        self.multiCodeEmbedderVariant = multiCodeEmbedderVariant ?? model.multiCodeEmbedderVariant
        self.textProjectorVariant = textProjectorVariant ?? model.textProjectorVariant
        self.speechDecoderVariant = speechDecoderVariant ?? model.speechDecoderVariant
        self.computeOptions = computeOptions
        self.verbose = verbose
        self.logLevel = logLevel
        self.token = token
        self.suppressTokenIds = suppressTokenIds
    }

    // MARK: Path resolution

    /// Resolve the full path to a component's model bundle.
    ///
    /// Requires `modelsPath` to be set (either directly or via download).
    /// Delegates to `detectModelURL(inFolder:)` from ArgmaxCore, which prefers
    /// a compiled `.mlmodelc` bundle and falls back to `.mlpackage`.
    public func modelURL(component: String, variant: String) -> URL? {
        guard let modelsPath else { return nil }

        let variantDir = URL(fileURLWithPath: modelsPath)
            .appending(path: Qwen3TTSConstants.modelFamilyDir)
            .appending(path: component)
            .appending(path: versionDir)
            .appending(path: variant)

        return detectModelURL(inFolder: variantDir)
    }

    /// Glob patterns used to download only the files needed for the configured variants.
    public var downloadPatterns: [String] {
        let components: [(String, String)] = [
            ("text_projector", textProjectorVariant),
            ("code_embedder", codeEmbedderVariant),
            ("multi_code_embedder", multiCodeEmbedderVariant),
            ("code_decoder", codeDecoderVariant),
            ("multi_code_decoder", multiCodeDecoderVariant),
            ("speech_decoder", speechDecoderVariant),
        ]
        return components.map {
            "\(Qwen3TTSConstants.modelFamilyDir)/\($0.0)/\(versionDir)/\($0.1)/**"
        }
    }
}
