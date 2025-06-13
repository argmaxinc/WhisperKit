//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Hub
import Tokenizers

public struct ModelUtilities {

    private init() {}

    // MARK: Public

    @available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
    public static func modelSupport(for deviceName: String, from config: ModelSupportConfig? = nil) -> ModelSupport {
        let config = config ?? Constants.fallbackModelSupportConfig
        let modelSupport = config.modelSupport(for: deviceName)
        return modelSupport
    }

    public static func loadTokenizer(
        for pretrained: ModelVariant,
        tokenizerFolder: URL? = nil,
        useBackgroundSession: Bool = false
    ) async throws -> WhisperTokenizer {
        let tokenizerName = tokenizerNameForVariant(pretrained)
        let hubApi = HubApi(downloadBase: tokenizerFolder, useBackgroundSession: useBackgroundSession)

        // Attempt to load tokenizer from local folder if specified
        let resolvedTokenizerFolder = hubApi.localRepoLocation(HubApi.Repo(id: tokenizerName))
        let tokenizerConfigPath = resolvedTokenizerFolder.appendingPathComponent("tokenizer.json")

        // Check if 'tokenizer.json' exists in the folder
        if FileManager.default.fileExists(atPath: tokenizerConfigPath.path) {
            do {
                let localConfig = LanguageModelConfigurationFromHub(modelFolder: resolvedTokenizerFolder, hubApi: hubApi)
                if let tokenizerConfig = try await localConfig.tokenizerConfig {
                    let tokenizerData = try await localConfig.tokenizerData
                    let whisperTokenizer = try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
                    Logging.debug("Loading tokenizer from local folder")
                    return WhisperTokenizerWrapper(tokenizer: whisperTokenizer)
                } else {
                    // tokenizerConfig is nil, fall through to load from Hub
                    Logging.debug("Tokenizer configuration not found in local config")
                }
            } catch {
                // Error during the local loading process and fall through to load from Hub
                Logging.debug("Error loading local tokenizer: \(error)")
            }
        }

        // Fallback to loading from the Hub if local loading is not possible or fails
        Logging.debug("Loading tokenizer from Hub")
        return try await WhisperTokenizerWrapper(
            tokenizer: AutoTokenizer.from(
                pretrained: tokenizerName,
                hubApi: hubApi
            )
        )
    }

    public static func detectModelURL(inFolder path: URL, named modelName: String) -> URL {
        let compiledUrl = path.appending(path: "\(modelName).mlmodelc")
        let packageUrl = path.appending(path: "\(modelName).mlpackage/Data/com.apple.CoreML/model.mlmodel")

        let compiledModelExists: Bool = FileManager.default.fileExists(atPath: compiledUrl.path)
        let packageModelExists: Bool = FileManager.default.fileExists(atPath: packageUrl.path)

        // Swap to mlpackage only if the following is true: we found the mlmodel within the mlpackage, and we did not find a .mlmodelc
        var modelURL = compiledUrl
        if packageModelExists && !compiledModelExists {
            modelURL = packageUrl
        }

        return modelURL
    }

    // MARK: Internal

    static func isModelMultilingual(logitsDim: Int?) -> Bool {
        logitsDim != 51864
    }

    static func detectVariant(logitsDim: Int, encoderDim: Int) -> ModelVariant {
        // Defaults
        var modelVariant: ModelVariant = .base

        // Determine model size
        if logitsDim == 51865 {
            // Muiltilingual
            switch encoderDim {
                case 384:
                    modelVariant = .tiny
                case 512:
                    modelVariant = .base
                case 768:
                    modelVariant = .small
                case 1024:
                    modelVariant = .medium
                case 1280:
                    modelVariant = .largev2 // same for v1
                default:
                    modelVariant = .base
            }
        } else if logitsDim == 51864 {
            // English only
            switch encoderDim {
                case 384:
                    modelVariant = .tinyEn
                case 512:
                    modelVariant = .baseEn
                case 768:
                    modelVariant = .smallEn
                case 1024:
                    modelVariant = .mediumEn
                default:
                    modelVariant = .baseEn
            }

        } else if logitsDim == 51866 {
            // Large v3 has 1 additional language token
            modelVariant = .largev3
        } else {
            Logging.error("Unrecognized vocabulary size: \(logitsDim), defaulting to base variant")
            modelVariant = .base
        }

        return modelVariant
    }

    static func getModelInputDimention(_ model: MLModel?, named: String, position: Int) -> Int? {
        guard let inputDescription = model?.modelDescription.inputDescriptionsByName[named] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[position]
    }

    static func getModelOutputDimention(_ model: MLModel?, named: String, position: Int) -> Int? {
        guard let inputDescription = model?.modelDescription.outputDescriptionsByName[named] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[position]
    }

    func getModelInputDimention(_ model: MLModel?, named: String, position: Int) -> Int? {
        guard let inputDescription = model?.modelDescription.inputDescriptionsByName[named] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[position]
    }

    func getModelOutputDimention(_ model: MLModel?, named: String, position: Int) -> Int? {
        guard let inputDescription = model?.modelDescription.outputDescriptionsByName[named] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[position]
    }

    // MARK: Private

    private static func tokenizerNameForVariant(_ variant: ModelVariant) -> String {
        var tokenizerName: String
        switch variant {
            case .tiny:
                tokenizerName = "openai/whisper-tiny"
            case .tinyEn:
                tokenizerName = "openai/whisper-tiny.en"
            case .base:
                tokenizerName = "openai/whisper-base"
            case .baseEn:
                tokenizerName = "openai/whisper-base.en"
            case .small:
                tokenizerName = "openai/whisper-small"
            case .smallEn:
                tokenizerName = "openai/whisper-small.en"
            case .medium:
                tokenizerName = "openai/whisper-medium"
            case .mediumEn:
                tokenizerName = "openai/whisper-medium.en"
            case .large:
                tokenizerName = "openai/whisper-large"
            case .largev2:
                tokenizerName = "openai/whisper-large-v2"
            case .largev3:
                tokenizerName = "openai/whisper-large-v3"
        }

        return tokenizerName
    }
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `ModelUtilities.loadTokenizer(for:pretrained:tokenizerFolder:useBackgroundSession:)` instead.")
public func loadTokenizer(
    for pretrained: ModelVariant,
    tokenizerFolder: URL? = nil,
    useBackgroundSession: Bool = false
) async throws -> WhisperTokenizer {
    return try await ModelUtilities.loadTokenizer(for: pretrained, tokenizerFolder: tokenizerFolder, useBackgroundSession: useBackgroundSession)
}

@available(*, deprecated, message: "Subject to removal in a future version. Use ModelUtilities.modelSupport(for:from:) -> ModelSupport instead.")
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public func modelSupport(for deviceName: String, from config: ModelSupportConfig? = nil) -> ModelSupport {
    return ModelUtilities.modelSupport(for: deviceName, from: config)
}

@available(*, deprecated, message: "Subject to removal in a future version. Use ModelUtilities.modelSupport(for:from:) -> ModelSupport instead.")
@_disfavoredOverload
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public func modelSupport(for deviceName: String, from config: ModelSupportConfig? = nil) -> (default: String, disabled: [String]) {
    let modelSupport = ModelUtilities.modelSupport(for: deviceName, from: config)
    return (modelSupport.default, modelSupport.disabled)
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `ModelUtilities.detectModelURL(inFolder:named:)` instead.")
public func detectModelURL(inFolder path: URL, named modelName: String) -> URL {
    return ModelUtilities.detectModelURL(inFolder: path, named: modelName)
}
