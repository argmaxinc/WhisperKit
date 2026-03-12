//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import CoreML
import Hub
import ArgmaxCore

/// Manages downloading and loading of SpeakerKit (Pyannote) models from HuggingFace.
///
/// Typical usage:
/// ```swift
/// let manager = SpeakerKitModelManager(config: PyannoteConfig())
/// try await manager.downloadModels()
/// try await manager.loadModels()
/// guard let models = manager.models else { return }
/// let speakerKit = SpeakerKit(models: models, config: manager.config)
/// ```
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public final class SpeakerKitModelManager {
    /// Current lifecycle state of the managed models.
    public private(set) var modelState: ModelState = .unloaded {
        didSet {
            guard oldValue != modelState else { return }
            modelStateCallback?(oldValue, modelState)
        }
    }

    /// Called whenever `modelState` transitions to a new value.
    public var modelStateCallback: ((ModelState, ModelState) -> Void)?
    /// Pyannote configuration used for model resolution and inference.
    public let config: PyannoteConfig
    /// Model metadata for the speaker segmenter.
    public let segmenterModelInfo: ModelInfo
    /// Model metadata for the speaker embedder.
    public let embedderModelInfo: ModelInfo
    /// Model metadata for the PLDA projector used in VBx clustering.
    public let pldaModelInfo: ModelInfo
    /// Local path where models were resolved (set after a successful `downloadModels()` or `loadModels()` call).
    public private(set) var modelPath: URL?
    /// Loaded model container, available after a successful `loadModels()` call.
    public private(set) var models: (any SpeakerKitModels)?

    /// Creates a model manager with the given Pyannote configuration.
    ///
    /// - Parameters:
    ///   - config: Pyannote runtime configuration (repo, token, download flags, etc.).
    ///   - segmenterModelInfo: Override for the segmenter model variant. Defaults to `.segmenter()`.
    ///   - embedderModelInfo: Override for the embedder model variant. Defaults to `.embedder()`.
    ///   - pldaModelInfo: Override for the PLDA projector variant. Defaults to `.plda()`.
    public init(
        config: PyannoteConfig,
        segmenterModelInfo: ModelInfo = .segmenter(),
        embedderModelInfo: ModelInfo = .embedder(),
        pldaModelInfo: ModelInfo = .plda()
    ) {
        self.config = config
        self.segmenterModelInfo = segmenterModelInfo
        self.embedderModelInfo = embedderModelInfo
        self.pldaModelInfo = pldaModelInfo
    }

    /// Downloads all required models from HuggingFace into the local cache.
    ///
    /// Skips download if `modelState` is not `.unloaded`. After a successful call,
    /// `modelState` transitions to `.downloaded` and `modelPath` is set.
    ///
    /// - Parameter progressCallback: Called with the Hub’s `Progress` as files are downloaded.
    /// - Throws: `SpeakerKitError` or network errors on failure.
    public func downloadModels(progressCallback: ((Progress) -> Void)? = nil) async throws {
        guard modelState == .unloaded else {
            Logging.debug("[SpeakerKit] Models already downloaded (state: \(modelState)), skipping download")
            return
        }

        if let modelFolder = config.modelFolder {
            Logging.debug("[SpeakerKit] modelFolder is set, skipping download")
            modelPath = modelFolder
            modelState = .downloaded
            return
        }

        Logging.info("[SpeakerKit] Starting download process...")
        modelState = .downloading

        do {
            let path = try await resolveModels(progressCallback: progressCallback)
            modelPath = path
            modelState = .downloaded
            Logging.info("[SpeakerKit] Models downloaded to \(path.path)")
        } catch {
            modelState = .unloaded
            Logging.error("[SpeakerKit] Failed to download models: \(error)")
            throw error
        }
    }

    /// Loads previously downloaded models into memory for inference.
    ///
    /// Requires either a prior `downloadModels()` call or `config.modelFolder` pointing
    /// to a local directory. After a successful call, `modelState` is `.loaded` and
    /// `models` is populated. Segmenter and embedder are loaded concurrently.
    ///
    /// - Throws: `SpeakerKitError` if state or paths are invalid, or model loading fails.
    public func loadModels() async throws {
        let path: URL
        if let modelPath = modelPath {
            path = modelPath
        } else if let modelFolder = config.modelFolder {
            path = modelFolder
            modelPath = modelFolder
        } else {
            throw SpeakerKitError.invalidConfiguration("Must download models before loading. Call downloadModels() first, or provide modelFolder for local models.")
        }

        guard modelState == .downloaded || modelState == .unloaded else {
            if modelState == .loaded {
                Logging.debug("[SpeakerKit] Models already loaded, skipping load")
                return
            }
            throw SpeakerKitError.invalidConfiguration("Invalid state for loading models: \(modelState)")
        }

        modelState = .loading

        do {
            let loadedModels = try await loadPyannoteModels(from: path)
            models = loadedModels
            modelState = .loaded
            Logging.info("[SpeakerKit] Models loaded successfully")
        } catch {
            modelState = .unloaded
            Logging.error("[SpeakerKit] Failed to load models: \(error)")
            throw error
        }
    }

    /// Releases loaded models from memory. No-op if models are not currently loaded.
    public func unloadModels() {
        guard modelState == .loaded else { return }
        modelState = .unloading
        models = nil
        modelState = .unloaded
    }

    /// `true` when models have been downloaded or are in the process of loading/being loaded.
    public var isAvailable: Bool {
        modelState == .downloaded || modelState == .loading || modelState == .loaded
    }

    /// `true` only when models are fully loaded and ready for inference.
    public var isLoaded: Bool {
        modelState == .loaded
    }

    // MARK: - Private

    private func resolveModels(progressCallback: ((Progress) -> Void)? = nil) async throws -> URL {
        if let modelFolder = config.modelFolder {
            Logging.info("[SpeakerKit] Using local models from: \(modelFolder.path)")
            return modelFolder
        }

        let repo = config.modelRepo ?? "argmaxinc/speakerkit-coreml"
        let downloader = ModelDownloader(
            repoName: repo,
            modelToken: config.modelToken,
            useBackgroundSession: config.useBackgroundDownloadSession
        )

        let patterns = [
            segmenterModelInfo.downloadPattern,
            embedderModelInfo.downloadPattern,
            pldaModelInfo.downloadPattern,
        ]

        return try await downloader.resolveRepo(
            patterns: patterns,
            downloadBase: config.downloadBase,
            download: config.download,
            progressCallback: progressCallback
        )
    }

    private func loadPyannoteModels(from modelPath: URL) async throws -> PyannoteModels {
        let segmenterVersionDir = segmenterModelInfo.modelURL(baseURL: modelPath)
        let embedderVersionDir = embedderModelInfo.modelURL(baseURL: modelPath)
        let pldaVersionDir = pldaModelInfo.modelURL(baseURL: modelPath)

        let segmenterURL = ModelUtilities.detectModelURL(inFolder: segmenterVersionDir, named: "SpeakerSegmenter")
        let embedderPreprocessorURL = ModelUtilities.detectModelURL(inFolder: embedderVersionDir, named: "SpeakerEmbedderPreprocessor")
        let embedderURL = ModelUtilities.detectModelURL(inFolder: embedderVersionDir, named: "SpeakerEmbedder")
        let pldaURL = ModelUtilities.detectModelURL(inFolder: pldaVersionDir, named: "PldaProjector")
        guard FileManager.default.fileExists(atPath: pldaURL.path) else {
            throw SpeakerKitError.modelUnavailable("PldaProjector model not found at \(pldaURL.path).")
        }
        let pldaModelURL: URL? = pldaURL

        let segmenterModel = try await SpeakerSegmenterModel(
            modelURL: segmenterURL,
            concurrentWorkers: config.concurrentSegmenterWorkers,
            verbose: config.verbose,
            useFullRedundancy: config.fullRedundancy,
            computeUnits: segmenterModelInfo.computeUnits
        )

        let embedderModel = SpeakerEmbedderModel(
            modelURL: embedderURL,
            preprocessorModelURL: embedderPreprocessorURL,
            pldaModelURL: pldaModelURL,
            computeUnits: embedderModelInfo.computeUnits
        )

        async let segmenterLoad: Void = segmenterModel.loadModel()
        async let embedderLoad: Void = embedderModel.loadModel()
        _ = try await (segmenterLoad, embedderLoad)

        return PyannoteModels(
            segmenter: segmenterModel,
            embedder: embedderModel,
            config: config,
            segmenterModelInfo: segmenterModelInfo,
            embedderModelInfo: embedderModelInfo,
            pldaModelInfo: pldaModelInfo
        )
    }
}

/// Marker protocol for diarizer model containers.
///
/// Conform to this protocol when implementing a custom model container.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public protocol SpeakerKitModels: Sendable {
    /// Metadata describing the model files that make up this bundle.
    var modelInfos: [ModelInfo] { get }
}

/// Loaded Pyannote model bundle — segmenter, embedder, PLDA projector, and associated configuration.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public struct PyannoteModels: SpeakerKitModels {
    public let segmenter: SpeakerSegmenterModel
    public let embedder: SpeakerEmbedderModel
    public let config: PyannoteConfig
    /// Metadata for the model variants that were actually loaded.
    public let segmenterModelInfo: ModelInfo
    public let embedderModelInfo: ModelInfo
    public let pldaModelInfo: ModelInfo

    public var modelInfos: [ModelInfo] { [segmenterModelInfo, embedderModelInfo, pldaModelInfo] }

    init(
        segmenter: SpeakerSegmenterModel,
        embedder: SpeakerEmbedderModel,
        config: PyannoteConfig,
        segmenterModelInfo: ModelInfo = .segmenter(),
        embedderModelInfo: ModelInfo = .embedder(),
        pldaModelInfo: ModelInfo = .plda()
    ) {
        self.segmenter = segmenter
        self.embedder = embedder
        self.config = config
        self.segmenterModelInfo = segmenterModelInfo
        self.embedderModelInfo = embedderModelInfo
        self.pldaModelInfo = pldaModelInfo
    }
}
