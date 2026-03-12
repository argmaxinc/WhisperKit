//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
import Hub

// MARK: - ModelDownloader

/// Downloads models from a HuggingFace repository with per-model resolution.
open class ModelDownloader {
    private let endpoint: String
    private let repoName: String
    private let modelToken: String?
    private let revision: String
    private let useBackgroundSession: Bool

    public init(endpoint: String = "https://huggingface.co",
                repoName: String,
                modelToken: String? = nil,
                revision: String = "main",
                useBackgroundSession: Bool = false) {
        self.endpoint = endpoint
        self.repoName = repoName
        self.modelToken = modelToken
        self.revision = revision
        self.useBackgroundSession = useBackgroundSession
    }

    public func downloadModel(modelInfo: ModelInfo, downloadBase: URL?) async throws -> URL {
        let searchPath = "\(modelInfo.name)/\(modelInfo.version ?? "*")/\(modelInfo.variant ?? "*")/*"

        let hubApi = HubApi(downloadBase: downloadBase, hfToken: modelToken, endpoint: endpoint, useBackgroundSession: useBackgroundSession)
        let repo = Hub.Repo(id: repoName, type: .models)

        Logging.debug("[ModelDownloader] Searching for models matching \"\(searchPath)\" in \(repo)")
        let modelFiles = try await hubApi.getFilenames(from: repo, revision: revision, matching: [searchPath])

        guard !modelFiles.isEmpty else {
            throw ModelDownloaderError.modelUnavailable("No models found matching \"\(searchPath)\" in \(repoName)")
        }

        Logging.debug("[ModelDownloader] Downloading model \(searchPath)...")
        let modelFolder = try await hubApi.snapshot(from: repo, revision: revision, matching: [searchPath])
        return modelFolder
    }

    public func localRepoLocation(downloadBase: URL?) -> URL {
        let hubApi = HubApi(downloadBase: downloadBase, hfToken: modelToken, endpoint: endpoint, useBackgroundSession: useBackgroundSession)
        let repo = Hub.Repo(id: repoName, type: .models)
        return hubApi.localRepoLocation(repo)
    }

    /// Resolves a specific model file using a three-step fallback strategy:
    /// 1. Model folder (if provided)
    /// 2. Local cache
    /// 3. Online download
    open func resolveModel(
        _ modelFileName: String,
        using info: ModelInfo,
        modelFolder: URL? = nil,
        downloadBase: URL? = nil,
        download: Bool = true,
        modelStateCallback: ((ModelState) -> Void)? = nil
    ) async throws -> URL {
        if let modelFolder {
            let folderURL = info.modelURL(baseURL: modelFolder)
            if let url = modelURLIfExists(inFolder: folderURL, named: modelFileName) {
                Logging.debug("[ModelDownloader] Found \(modelFileName) in model folder")
                return url
            }
        }

        let defaultBase = localRepoLocation(downloadBase: downloadBase)
        let cacheFolderURL = info.modelURL(baseURL: defaultBase)
        if let url = modelURLIfExists(inFolder: cacheFolderURL, named: modelFileName) {
            Logging.debug("[ModelDownloader] Found existing \(modelFileName) in cache")
            return url
        }

        guard download else {
            throw ModelDownloaderError.modelUnavailable("No model found for \(modelFileName) (download disabled)")
        }

        modelStateCallback?(.downloading)
        let downloadedBase = try await downloadModel(modelInfo: info, downloadBase: downloadBase)
        let folderURL = info.modelURL(baseURL: downloadedBase)
        if let url = modelURLIfExists(inFolder: folderURL, named: modelFileName) {
            Logging.debug("[ModelDownloader] Downloaded \(modelFileName)")
            modelStateCallback?(.downloaded)
            return url
        }

        throw ModelDownloaderError.modelUnavailable("No model found for \(modelFileName)")
    }

    /// Resolves an entire repository in a single snapshot call.
    ///
    /// Resolution order:
    /// 1. Local Hub cache — if all required pattern directories exist, return the cache root immediately.
    /// 2. Online download — fetches all files matching `patterns` at this downloader’s `revision` in one `HubApi.snapshot()` call.
    ///
    /// - Parameters:
    ///   - patterns: Glob patterns selecting the files to download (e.g. `modelInfo.downloadPattern`).
    ///   - downloadBase: Override for the Hub cache root. Defaults to the Hub default location.
    ///   - download: When `false`, throws if models are not already cached locally.
    ///   - progressCallback: Called with the Hub’s `Progress` each time it updates.
    /// - Returns: The Hub snapshot root URL containing the downloaded files.
    public func resolveRepo(
        patterns: [String],
        downloadBase: URL? = nil,
        download: Bool = true,
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        let localRoot = localRepoLocation(downloadBase: downloadBase)
        if patternsExistLocally(patterns, in: localRoot) {
            Logging.debug("[ModelDownloader] All models found in local cache at \(localRoot.path)")
            return localRoot
        }

        guard download else {
            throw ModelDownloaderError.modelUnavailable(
                "No local models found for repo '\(repoName)' and download is disabled."
            )
        }

        let hubApi = HubApi(downloadBase: downloadBase, hfToken: modelToken, endpoint: endpoint, useBackgroundSession: useBackgroundSession)
        let repo = Hub.Repo(id: repoName, type: .models)

        Logging.info("[ModelDownloader] Downloading \(patterns.count) model(s) from \(repoName)...")
        let snapshotRoot = try await hubApi.snapshot(from: repo, revision: revision, matching: patterns, progressHandler: progressCallback ?? { _ in })
        Logging.info("[ModelDownloader] Download complete: \(snapshotRoot.path)")
        return snapshotRoot
    }

    /// Returns the list of filenames in the repository matching the given glob patterns.
    /// Uses this downloader’s `revision` so the result matches the branch/tag/commit used for downloads.
    /// - Parameters:
    ///   - patterns: Glob patterns to filter files.
    ///   - downloadBase: Override for the Hub cache root. `nil` uses the Hub library default.
    /// - Returns: Matching filenames at the configured revision.
    public func fetchFilenames(matching patterns: [String], downloadBase: URL? = nil) async throws -> [String] {
        let hubApi = HubApi(downloadBase: downloadBase, hfToken: modelToken, endpoint: endpoint, useBackgroundSession: useBackgroundSession)
        let repo = Hub.Repo(id: repoName, type: .models)
        return try await hubApi.getFilenames(from: repo, revision: revision, matching: patterns)
    }

    /// Returns `true` when the deepest concrete directory for every pattern exists in `root` and is non-empty.
    ///
    /// Walks each pattern up to its first wildcard segment (`*` or `**`) and checks that the resulting
    /// directory exists and contains at least one file. Checking only the top-level component is
    /// insufficient when multiple patterns share the same root (e.g. `qwen3_tts/code_decoder/**` and
    /// `qwen3_tts/speech_decoder/**` both start with `qwen3_tts/`): a single partially-downloaded
    /// component would pass the cache check for all patterns and skip re-downloading missing files.
    private func patternsExistLocally(_ patterns: [String], in root: URL) -> Bool {
        patterns.allSatisfy { pattern in
            let concreteComponents = pattern
                .split(separator: "/")
                .prefix(while: { !$0.contains("*") })
                .map(String.init)
            guard !concreteComponents.isEmpty else { return false }
            let dir = concreteComponents.reduce(root) { $0.appendingPathComponent($1) }
            guard FileManager.default.fileExists(atPath: dir.path) else { return false }
            let contents = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
            return !contents.isEmpty
        }
    }

    public func modelURLIfExists(inFolder folder: URL, named name: String) -> URL? {
        let candidate = ModelUtilities.detectModelURL(inFolder: folder, named: name)
        guard FileManager.default.fileExists(atPath: candidate.path) else {
            return nil
        }
        return candidate
    }
}

public enum ModelDownloaderError: Error, LocalizedError {
    case modelUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .modelUnavailable(let msg): return msg
        }
    }
}


// MARK: - ModelInfo

/// Metadata needed to identify and configure a model for download and execution.
public struct ModelInfo: CustomStringConvertible, CustomDebugStringConvertible, Sendable {
    public let version: String?
    public let variant: String?
    public let name: String
    public let computeUnits: MLComputeUnits

    public init(version: String? = nil, variant: String? = nil, name: String, computeUnits: MLComputeUnits) {
        self.version = version
        self.variant = variant
        self.name = name
        self.computeUnits = computeUnits
    }

    /// Compute model path based on model folder, name, version, and variant
    public func modelURL(baseURL: URL) -> URL {
        var result = baseURL.appendingPathComponent(name)
        if let version = version {
            result = result.appendingPathComponent(version)
        }
        if let variant = variant {
            result = result.appendingPathComponent(variant)
        }
        return result
    }

    /// Glob pattern for selecting all files belonging to this model within a HuggingFace repo.
    public var downloadPattern: String {
        "\(name)/\(version ?? "*")/\(variant ?? "*")/*"
    }

    public var description: String {
        [name, version, variant].compactMap { $0 }.joined(separator: "/")
    }

    public var debugDescription: String {
        "ModelInfo(name: \(name), version: \(version ?? "nil"), variant: \(variant ?? "nil"), computeUnit: \(computeUnits.description))"
    }

    /// Finds the base folder by traversing up the URL path until finding a directory with the model name
    public func findBaseFolder(in url: URL) -> URL? {
        var currentURL = url
        while currentURL.pathComponents.count > 1 {
            if currentURL.lastPathComponent == name {
                return currentURL.deletingLastPathComponent()
            }
            currentURL.deleteLastPathComponent()
        }
        return nil
    }
}
