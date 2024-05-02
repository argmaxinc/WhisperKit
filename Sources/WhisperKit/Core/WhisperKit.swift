//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import AVFoundation
import CoreML
import Foundation
import Hub
import TensorUtils
import Tokenizers

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class WhisperKit {
    /// Models
    public private(set) var modelVariant: ModelVariant = .tiny
    public private(set) var modelState: ModelState = .unloaded
    public var modelCompute: ModelComputeOptions
    public var tokenizer: WhisperTokenizer?

    /// Protocols
    public var audioProcessor: any AudioProcessing
    public var featureExtractor: any FeatureExtracting
    public var audioEncoder: any AudioEncoding
    public var textDecoder: any TextDecoding
    public var logitsFilters: [any LogitsFiltering]
    public var segmentSeeker: any SegmentSeeking

    /// Shapes
    public static var sampleRate: Int = 16000
    public static var hopLength: Int = 160
    public static var chunkLength: Int = 30 // seconds
    public static var windowSamples: Int = 480_000 // sampleRate * chunkLength
    public static var secondsPerTimeToken = Float(0.02)

    /// Progress
    public private(set) var currentTimings: TranscriptionTimings
    public let progress = Progress()

    /// Configuration
    public var modelFolder: URL?
    public var tokenizerFolder: URL?
    public let useBackgroundDownloadSession: Bool

    public init(
        model: String? = nil,
        downloadBase: URL? = nil,
        modelRepo: String? = nil,
        modelFolder: String? = nil,
        tokenizerFolder: URL? = nil,
        computeOptions: ModelComputeOptions? = nil,
        audioProcessor: (any AudioProcessing)? = nil,
        featureExtractor: (any FeatureExtracting)? = nil,
        audioEncoder: (any AudioEncoding)? = nil,
        textDecoder: (any TextDecoding)? = nil,
        logitsFilters: [any LogitsFiltering]? = nil,
        segmentSeeker: (any SegmentSeeking)? = nil,
        verbose: Bool = true,
        logLevel: Logging.LogLevel = .info,
        prewarm: Bool? = nil,
        load: Bool? = nil,
        download: Bool = true,
        useBackgroundDownloadSession: Bool = false
    ) async throws {
        self.modelCompute = computeOptions ?? ModelComputeOptions()
        self.audioProcessor = audioProcessor ?? AudioProcessor()
        self.featureExtractor = featureExtractor ?? FeatureExtractor()
        self.audioEncoder = audioEncoder ?? AudioEncoder()
        self.textDecoder = textDecoder ?? TextDecoder()
        self.logitsFilters = logitsFilters ?? []
        self.segmentSeeker = segmentSeeker ?? SegmentSeeker()
        self.tokenizerFolder = tokenizerFolder
        self.useBackgroundDownloadSession = useBackgroundDownloadSession
        self.currentTimings = TranscriptionTimings()
        Logging.shared.logLevel = verbose ? logLevel : .none

        try await setupModels(
            model: model,
            downloadBase: downloadBase,
            modelRepo: modelRepo,
            modelFolder: modelFolder,
            download: download
        )

        if let prewarm = prewarm, prewarm {
            Logging.info("Prewarming models...")
            try await prewarmModels()
        }

        // If load is not passed in, load based on whether a modelFolder is passed
        if load ?? (modelFolder != nil) {
            Logging.info("Loading models...")
            try await loadModels()
        }
    }

    // MARK: - Model Loading

    public static func recommendedModels() -> (default: String, disabled: [String]) {
        let deviceName = Self.deviceName()
        Logging.debug("Running on \(deviceName)")

        let defaultModel = modelSupport(for: deviceName).default
        let disabledModels = modelSupport(for: deviceName).disabled
        return (defaultModel, disabledModels)
    }

    public static func deviceName() -> String {
        var utsname = utsname()
        uname(&utsname)
        let deviceName = withUnsafePointer(to: &utsname.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) {
                String(cString: $0)
            }
        }
        return deviceName
    }

    public static func fetchAvailableModels(from repo: String = "argmaxinc/whisperkit-coreml", matching: [String] = ["openai_*", "distil-whisper_*"]) async throws -> [String] {
        let hubApi = HubApi()
        let modelFiles = try await hubApi.getFilenames(from: repo, matching: matching)

        return formatModelFiles(modelFiles)
    }

    public static func formatModelFiles(_ modelFiles: [String]) -> [String] {
        let modelFilters = ModelVariant.allCases.map { "\($0.description)\($0.description.contains("large") ? "" : "/")" } // Include quantized models for large
        let modelVariants = modelFiles.map { $0.components(separatedBy: "/")[0] + "/" }
        let filteredVariants = Set(modelVariants.filter { item in
            let count = modelFilters.reduce(0) { count, filter in
                let isContained = item.contains(filter) ? 1 : 0
                return count + isContained
            }
            return count > 0
        })

        let availableModels = filteredVariants.map { variant -> String in
            variant.trimmingFromEnd(character: "/", upto: 1)
        }

        // Sorting order based on enum
        let sizeOrder = ModelVariant.allCases.map { $0.description }

        let sortedModels = availableModels.sorted { firstModel, secondModel in
            // Extract the base size without any additional qualifiers
            let firstModelBase = sizeOrder.first(where: { firstModel.contains($0) }) ?? ""
            let secondModelBase = sizeOrder.first(where: { secondModel.contains($0) }) ?? ""

            if firstModelBase == secondModelBase {
                // If base sizes are the same, sort alphabetically
                return firstModel < secondModel
            } else {
                // Sort based on the size order
                return sizeOrder.firstIndex(of: firstModelBase) ?? sizeOrder.count
                    < sizeOrder.firstIndex(of: secondModelBase) ?? sizeOrder.count
            }
        }

        return sortedModels
    }

    public static func download(
        variant: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        from repo: String = "argmaxinc/whisperkit-coreml",
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        let hubApi = HubApi(downloadBase: downloadBase, useBackgroundSession: useBackgroundSession)
        let repo = Hub.Repo(id: repo, type: .models)
        let modelSearchPath = "*\(variant.description)/*"
        do {
            Logging.debug("Searching for models matching \"\(modelSearchPath)\" in \(repo)")
            let modelFiles = try await hubApi.getFilenames(from: repo, matching: [modelSearchPath])
            var uniquePaths = Set(modelFiles.map { $0.components(separatedBy: "/").first! })

            var variantPath: String? = nil

            if uniquePaths.count == 1 {
                variantPath = uniquePaths.first
            } else {
                // If the model name search returns more than one unique model folder, then prepend the default "openai" prefix from whisperkittools to disambiguate
                Logging.debug("Multiple models found matching \"\(modelSearchPath)\"")
                let adjustedModelSearchPath = "*openai*\(variant.description)/*"
                Logging.debug("Searching for models matching \"\(adjustedModelSearchPath)\" in \(repo)")
                let adjustedModelFiles = try await hubApi.getFilenames(from: repo, matching: [adjustedModelSearchPath])
                uniquePaths = Set(adjustedModelFiles.map { $0.components(separatedBy: "/").first! })

                if uniquePaths.count == 1 {
                    variantPath = uniquePaths.first
                }
            }

            guard let variantPath else {
                // If there is still ambiguity, throw an error
                throw WhisperError.modelsUnavailable("Multiple models found matching \"\(modelSearchPath)\"")
            }

            Logging.debug("Downloading model \(variantPath)...")
            let modelFolder = try await hubApi.snapshot(from: repo, matching: [modelSearchPath]) { progress in
                Logging.debug(progress)
                if let callback = progressCallback {
                    callback(progress)
                }
            }

            let modelFolderName = modelFolder.appending(path: variantPath)
            return modelFolderName
        } catch {
            Logging.debug(error)
            throw error
        }
    }

    /// Sets up the model folder either from a local path or by downloading from a repository.
    public func setupModels(
        model: String?,
        downloadBase: URL? = nil,
        modelRepo: String?,
        modelFolder: String?,
        download: Bool
    ) async throws {
        // Determine the model variant to use
        let modelVariant = model ?? WhisperKit.recommendedModels().default

        // If a local model folder is provided, use it; otherwise, download the model
        if let folder = modelFolder {
            self.modelFolder = URL(fileURLWithPath: folder)
        } else if download {
            let repo = modelRepo ?? "argmaxinc/whisperkit-coreml"
            do {
                self.modelFolder = try await Self.download(
                    variant: modelVariant,
                    downloadBase: downloadBase,
                    useBackgroundSession: useBackgroundDownloadSession,
                    from: repo
                )
            } catch {
                // Handle errors related to model downloading
                throw WhisperError.modelsUnavailable("""
                Model not found. Please check the model or repo name and try again.
                Error: \(error)
                """)
            }
        }
    }

    public func prewarmModels() async throws {
        try await loadModels(prewarmMode: true)
    }

    public func loadModels(
        prewarmMode: Bool = false
    ) async throws {
        modelState = prewarmMode ? .prewarming : .loading

        let modelLoadStart = CFAbsoluteTimeGetCurrent()

        guard let path = modelFolder else {
            throw WhisperError.modelsUnavailable("Model folder is not set.")
        }

        Logging.debug("Loading models from \(path.path) with prewarmMode: \(prewarmMode)")

        let logmelUrl = path.appending(path: "MelSpectrogram.mlmodelc")
        let encoderUrl = path.appending(path: "AudioEncoder.mlmodelc")
        let decoderUrl = path.appending(path: "TextDecoder.mlmodelc")
        let decoderPrefillUrl = path.appending(path: "TextDecoderContextPrefill.mlmodelc")

        for item in [logmelUrl, encoderUrl, decoderUrl] {
            if !FileManager.default.fileExists(atPath: item.path) {
                throw WhisperError.modelsUnavailable("Model file not found at \(item.path)")
            }
        }

        if var featureExtractor = featureExtractor as? WhisperMLModel {
            Logging.debug("Loading feature extractor")
            try await featureExtractor.loadModel(
                at: logmelUrl,
                computeUnits: modelCompute.melCompute, // hardcoded to use GPU
                prewarmMode: prewarmMode
            )
            Logging.debug("Loaded feature extractor")
        }

        if var audioEncoder = audioEncoder as? WhisperMLModel {
            Logging.debug("Loading audio encoder")
            try await audioEncoder.loadModel(
                at: encoderUrl,
                computeUnits: modelCompute.audioEncoderCompute,
                prewarmMode: prewarmMode
            )
            Logging.debug("Loaded audio encoder")
        }

        if var textDecoder = textDecoder as? WhisperMLModel {
            Logging.debug("Loading text decoder")
            try await textDecoder.loadModel(
                at: decoderUrl,
                computeUnits: modelCompute.textDecoderCompute,
                prewarmMode: prewarmMode
            )
            Logging.debug("Loaded text decoder")
        }

        if FileManager.default.fileExists(atPath: decoderPrefillUrl.path) {
            Logging.debug("Loading text decoder prefill data")
            textDecoder.prefillData = TextDecoderContextPrefill()
            try await textDecoder.prefillData?.loadModel(
                at: decoderPrefillUrl,
                computeUnits: modelCompute.prefillCompute,
                prewarmMode: prewarmMode
            )
            Logging.debug("Loaded text decoder prefill data")
        }

        if prewarmMode {
            modelState = .prewarmed
            currentTimings.modelLoading = CFAbsoluteTimeGetCurrent() - modelLoadStart
            return
        }

        // Check model dimensions to assign appropriate tokenizer
        guard let logitsDim = textDecoder.logitsSize, let encoderDim = audioEncoder.embedSize else {
            throw WhisperError.tokenizerUnavailable()
        }
        textDecoder.isModelMultilingual = isModelMultilingual(logitsDim: logitsDim)
        modelVariant = detectVariant(logitsDim: logitsDim, encoderDim: encoderDim)
        Logging.debug("Loading tokenizer for \(modelVariant)")
        let tokenizer = try await loadTokenizer(
            for: modelVariant,
            tokenizerFolder: tokenizerFolder,
            useBackgroundSession: useBackgroundDownloadSession
        )
        self.tokenizer = tokenizer
        textDecoder.tokenizer = tokenizer
        Logging.debug("Loaded tokenizer")

        modelState = .loaded

        currentTimings.modelLoading = CFAbsoluteTimeGetCurrent() - modelLoadStart

        Logging.info("Loaded models for whisper size: \(modelVariant)")
    }

    public func unloadModels() async {
        modelState = .unloading

        for model in [featureExtractor, audioEncoder, textDecoder] {
            if var model = model as? WhisperMLModel {
                model.unloadModel()
            }
        }

        modelState = .unloaded

        Logging.info("Unloaded all models")
    }

    public func clearState() {
        audioProcessor.stopRecording()
        currentTimings = TranscriptionTimings()
    }

    deinit {
        audioProcessor.stopRecording()
    }

    /// Pass in your own logging callback here
    public func loggingCallback(_ callback: Logging.LoggingCallback?) {
        Logging.shared.loggingCallback = callback
    }

    // MARK: - Transcribe multiple audio files

    public func transcribe(
        audioPaths: [String],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async -> [[TranscriptionResult]?] {
        let transcribeResults: [Result<[TranscriptionResult], Swift.Error>] = await transcribe(
            audioPaths: audioPaths,
            decodeOptions: decodeOptions,
            callback: callback
        )
        var results = [[TranscriptionResult]?]()
        for result in transcribeResults {
            results.append(try? result.get())
        }
        return results
    }

    public func transcribe(
        audioPaths: [String],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async -> [Result<[TranscriptionResult], Swift.Error>] {
        let loadedAudioResult = await AudioProcessor.loadAudio(at: audioPaths)
        let transcribeResults: [Result<[TranscriptionResult], Swift.Error>] = await transcribe(
            audioArrays: loadedAudioResult.compactMap { try? $0.get() },
            decodeOptions: decodeOptions,
            callback: callback
        )
        var result = [Result<[TranscriptionResult], Swift.Error>]()
        var transcribeResultIndex = 0
        for audioResult in loadedAudioResult {
            switch audioResult {
                case .success:
                    result.append(transcribeResults[transcribeResultIndex])
                    transcribeResultIndex += 1
                case let .failure(error):
                    result.append(.failure(error))
            }
        }
        return result
    }

    // MARK: - Transcribe multiple audio samples

    public func transcribe(
        audioArrays: [[Float]],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async -> [[TranscriptionResult]?] {
        let transcribeResults: [Result<[TranscriptionResult], Swift.Error>] = await transcribe(
            audioArrays: audioArrays,
            decodeOptions: decodeOptions,
            callback: callback
        )
        var results = [[TranscriptionResult]?]()
        for result in transcribeResults {
            results.append(try? result.get())
        }
        return results
    }

    public func transcribe(
        audioArrays: [[Float]],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async -> [Result<[TranscriptionResult], Swift.Error>] {
        var result = [Result<[TranscriptionResult], Swift.Error>]()
        let concurrentWorkerCount = decodeOptions?.concurrentWorkerCount ?? 0
        let chunkedAudioArrays = concurrentWorkerCount == 0 ? [audioArrays] : audioArrays.chunked(into: concurrentWorkerCount)
        for chunkedAudioArray in chunkedAudioArrays {
            let partialResult = await withTaskGroup(of: [(index: Int, result: Result<[TranscriptionResult], Swift.Error>)].self) { taskGroup -> [Result<[TranscriptionResult], Swift.Error>] in
                for (index, audioArray) in chunkedAudioArray.enumerated() {
                    taskGroup.addTask {
                        do {
                            let transcribeResult: [TranscriptionResult] = try await self.transcribe(
                                audioArray: audioArray,
                                decodeOptions: decodeOptions,
                                callback: callback
                            )
                            return [(index: index, result: .success(transcribeResult))]
                        } catch {
                            return [(index: index, result: .failure(error))]
                        }
                    }
                }
                var batchResult = [(index: Int, result: Result<[TranscriptionResult], Swift.Error>)]()
                for await result in taskGroup {
                    batchResult.append(contentsOf: result)
                }
                batchResult.sort(by: { $0.index < $1.index })
                return batchResult.map { $0.result }
            }
            result.append(contentsOf: partialResult)
        }
        return result
    }

    // MARK: - Transcribe audio file

    @available(*, deprecated, message: "Subject to removal in a future version. Use `transcribe(audioPath:decodeOptions:callback:) async throws -> [TranscriptionResult]` instead.")
    public func transcribe(
        audioPath: String,
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> TranscriptionResult? {
        let result: [TranscriptionResult] = try await transcribe(audioPath: audioPath, decodeOptions: decodeOptions, callback: callback)
        return result.first
    }

    public func transcribe(
        audioPath: String,
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> [TranscriptionResult] {
        // Process input audio file into audio samples
        let loadAudioStart = Date()
        let audioBuffer = try AudioProcessor.loadAudio(fromPath: audioPath)
        let loadTime = Date().timeIntervalSince(loadAudioStart)

        let convertAudioStart = Date()
        let audioArray = AudioProcessor.convertBufferToArray(buffer: audioBuffer)
        let convertTime = Date().timeIntervalSince(convertAudioStart)

        currentTimings.audioLoading = loadTime + convertTime
        Logging.debug("Audio loading time: \(loadTime), Audio convert time: \(convertTime)")

        // Send converted samples to transcribe
        return try await transcribe(
            audioArray: audioArray,
            decodeOptions: decodeOptions,
            callback: callback
        )
    }

    // MARK: - Transcribe audio samples

    @available(*, deprecated, message: "Subject to removal in a future version. Use `transcribe(audioArray:decodeOptions:callback:) async throws -> [TranscriptionResult]` instead.")
    public func transcribe(
        audioArray: [Float],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> TranscriptionResult? {
        let result: [TranscriptionResult] = try await transcribe(audioArray: audioArray, decodeOptions: decodeOptions, callback: callback)
        return result.first
    }

    public func transcribe(
        audioArray: [Float],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> [TranscriptionResult] {
        if self.modelState != .loaded {
            try await loadModels()
        }

        guard let tokenizer else {
            // Tokenizer required for decoding
            throw WhisperError.tokenizerUnavailable()
        }
        try Task.checkCancellation()
        let transcribeTask = TranscribeTask(
            currentTimings: currentTimings,
            progress: progress,
            audioEncoder: audioEncoder,
            featureExtractor: featureExtractor,
            segmentSeeker: segmentSeeker,
            textDecoder: textDecoder,
            tokenizer: tokenizer
        )
        let transcribeTaskResult = try await transcribeTask.run(
            audioArray: audioArray,
            decodeOptions: decodeOptions,
            callback: callback
        )
        if let decodeOptions, decodeOptions.verbose {
            transcribeTaskResult.logTimings()
        }
        transcribeTaskResult.logSegments()
        return [transcribeTaskResult]
    }
}
