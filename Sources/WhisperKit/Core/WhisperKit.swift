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
    public private(set) var progress = Progress()

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
        modelCompute = computeOptions ?? ModelComputeOptions()
        self.audioProcessor = audioProcessor ?? AudioProcessor()
        self.featureExtractor = featureExtractor ?? FeatureExtractor()
        self.audioEncoder = audioEncoder ?? AudioEncoder()
        self.textDecoder = textDecoder ?? TextDecoder()
        self.logitsFilters = logitsFilters ?? []
        self.segmentSeeker = segmentSeeker ?? SegmentSeeker()
        self.tokenizerFolder = tokenizerFolder
        self.useBackgroundDownloadSession = useBackgroundDownloadSession
        currentTimings = TranscriptionTimings()
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

        if let featureExtractor = featureExtractor as? WhisperMLModel {
            Logging.debug("Loading feature extractor")
            try await featureExtractor.loadModel(
                at: logmelUrl,
                computeUnits: modelCompute.melCompute, // hardcoded to use GPU
                prewarmMode: prewarmMode
            )
            Logging.debug("Loaded feature extractor")
        }

        if let audioEncoder = audioEncoder as? WhisperMLModel {
            Logging.debug("Loading audio encoder")
            try await audioEncoder.loadModel(
                at: encoderUrl,
                computeUnits: modelCompute.audioEncoderCompute,
                prewarmMode: prewarmMode
            )
            Logging.debug("Loaded audio encoder")
        }

        if let textDecoder = textDecoder as? WhisperMLModel {
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
            if let model = model as? WhisperMLModel {
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

    // MARK: - Detect language

    /// Detects the language of the audio file at the specified path.
    ///
    /// - Parameter audioPath: The file path of the audio file.
    /// - Returns: A tuple containing the detected language and the language log probabilities.
    public func detectLanguage(
        audioPath: String
    ) async throws -> (language: String, langProbs: [String: Float]) {
        let audioBuffer = try AudioProcessor.loadAudio(fromPath: audioPath)
        let audioArray = AudioProcessor.convertBufferToArray(buffer: audioBuffer)
        return try await detectLangauge(audioArray: audioArray)
    }

    /// Detects the language of the audio samples in the provided array.
    ///
    /// - Parameter audioArray: An array of audio samples.
    /// - Returns: A tuple containing the detected language and the language log probabilities.
    public func detectLangauge(
        audioArray: [Float]
    ) async throws -> (language: String, langProbs: [String: Float]) {
        if modelState != .loaded {
            try await loadModels()
        }

        // Ensure the model is multilingual, as language detection is only supported for these models
        guard textDecoder.isModelMultilingual else {
            throw WhisperError.decodingFailed("Language detection not supported for this model")
        }
        
        // Tokenizer required for decoding
        guard let tokenizer else {
            throw WhisperError.tokenizerUnavailable()
        }

        let options = DecodingOptions(verbose: Logging.shared.logLevel != .none)
        let decoderInputs = try textDecoder.prepareDecoderInputs(withPrompt: [tokenizer.specialTokens.startOfTranscriptToken])
        decoderInputs.kvCacheUpdateMask[0] = 1.0
        decoderInputs.decoderKeyPaddingMask[0] = 0.0

        // Detect language using up to the first 30 seconds
        guard let audioSamples = AudioProcessor.padOrTrimAudio(fromArray: audioArray, startAt: 0, toLength: WhisperKit.windowSamples) else {
            throw WhisperError.transcriptionFailed("Audio samples are nil")
        }
        guard let melOutput = try await featureExtractor.logMelSpectrogram(fromAudio: audioSamples) else {
            throw WhisperError.transcriptionFailed("Mel output is nil")
        }
        guard let encoderOutput = try await audioEncoder.encodeFeatures(melOutput) else {
            throw WhisperError.transcriptionFailed("Encoder output is nil")
        }

        let tokenSampler = GreedyTokenSampler(temperature: 0, eotToken: tokenizer.specialTokens.endToken, decodingOptions: options)
        guard let languageDecodingResult: DecodingResult = try? await textDecoder.detectLanguage(
            from: encoderOutput,
            using: decoderInputs,
            sampler: tokenSampler,
            options: options,
            temperature: 0
        ) else {
            throw WhisperError.decodingFailed("Language detection failed")
        }

        return (language: languageDecodingResult.language, langProbs: languageDecodingResult.languageProbs)
    }

    // MARK: - Transcribe multiple audio files

    /// Convenience method to transcribe multiple audio files asynchronously and return the results as an array of optional arrays of `TranscriptionResult`.
    /// - Returns: An array of optional arrays containing `TranscriptionResult`.
    public func transcribe(
        audioPaths: [String],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async -> [[TranscriptionResult]?] {
        let transcribeResults = await transcribeWithResults(
            audioPaths: audioPaths,
            decodeOptions: decodeOptions,
            callback: callback
        )
        let results = transcribeResults.toOptionalArrays()
        return results
    }

    /// Transcribes multiple audio files asynchronously and returns the results as an array of tuples containing the file path and the `Result` object.
    ///
    /// This method processes the provided audio file paths by loading the audio data and then transcribing the audio arrays.
    /// It handles any errors that occur during loading or transcription and ensures that the results are returned in the correct order.
    ///
    /// - Parameters:
    ///   - audioPaths: An array of file paths pointing to the audio files to be transcribed.
    ///   - decodeOptions: Optional decoding options to customize the transcription process.
    ///   - callback: Optional callback to receive updates during the transcription process.
    ///
    /// - Returns: An array of `Result` objects with either a successful transcription result or an error.
    public func transcribeWithResults(
        audioPaths: [String],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async -> [Result<[TranscriptionResult], Swift.Error>] {
        // Start timing the audio loading and conversion process
        let loadAudioStart = Date()

        // Load and extract audio data from the provided file paths
        let loadedAudioResult = await AudioProcessor.loadAudio(at: audioPaths)
        let audioArrays = loadedAudioResult.compactMap { try? $0.get() }

        // Calculate the time taken to load and convert audio
        let loadAndConvertTime = Date().timeIntervalSince(loadAudioStart)
        currentTimings.audioLoading = loadAndConvertTime
        Logging.debug("Total Audio Loading and Converting Time: \(loadAndConvertTime)")

        // Transcribe the loaded audio arrays
        let transcribeResults = await transcribeWithResults(
            audioArrays: audioArrays,
            decodeOptions: decodeOptions,
            callback: callback
        )

        // Initialize the result array to hold final transcription results
        var result = [Result<[TranscriptionResult], Swift.Error>]()
        var transcribeResultIndex = 0

        // Iterate over loadedAudioResult and map each to the corresponding transcription result
        for audioResult in loadedAudioResult {
            switch audioResult {
                case .success:
                    // Append transcription result if audio loading was successful (may still contain failure)
                    result.append(transcribeResults[transcribeResultIndex])
                    transcribeResultIndex += 1
                case let .failure(error):
                    // Append failure result if audio loading failed
                    result.append(.failure(error))
            }
        }

        return result
    }

    // MARK: - Transcribe multiple audio arrays

    /// Convenience method to transcribe multiple audio arrays asynchronously and return the results as an array of optional arrays of `TranscriptionResult`.
    /// - Returns: An array of optional arrays containing `TranscriptionResult`.
    public func transcribe(
        audioArrays: [[Float]],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async -> [[TranscriptionResult]?] {
        let transcribeResults = await transcribeWithResults(
            audioArrays: audioArrays,
            decodeOptions: decodeOptions,
            callback: callback
        )

        return transcribeResults.toOptionalArrays()
    }

    /// Transcribes multiple audio arrays asynchronously and returns the results as an array of `Result` objects.
    ///
    /// This method processes the provided audio arrays by dividing them into batches based on the concurrent worker count
    /// specified in `decodeOptions`, if any. The transcription is performed concurrently on these chunks, and the results
    /// are aggregated and returned in the original order.
    ///
    /// - Parameters:
    ///   - audioArrays: An array of arrays, each containing audio sample data to be transcribed.
    ///   - decodeOptions: Optional decoding options to customize the transcription process.
    ///   - callback: Optional callback to receive updates during the transcription process.
    ///
    /// - Returns: An array of `Result` objects, each containing either a successful transcription result or an error.
    public func transcribeWithResults(
        audioArrays: [[Float]],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async -> [Result<[TranscriptionResult], Swift.Error>] {
        // Create an array of decoding options with the same value for each audio array
        let decodeOptionsArray = Array(repeating: decodeOptions, count: audioArrays.count)
        return await transcribeWithOptions(
            audioArrays: audioArrays,
            decodeOptionsArray: decodeOptionsArray,
            callback: callback
        )
    }

    /// Method to transcribe multiple audio arrays asynchronously with optional associated decoding options and return the results as an array of `Result` objects.
    /// - Parameters:
    ///  - audioArrays: An array of arrays, each containing audio
    ///  - decodeOptionsArray: An array of optional decoding options corresponding to each audio array
    ///  - callback: Optional callback to receive updates during the transcription process.
    ///
    /// - Returns: An array of `Result` objects, each containing either a successful transcription result or an error.
    public func transcribeWithOptions(
        audioArrays: [[Float]],
        decodeOptionsArray: [DecodingOptions?] = [nil],
        callback: TranscriptionCallback = nil
    ) async -> [Result<[TranscriptionResult], Swift.Error>] {
        var result = [Result<[TranscriptionResult], Swift.Error>]()

        guard audioArrays.count == decodeOptionsArray.count else {
            return [.failure(WhisperError.transcriptionFailed("The number of audio arrays and decoding options must be balanced."))]
        }

        // Determine the number of concurrent workers from decodeOptions based on the maximum value or default to 0
        let concurrentWorkerCount = decodeOptionsArray.map { $0?.concurrentWorkerCount ?? 0 }.max() ?? 0

        // Chunk the audio arrays based on the number of concurrent workers
        // If concurrentWorkerCount is 0, all audio arrays are processed in one batch
        let batchedAudioArrays = concurrentWorkerCount == 0 ? [audioArrays] : audioArrays.batched(into: concurrentWorkerCount)

        for (batchIndex, audioArrayBatch) in batchedAudioArrays.enumerated() {
            // Use withTaskGroup to manage concurrent transcription tasks
            let partialResult = await withTaskGroup(of: [(index: Int, result: Result<[TranscriptionResult], Swift.Error>)].self) { taskGroup -> [Result<[TranscriptionResult], Swift.Error>] in
                for (audioIndex, audioArray) in audioArrayBatch.enumerated() {
                    // Setup callback to keep track of batches and chunks
                    let batchedAudioCallback: ((TranscriptionProgress) -> Bool?) = { progress in
                        var batchedProgress = progress
                        batchedProgress.windowId = audioIndex + batchIndex * audioArrayBatch.count
                        return callback?(batchedProgress)
                    }

                    // Setup decoding options for the current audio array
                    let batchedDecodeOptions = decodeOptionsArray[audioIndex]

                    // Add a new task to the task group for each audio array
                    taskGroup.addTask {
                        do {
                            let transcribeResult: [TranscriptionResult] = try await self.transcribe(
                                audioArray: audioArray,
                                decodeOptions: batchedDecodeOptions,
                                callback: batchedAudioCallback
                            )
                            // Return the successful transcription result with its index
                            return [(index: audioIndex, result: .success(transcribeResult))]
                        } catch {
                            // Return the failure result with its index in case of an error
                            return [(index: audioIndex, result: .failure(error))]
                        }
                    }
                }

                // Collect results from all completed tasks in the task group
                var batchResult = [(index: Int, result: Result<[TranscriptionResult], Swift.Error>)]()
                for await result in taskGroup {
                    batchResult.append(contentsOf: result)
                }

                // Sort the results by index to maintain the original order (they may not be in order due to concurrency)
                batchResult.sort(by: { $0.index < $1.index })

                // Map the sorted batch results to a simple array of results
                return batchResult.map { $0.result }
            }

            // Append the results of each batch to the final result array
            result.append(contentsOf: partialResult)
        }

        return result
    }

    // MARK: - Transcribe single audio file

    @available(*, deprecated, message: "Subject to removal in a future version. Use `transcribe(audioPath:decodeOptions:callback:) async throws -> [TranscriptionResult]` instead.")
    @_disfavoredOverload
    public func transcribe(
        audioPath: String,
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> TranscriptionResult? {
        let result: [TranscriptionResult] = try await transcribe(audioPath: audioPath, decodeOptions: decodeOptions, callback: callback)
        return result.first
    }

    /// Transcribes an audio file from the given path asynchronously.
    /// - Parameters:
    ///   - audioPath: The file path to the audio file to be transcribed.
    ///   - decodeOptions: Options for how to transcribe audio. Includes a chunking strategy and the number of concurrent workers to parallelize the task.
    ///   - callback: Optional callback to receive updates during the transcription process.
    /// - Returns: An array of `TranscriptionResult`.
    /// - Throws: An error if the transcription fails.
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

        let transcribeResults: [TranscriptionResult] = try await transcribe(
            audioArray: audioArray,
            decodeOptions: decodeOptions,
            callback: callback
        )

        return transcribeResults
    }

    // MARK: - Transcribe single audio sample array

    /// Deprecated
    @available(*, deprecated, message: "Subject to removal in a future version. Use `transcribe(audioArray:decodeOptions:callback:) async throws -> [TranscriptionResult]` instead.")
    @_disfavoredOverload
    public func transcribe(
        audioArray: [Float],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> TranscriptionResult? {
        let result: [TranscriptionResult] = try await transcribe(audioArray: audioArray, decodeOptions: decodeOptions, callback: callback)
        return result.first
    }

    /// Main entry point for transcribing audio
    /// - Parameters:
    ///   - audioArray: Array of 16khz raw float audio samples
    ///   - decodeOptions: Options for how to transcribe audio. Including a chunking strategy and the number of concurrent workers will paralleize this task.
    ///   - callback: Optional callback to receive updates during the transcription process.
    /// - Returns: An array of sorted `TranscriptionResult`.
    /// - Throws: An error if the transcription fails.
    public func transcribe(
        audioArray: [Float],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> [TranscriptionResult] {
        var transcribeResults = [TranscriptionResult]()

        // Determine if the audio array requires chunking
        let isChunkable = audioArray.count > WhisperKit.windowSamples
        switch (isChunkable, decodeOptions?.chunkingStrategy) {
            case (true, .vad):
                // We have some audio that will require multiple windows and a strategy to chunk them
                let chunker = VADAudioChunker()
                let audioChunks: [AudioChunk] = try await chunker.chunkAll(
                    audioArray: audioArray,
                    maxChunkLength: WhisperKit.windowSamples,
                    decodeOptions: decodeOptions
                )

                // Reset the seek times since we've already chunked the audio
                var chunkedOptions = decodeOptions
                chunkedOptions?.clipTimestamps = []
                let chunkedDecodeOptions = Array(repeating: chunkedOptions, count: audioChunks.count)

                // Send chunked samples to transcribe (note: this is recursive)
                let chunkedResults: [Result<[TranscriptionResult], Swift.Error>] = await transcribeWithOptions(
                    audioArrays: audioChunks.map { $0.audioSamples },
                    decodeOptionsArray: chunkedDecodeOptions,
                    callback: callback
                )

                // Update the seek offsets based on the audio chunks
                let updatedTranscriptionResults = chunker.updateSeekOffsetsForResults(
                    chunkedResults: chunkedResults,
                    audioChunks: audioChunks
                )

                transcribeResults = updatedTranscriptionResults
            default:
                // Audio is short enough to transcribe in a single window and doesn't require chunking
                transcribeResults = try await runTranscribeTask(
                    audioArray: audioArray,
                    decodeOptions: decodeOptions,
                    callback: callback
                )
        }

        if let decodeOptions, decodeOptions.verbose {
            Logging.info("Total Transcription Results: \(transcribeResults.count)")
            for (i, transcribeTaskResult) in transcribeResults.enumerated() {
                Logging.debug("[Result \(i)]")
                transcribeTaskResult.logSegments()
            }
        }

        return transcribeResults
    }

    /// Runs the transcription task on a single audio sample array asynchronously.
    /// - Returns: An array of `TranscriptionResult`.
    /// - Throws: An error if the transcription fails or if the tokenizer is unavailable.
    private func runTranscribeTask(
        audioArray: [Float],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> [TranscriptionResult] {
        if modelState != .loaded {
            try await loadModels()
        }

        guard let tokenizer else {
            // Tokenizer required for decoding
            throw WhisperError.tokenizerUnavailable()
        }

        let childProgress = Progress()
        progress.totalUnitCount += 1
        progress.addChild(childProgress, withPendingUnitCount: 1)

        let transcribeTask = TranscribeTask(
            currentTimings: currentTimings,
            progress: childProgress,
            audioEncoder: audioEncoder,
            featureExtractor: featureExtractor,
            segmentSeeker: segmentSeeker,
            textDecoder: textDecoder,
            tokenizer: tokenizer
        )

        do {
            try Task.checkCancellation()

            let transcribeTaskResult = try await transcribeTask.run(
                audioArray: audioArray,
                decodeOptions: decodeOptions,
                callback: callback
            )

            if let decodeOptions, decodeOptions.verbose {
                transcribeTaskResult.logTimings()
            }

            if progress.isFinished {
                // Reset progress if it is completed
                progress = Progress()
            }

            return [transcribeTaskResult]
        } catch {
            // Handle cancellation
            if error is CancellationError {
                // Reset progress when cancelled
                progress = Progress()
            }
            throw error
        }
    }
}
