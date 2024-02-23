//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import AVFoundation
import CoreML
import Foundation
import Hub
import TensorUtils
import Tokenizers

public protocol Transcriber {
    func transcribe(audioPath: String, decodeOptions: DecodingOptions?, callback: TranscriptionCallback) async throws -> TranscriptionResult?
    func transcribe(audioArray: [Float], decodeOptions: DecodingOptions?, callback: TranscriptionCallback) async throws -> TranscriptionResult?
}

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
public class WhisperKit: Transcriber {
    // Models
    public var modelVariant: ModelVariant = .tiny
    public var modelState: ModelState = .unloaded
    public var modelCompute: ModelComputeOptions
    public var modelFolder: URL?
    public var tokenizer: Tokenizer?

    // Protocols
    public var audioProcessor: any AudioProcessing
    public var featureExtractor: any FeatureExtracting
    public var audioEncoder: any AudioEncoding
    public var textDecoder: any TextDecoding
    public var logitsFilters: [any LogitsFiltering]
    public var segmentSeeker: any SegmentSeeking

    // Shapes
    public static var maxTokenContext = Int(448 / 2)
    public static var sampleRate: Int = 16000
    public static var hopLength: Int = 160
    public static var chunkLength: Int = 30 // seconds
    public static var windowSamples: Int = 480_000 // sampleRate * chunkLength
    public static var secondsPerTimeToken = Float(0.02)

    // Features
    public var audioSamples: MLMultiArray?
    public var melOutput: MLMultiArray?
    public var encoderOutput: MLMultiArray?
    public var decoderInputs: DecodingInputs?
    public var currentTimings: TranscriptionTimings?

    public init(
        model: String? = nil,
        modelRepo: String? = nil,
        modelFolder: String? = nil,
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
        download: Bool = true
    ) async throws {
        self.modelCompute = computeOptions ?? ModelComputeOptions()
        self.audioProcessor = audioProcessor ?? AudioProcessor()
        self.featureExtractor = featureExtractor ?? FeatureExtractor()
        self.audioEncoder = audioEncoder ?? AudioEncoder()
        self.textDecoder = textDecoder ?? TextDecoder()
        self.logitsFilters = logitsFilters ?? []
        self.segmentSeeker = segmentSeeker ?? SegmentSeeker()
        Logging.shared.logLevel = verbose ? logLevel : .none
        currentTimings = TranscriptionTimings()

        try await setupModels(model: model, modelRepo: modelRepo, modelFolder: modelFolder, download: download)

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

    public static func fetchAvailableModels(from repo: String = "argmaxinc/whisperkit-coreml") async throws -> [String] {
        let hubApi = HubApi()
        // TODO: get config from the source repo
        _ = try await hubApi.httpGet(for: URL(string: "https://huggingface.co/argmaxinc/whisperkit-coreml/blob/main/config.json")!)
        let modelFiles = try await hubApi.getFilenames(from: repo, matching: ["openai_whisper*"])

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
            let parts = variant.split(separator: "_")
            let modelInfo = parts[1].split(separator: "-").dropFirst().joined(separator: "-")
            let additionalInfo = parts.count > 2 ? "_\(parts[2...].joined(separator: "_"))" : ""
            return (modelInfo + additionalInfo).trimmingFromEnd(character: "/", upto: 1)
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

    public static func download(variant: String, downloadBase: URL? = nil, from repo: String = "argmaxinc/whisperkit-coreml", progressCallback: ((Progress) -> Void)? = nil) async throws -> URL? {
        let hubApi = HubApi(downloadBase: downloadBase)
        let repo = Hub.Repo(id: repo, type: .models)
        do {
            let modelFolder = try await hubApi.snapshot(from: repo, matching: ["*\(variant.description)/*"]) { progress in
                Logging.debug(progress)
                progressCallback?(progress)
            }

            let modelFolderName = modelFolder.appending(path: "openai_whisper-\(variant)")
            return modelFolderName
        } catch {
            Logging.debug(error)
        }

        return nil
    }

    /// Sets up the model folder either from a local path or by downloading from a repository.
    public func setupModels(model: String?, modelRepo: String?, modelFolder: String?, download: Bool) async throws {
        // Determine the model variant to use
        let modelVariant = model ?? WhisperKit.recommendedModels().default

        // If a local model folder is provided, use it; otherwise, download the model
        if let folder = modelFolder {
            self.modelFolder = URL(fileURLWithPath: folder)
        } else if download {
            let repo = modelRepo ?? "argmaxinc/whisperkit-coreml"
            do {
                let hubModelFolder = try await Self.download(variant: modelVariant, from: repo)
                self.modelFolder = hubModelFolder!
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

    public func loadModels(prewarmMode: Bool = false) async throws {
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

        try [logmelUrl, encoderUrl, decoderUrl].forEach {
            if !FileManager.default.fileExists(atPath: $0.path) {
                throw WhisperError.modelsUnavailable("Model file not found at \($0.path)")
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
            currentTimings?.modelLoading = CFAbsoluteTimeGetCurrent() - modelLoadStart
            return
        }

        // Check model dimensions to assign appropriate tokenizer
        if let logitsDim = textDecoder.logitsSize,
           let encoderDim = audioEncoder.embedSize
        {
            modelVariant = detectVariant(logitsDim: logitsDim, encoderDim: encoderDim)
            Logging.debug("Loading tokenizer for \(modelVariant)")
            tokenizer = try await loadTokenizer(for: modelVariant)
            textDecoder.tokenizer = tokenizer
            Logging.debug("Loaded tokenizer")
        } else {
            Logging.error("Could not load tokenizer")
        }

        modelState = .loaded

        currentTimings?.modelLoading = CFAbsoluteTimeGetCurrent() - modelLoadStart

        Logging.info("Loaded models for whisper size: \(modelVariant)")
    }

    public func unloadModels() async {
        modelState = .unloading

        [featureExtractor, audioEncoder, textDecoder].forEach { model in
            if var model = model as? WhisperMLModel {
                model.unloadModel()
            }
        }

        modelState = .unloaded

        Logging.info("Unloaded all models")
    }

    public func clearState() {
        audioProcessor.stopRecording()
        currentTimings = nil
    }

    deinit {
        clearState()
    }

    /// Pass in your own logging callback here
    public func loggingCallback(_ callback: Logging.LoggingCallback?) {
        Logging.shared.loggingCallback = callback
    }

    // MARK: - Transcribe audio file

    public func transcribe(audioPath: String,
                           decodeOptions: DecodingOptions? = nil,
                           callback: TranscriptionCallback = nil) async throws -> TranscriptionResult?
    {
        if currentTimings == nil {
            currentTimings = TranscriptionTimings()
        }

        // Process input audio file into audio samples
        let loadAudioStart = Date()
        guard let audioBuffer = AudioProcessor.loadAudio(fromPath: audioPath) else {
            return TranscriptionResult(text: "", segments: [], language: "")
        }
        let loadTime = Date().timeIntervalSince(loadAudioStart)

        let convertAudioStart = Date()
        let audioArray = AudioProcessor.convertBufferToArray(buffer: audioBuffer)
        let convertTime = Date().timeIntervalSince(convertAudioStart)

        currentTimings?.audioLoading = loadTime + convertTime
        Logging.debug("Audio loading time: \(loadTime)")
        Logging.debug("Audio convert time: \(convertTime)")

        // Send converted samples to transcribe
        return try await transcribe(
            audioArray: audioArray,
            decodeOptions: decodeOptions,
            callback: callback
        )
    }

    // MARK: - Transcribe audio samples

    public func transcribe(audioArray: [Float],
                           decodeOptions: DecodingOptions? = nil,
                           callback: TranscriptionCallback = nil) async throws -> TranscriptionResult?
    {
        if currentTimings == nil {
            currentTimings = TranscriptionTimings()
        }

        if self.modelState != .loaded {
            try await loadModels()
        }

        var timings = currentTimings!
        timings.pipelineStart = CFAbsoluteTimeGetCurrent()

        var options = decodeOptions ?? DecodingOptions()
        options.verbose = Logging.shared.logLevel != .none

        let contentFrames = audioArray.count
        timings.inputAudioSeconds = Double(Int(contentFrames) / WhisperKit.sampleRate) - Double(decodeOptions?.clipTimestamps.first ?? 0)

        // MARK: Init decoder inputs

        // These accumulate across windows
        var allSegments: [TranscriptionSegment] = []
        var allTokens: [Int] = []
        var transcription = ""

        guard let tokenizer = tokenizer else {
            // Tokenizer required for decoding
            throw WhisperError.tokenizerUnavailable()
        }

        let startDecoderInit = CFAbsoluteTimeGetCurrent()
        decoderInputs = textDecoder.prepareDecoderInputs(withPrompt: [tokenizer.startOfTranscriptToken])
        guard var decoderInputs = decoderInputs else {
            throw WhisperError.prefillFailed("Unable to prepare decoder inputs")
        }
        let decoderInitTime = CFAbsoluteTimeGetCurrent() - startDecoderInit
        timings.decodingInit = decoderInitTime
        Logging.debug("Decoder init time: \(decoderInitTime)")

        // MARK: - Prefill KV Cache

        let prefillStartTime = CFAbsoluteTimeGetCurrent()
        var prefilledCacheSize = 0
        if options.usePrefillPrompt {
            guard let prefilledInputs = try? await textDecoder.prefillDecoderInputs(decoderInputs, withOptions: options, multilingual: modelVariant.isMultilingual) else {
                throw WhisperError.prefillFailed()
            }
            decoderInputs = prefilledInputs
            prefilledCacheSize = decoderInputs.cacheLength[0].intValue
        }
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStartTime
        timings.prefill = prefillTime

        // Add intial prompt to history
        let currentTokens = decoderInputs.initialPrompt

        // Setup masks based on prefill values
        prefilledCacheSize += 1 // Add 1 for initial masked cache update
        decoderInputs.kvCacheUpdateMask[prefilledCacheSize - 1] = 1.0
        for i in 0..<prefilledCacheSize {
            decoderInputs.decoderKeyPaddingMask[i] = 0.0
        }

        allTokens.append(contentsOf: currentTokens)

        Logging.debug("Prefill time: \(prefillTime)")
        Logging.debug("Prefill prompt: \(currentTokens.map { tokenizer.convertIdToToken($0) ?? "" })")

        // MARK: - Main decoder loop

        var fallbackCount: Double = 0

        // Process seek clips
        var seekPoints: [Int] = options.clipTimestamps.map { Int(round($0 * Float(WhisperKit.sampleRate))) }
        if seekPoints.count == 0 {
            seekPoints.append(0)
        }
        if seekPoints.count % 2 == 1 {
            seekPoints.append(contentFrames)
        }

        var seekClips: [(start: Int, end: Int)] = []
        for i in stride(from: 0, to: seekPoints.count, by: 2) {
            let start = seekPoints[i]
            let end = i + 1 < seekPoints.count ? seekPoints[i + 1] : contentFrames
            seekClips.append((start, end))
        }

        let startDecodeLoopTime = CFAbsoluteTimeGetCurrent()

        for (seekClipStart, seekClipEnd) in seekClips {
            // Loop through the current clip until we reach the end
            // Typically this will be the full audio file, unless seek points are explicitly provided
            var seek: Int = seekClipStart

            let windowPadding = 16000 // prevent hallucinations at the end of the clip by stopping up to 1.0s early
            while seek < seekClipEnd - windowPadding {
                // calculate new encoder segment features
                Logging.debug("Decoding Seek: \(seek)")
                let timeOffset = Float(seek) / Float(WhisperKit.sampleRate)
                let segmentSize = min(WhisperKit.windowSamples, contentFrames - seek, seekClipEnd - seek)
                let timeOffsetEnd = Float(seek + segmentSize) / Float(WhisperKit.sampleRate)

                let audioProcessingStart = Date()
                guard let audioSamples = AudioProcessor.padOrTrimAudio(fromArray: audioArray, startAt: seek, toLength: WhisperKit.windowSamples) else {
                    Logging.error("Audio samples are nil")
                    return nil
                }
                let processTime = Date().timeIntervalSince(audioProcessingStart)
                timings.audioProcessing += processTime
                timings.totalAudioProcessingRuns += 1

                let melStart = Date()
                guard let melOutput = try? await featureExtractor.logMelSpectrogram(fromAudio: audioSamples) else {
                    Logging.error("Mel output is nil")
                    return nil
                }
                let melTime = Date().timeIntervalSince(melStart)
                timings.logmels += melTime
                timings.totalLogmelRuns += 1

                let encoderStart = Date()
                guard let encoderOutput = try await audioEncoder.encodeFeatures(melOutput) else {
                    Logging.error("Encoder output is nil")
                    return nil
                }
                let encoderTime = Date().timeIntervalSince(encoderStart)
                timings.encoding += encoderTime
                timings.totalEncodingRuns += 1

                // All features are computed, send to decoder
                Logging.info("Decoding \(timeOffset)s - \(timeOffsetEnd)s")
                if timeOffset + 1 > timeOffsetEnd {
                    print("broken")
                }
                guard let decodingResult = try? await decodeWithFallback(encoderSegment: encoderOutput, decodingOptions: options, callback: callback) else {
                    Logging.error("Unable to decode text")
                    return nil
                }

                // MARK: Windowing

                // at this point we have a completed window aka segment
                let windowingStart = Date()

                let (newSeek, currentSegments) = segmentSeeker.findSeekPointAndSegments(
                    decodingResult: decodingResult,
                    options: options,
                    allSegmentsCount: allSegments.count,
                    currentSeek: seek,
                    segmentSize: segmentSize,
                    sampleRate: WhisperKit.sampleRate,
                    timeToken: tokenizer.timeTokenBegin,
                    specialToken: tokenizer.specialTokenBegin,
                    tokenizer: tokenizer
                )

                // Update seek point without moving backward backward
                seek = max(seek, newSeek)

                guard var currentSegments = currentSegments else {
                    // No current segment found, skip to next window
                    continue
                }

                if options.verbose {
                    let lines = formatSegments(currentSegments)
                    for line in lines {
                        Logging.debug(line)
                    }
                }

                // Clear invalid segments
                // remove any segments that have very close start and end times
                // or that have no text
                for i in 0..<currentSegments.count {
                    if currentSegments[i].start == currentSegments[i].end ||
                        currentSegments[i].text.trimmingCharacters(in: .whitespacesAndNewlines) == "" ||
                        // TODO: make this more robust or a decoding option, 1s hallucinations are anecdotally common when forcing prefill tokens
                        (currentSegments[i].end - currentSegments[i].start) <= 1.0
                    {
                        currentSegments[i].text = tokenizer.convertIdToToken(tokenizer.noSpeechToken) ?? ""
                        currentSegments[i].tokens = [tokenizer.noSpeechToken]
                    }
                }

                // add them to the `allSegments` list
                allSegments.append(contentsOf: currentSegments)
                let allCurrentTokens = currentSegments.flatMap { $0.tokens }
                allTokens.append(contentsOf: allCurrentTokens)

                timings.decodingWindowing += Date().timeIntervalSince(windowingStart)
                timings.totalDecodingWindows += 1

                // Reset cache and move on to the next window
                resetDecoderInputs()
            }
        }

        func decodeWithFallback(
            encoderSegment encoderOutput: MLMultiArray,
            decodingOptions options: DecodingOptions,
            callback: TranscriptionCallback = nil
        ) async throws -> DecodingResult? {
            // Fallback `options.temperatureFallbackCount` times with increasing temperatures, starting at `options.temperature`
            let temperatures = (0...options.temperatureFallbackCount).map { FloatType(options.temperature) + FloatType($0) * FloatType(options.temperatureIncrementOnFallback) }

            Logging.debug("Decoding with tempeartures \(temperatures)")

            var decodingResult: DecodingResult?

            for (i, temp) in temperatures.enumerated() {
                Logging.info("Decoding Temperature: \(temp)")
                let decodeWithFallbackStart = Date()

                let tokenSampler = GreedyTokenSampler(temperature: temp, eotToken: tokenizer.endToken, decodingOptions: options)

                decodingResult = try await textDecoder.decodeText(
                    from: encoderOutput,
                    using: decoderInputs,
                    sampler: tokenSampler,
                    options: options,
                    callback: callback
                ).first

                // Update timings from the decoder main loop
                if let decodingTimings = decodingResult?.timings {
                    if timings.firstTokenTime == 0 {
                        timings.firstTokenTime = decodingTimings.firstTokenTime
                    }
                    timings.decodingPredictions += decodingTimings.decodingPredictions
                    timings.totalDecodingLoops += decodingTimings.totalDecodingLoops
                    timings.decodingNonPrediction += decodingTimings.decodingNonPrediction
                    timings.decodingSampling += decodingTimings.decodingSampling
                    timings.decodingKvCaching += decodingTimings.decodingKvCaching
                    timings.totalKVUpdateRuns += decodingTimings.totalKVUpdateRuns
                }

                // MARK: Fallback checks

                var needsFallback = false
                var fallbackReason = ""
                if let result = decodingResult {
                    if let threshold = options.compressionRatioThreshold,
                       result.compressionRatio > threshold
                    {
                        needsFallback = true // too repetitive
                        fallbackReason = "compressionRatioThreshold"
                    }

                    if let threshold = options.logProbThreshold,
                       result.avgLogProb < threshold
                    {
                        needsFallback = true // average log probablity too low (model is not confident enough)
                        fallbackReason = "logProbThreshold"
                    }

                    if let threshold = options.noSpeechThreshold,
                       result.noSpeechProb > threshold
                    {
                        needsFallback = false // silence
                    }
                }

                if !needsFallback {
                    break
                } else {
                    // Reset decoder inputs for fallback
                    fallbackCount = Double(i)
                    timings.decodingFallback += Date().timeIntervalSince(decodeWithFallbackStart)
                    timings.totalDecodingFallbacks = fallbackCount
                    resetDecoderInputs()
                    Logging.info("Fallback #\(fallbackCount + 1) (\(fallbackReason))")
                }
            }

            return decodingResult
        }

        func resetDecoderInputs() {
            // NOTE: Because we have a mask on the kvcache,
            // we can simply shift the masks without touching the data,
            // it will be overwritten by the new data without impact on the output
            decoderInputs.cacheLength[0] = NSNumber(value: prefilledCacheSize - 1)

            // Store token history and
            // Reset masks to prepare for next window
            for i in 0..<WhisperKit.maxTokenContext {
                if i <= prefilledCacheSize - 1 {
                    // Inside overlap window
                    decoderInputs.decoderKeyPaddingMask[i] = 0
                    decoderInputs.kvCacheUpdateMask[i - 1] = 0
                    decoderInputs.kvCacheUpdateMask[i] = 1
                } else {
                    // Padding
                    decoderInputs.decoderKeyPaddingMask[i] = -10000
                    decoderInputs.kvCacheUpdateMask[i] = 0
                }
            }
        }

        // MARK: Timings and logging

        let decodeLoopTime = CFAbsoluteTimeGetCurrent() - startDecodeLoopTime
        let pipelineTime = CFAbsoluteTimeGetCurrent() - timings.pipelineStart

        timings.decodingLoop = decodeLoopTime
        timings.fullPipeline = pipelineTime

        if options.verbose {
            let totalTokens = allTokens.count
            let totalLoops = timings.totalDecodingLoops
            let timeToFirstToken = timings.firstTokenTime - timings.pipelineStart
            let tokensPerSecond = timings.tokensPerSecond
            let rtf = timings.realTimeFactor

            let fullPipelineDuration = timings.fullPipeline * 1000 // Convert to milliseconds

            let audioLoadTime = formatTimeWithPercentage(timings.audioLoading, 1, fullPipelineDuration)
            let audioProcTime = formatTimeWithPercentage(timings.audioProcessing, timings.totalAudioProcessingRuns, fullPipelineDuration)
            let logmelsTime = formatTimeWithPercentage(timings.logmels, timings.totalLogmelRuns, fullPipelineDuration)
            let encodingTime = formatTimeWithPercentage(timings.encoding, timings.totalEncodingRuns, fullPipelineDuration)
            let decodingInitTime = formatTimeWithPercentage(timings.decodingInit, 1, fullPipelineDuration)
            let prefillInfo = formatTimeWithPercentage(timings.prefill, 1, fullPipelineDuration)
            let predictionsInfo = formatTimeWithPercentage(timings.decodingPredictions, totalLoops, fullPipelineDuration)
            let samplingInfo = formatTimeWithPercentage(timings.decodingSampling, totalLoops, fullPipelineDuration)
            let kvCachingInfo = formatTimeWithPercentage(timings.decodingKvCaching, timings.totalKVUpdateRuns, fullPipelineDuration)
            let nonPredTimeInfo = formatTimeWithPercentage(timings.decodingNonPrediction, totalLoops, fullPipelineDuration)
            let windowingInfo = formatTimeWithPercentage(timings.decodingWindowing, timings.totalDecodingWindows, fullPipelineDuration)
            let fallbackInfo = formatTimeWithPercentage(timings.decodingFallback, timings.totalDecodingFallbacks, fullPipelineDuration)
            let decodingLoopInfo = formatTimeWithPercentage(timings.decodingLoop, totalLoops, fullPipelineDuration)

            // Logging
            Logging.info("---- Transcription Timings ----")

            Logging.info("Audio Load:          \(audioLoadTime)")
            Logging.info("Audio Processing:    \(audioProcTime)")
            Logging.info("Mels:                \(logmelsTime)")
            Logging.info("Encoding:            \(encodingTime)")
            Logging.info("Matrices Init:       \(decodingInitTime)")
            Logging.info("Prefill:             \(prefillInfo)")
            Logging.info("Decoding:            \(predictionsInfo)")
            Logging.info("Non-inference:       \(nonPredTimeInfo)")
            Logging.info("- Sampling:          \(samplingInfo)")
            Logging.info("- Kv Caching:        \(kvCachingInfo)")
            Logging.info("- Windowing:         \(windowingInfo)")
            Logging.info("Fallbacks:           \(fallbackInfo)")
            Logging.info("Decoding Full Loop:  \(decodingLoopInfo)")
            Logging.info("-------------------------------")

            // Summary statistics
            Logging.info("Model Load Time:     \(String(format: "%.2f", timings.modelLoading)) seconds")
            Logging.info("Inference Duration:  \(String(format: "%.2f", timings.fullPipeline)) seconds")
            Logging.info("- Decoding Loop:     \(String(format: "%.2f", decodeLoopTime)) seconds")
            Logging.info("Time to first token: \(String(format: "%.2f", timeToFirstToken)) seconds")
            Logging.info("Total Tokens:        \(totalTokens)")
            Logging.info("Tokens per Second:   \(String(format: "%.2f", tokensPerSecond)) tok/s")
            Logging.info("Real Time Factor:    \(String(format: "%.2f", rtf))")
            Logging.info("Fallbacks:           \(timings.totalDecodingFallbacks)")
        }

        for segment in allSegments {
            // Log segments
            let start = segment.start
            let end = segment.end
            let text = segment.text
            let line = "[\(formatTimestamp(start)) --> \(formatTimestamp(end))] \(text)"
            Logging.debug(line)
        }

        let wordTokens = allTokens.filter { $0 < tokenizer.specialTokenBegin }
        transcription = tokenizer.decode(tokens: wordTokens)

        transcription = transcription.trimmingCharacters(in: .whitespaces)

        return TranscriptionResult(text: transcription, segments: allSegments, language: "en", timings: timings)
    }
}
