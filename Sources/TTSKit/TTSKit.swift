//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation
import Hub
import Tokenizers

// MARK: - TTSKit

/// Generic TTS orchestrator: text chunking, concurrent generation, crossfade, and audio playback.
///
/// Following the WhisperKit pattern, `TTSKit` exposes each model component as a
/// protocol-typed `public var`. Swap any component at runtime to change behaviour:
/// ```swift
/// let tts = try await TTSKit(config, load: false)
/// tts.codeDecoder = MyOptimisedCodeDecoder()
/// try await tts.loadModels()
/// ```
///
/// The default implementation uses Qwen3 TTS components (`Sources/TTSKit/Qwen3TTS/`).
/// Components from entirely different model families can be plugged in by conforming
/// to the same component protocols, or by implementing `TTSModel` directly.
///
/// `makeTask(progress:)` returns an `any TTSGenerating` - override `makeTask` to
/// use a completely different generation algorithm while keeping the chunking, concurrency,
/// crossfade, and playback orchestration provided by `generateSpeech` and `playSpeech`.
open class TTSKit: @unchecked Sendable {
    // MARK: - Model components (protocol-typed, swappable)

    /// Text token -> embedding. Conforms to `TextProjecting`.
    public var textProjector: any TextProjecting
    /// Codec-0 token -> embedding. Conforms to `CodeEmbedding`.
    public var codeEmbedder: any CodeEmbedding
    /// Multi-code token -> embedding. Conforms to `MultiCodeEmbedding`.
    public var multiCodeEmbedder: any MultiCodeEmbedding
    /// Autoregressive code-0 decoder. Conforms to `CodeDecoding`.
    public var codeDecoder: any CodeDecoding
    /// Per-frame decoder. Conforms to `MultiCodeDecoding`.
    public var multiCodeDecoder: any MultiCodeDecoding
    /// RVQ codes -> audio waveform. Conforms to `SpeechDecoding`.
    public var speechDecoder: any SpeechDecoding
    /// Tokenizer. `nil` before the first `loadModels()` call or after `unloadModels()`.
    public var tokenizer: (any Tokenizer)?

    // MARK: - Configuration & timing

    public var config: TTSKitConfig

    // TODO: store in currentTimings
    /// Wall-clock seconds for the most recent full model load.
    public private(set) var modelLoadTime: TimeInterval = 0
    /// Wall-clock seconds for the most recent tokenizer load.
    public private(set) var tokenizerLoadTime: TimeInterval = 0

    // MARK: - Audio output

    /// Audio output used by `playSpeech`.
    // TODO: Invesigate merging with AudioProcesor from WhisperKit
    public let audioOutput = TTSAudioOutput()

    // MARK: - Seed (for sampler repeatability)

    public let seed: UInt64?
    private var taskCounter: UInt64 = 0

    // MARK: - Initialization

    /// Create a `TTSKit` instance from a `TTSKitConfig`.
    ///
    /// Uses the component overrides in `config` if set; otherwise instantiates the default
    /// Qwen3 TTS components. Components can also be replaced after init.
    ///
    /// - Parameters:
    ///   - config: Pipeline configuration (model preset, paths, compute units, component overrides).
    ///   - seed: Optional seed for reproducible generation. Each concurrent task gets a derived
    ///           seed (`seed ^ taskIndex`) so tasks don't race on shared RNG state.
    ///   - load: When `true`, downloads (if needed) and loads all models immediately.
    public init(
        _ config: TTSKitConfig = TTSKitConfig(),
        seed: UInt64? = nil,
        load: Bool = true
    ) async throws {
        self.config = config
        self.seed = seed

        Logging.shared.logLevel = config.verbose ? config.logLevel : .none

        // Use overrides from config, falling back to Qwen3 TTS defaults
        self.textProjector = config.textProjector ?? TTSTextProjector()
        self.codeEmbedder = config.codeEmbedder ?? TTSCodeEmbedder()
        self.multiCodeEmbedder = config.multiCodeEmbedder ?? TTSMultiCodeEmbedder()
        self.codeDecoder = config.codeDecoder ?? TTSCodeDecoder()
        self.multiCodeDecoder = config.multiCodeDecoder ?? TTSMultiCodeDecoder()
        self.speechDecoder = config.speechDecoder ?? TTSSpeechDecoder()

        // TODO: add download configurability
        if load {
            try await loadModels()
        }
    }

    // MARK: - Download

    /// Download Qwen3 TTS models from HuggingFace Hub.
    ///
    /// Downloads only the files matching the configured component variants.
    /// Files are cached locally by the Hub library.
    open class func download(
        config: TTSKitConfig = TTSKitConfig(),
        progressCallback: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        // TODO: match whisperkit download options
        let hubApi = HubApi(hfToken: config.token)
        let hubRepo = Hub.Repo(id: config.modelRepo, type: .models)

        do {
            return try await hubApi.snapshot(from: hubRepo, matching: config.downloadPatterns) { progress in
                progressCallback?(progress)
            }
        } catch {
            throw TTSError.generationFailed(
                "Failed to download models from \(config.modelRepo). " +
                    "Check that the repo exists and you have access. Error: \(error.localizedDescription)"
            )
        }
    }

    // MARK: - Model lifecycle

    /// Prewarm all CoreML models by compiling them sequentially, then discarding.
    ///
    /// Serializes CoreML compilation to cap peak memory. Call before `loadModels()` on
    /// first launch or after a model update.
    open func prewarmModels() async throws {
        try await loadModels(prewarmMode: true)
    }

    /// Load all models and the tokenizer.
    ///
    /// If `config.modelsPath` is nil, downloads from HuggingFace Hub first.
    ///
    /// - Parameter prewarmMode: When `true`, compile models one at a time and discard weights
    ///   to limit peak memory (prewarm). When `false` (default), load all concurrently.
    open func loadModels(prewarmMode: Bool = false) async throws {
        let embedUnits = config.computeOptions.embedderComputeUnits
        let cdUnits = config.computeOptions.codeDecoderComputeUnits
        let mcdUnits = config.computeOptions.multiCodeDecoderComputeUnits
        let sdUnits = config.computeOptions.speechDecoderComputeUnits

        // Ensure a local model path - download if needed
        if config.modelsPath == nil {
            Logging.info("Downloading models from \(config.modelRepo)...")
            let modelFolder = try await TTSKit.download(config: config) { progress in
                let percent = Int(progress.fractionCompleted * 100)
                Logging.debug("  Download: \(percent)%")
            }
            config.modelsPath = modelFolder.path
            Logging.info("Models cached at \(modelFolder.path)")
        }

        guard let modelsPath = config.modelsPath,
              FileManager.default.fileExists(atPath: modelsPath)
        else {
            throw TTSError.modelNotFound(config.modelsPath ?? "<nil>")
        }

        // Resolve all six component URLs. A nil result means the .mlmodelc bundle is
        // missing from disk - surface this immediately rather than crashing later.
        func requireURL(_ component: String, _ variant: String) throws -> URL {
            guard let url = config.modelURL(component: component, variant: variant) else {
                throw TTSError.invalidConfiguration(
                    "No .mlmodelc found at \(component)/\(config.versionDir)/\(variant) " +
                        "inside \(config.modelsPath ?? "<nil>"). "
                )
            }
            return url
        }
        let tpURL = try requireURL("text_projector", config.textProjectorVariant)
        let ceURL = try requireURL("code_embedder", config.codeEmbedderVariant)
        let mceURL = try requireURL("multi_code_embedder", config.multiCodeEmbedderVariant)
        let cdURL = try requireURL("code_decoder", config.codeDecoderVariant)
        let mcdURL = try requireURL("multi_code_decoder", config.multiCodeDecoderVariant)
        let sdURL = try requireURL("speech_decoder", config.speechDecoderVariant)

        // Load tokenizer (skipped in prewarm - only CoreML compilation needed)
        // TODO: seperate function for this that supports offline loading similar to whisperkit
        if !prewarmMode {
            let tokenizerStart = CFAbsoluteTimeGetCurrent()
            Logging.info("Loading tokenizer from \(config.tokenizerSource)...")
            let tokenizerURL = URL(fileURLWithPath: config.tokenizerSource)
            if FileManager.default.fileExists(atPath: tokenizerURL.appending(path: "tokenizer.json").path) {
                self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
            } else {
                self.tokenizer = try await AutoTokenizer.from(pretrained: config.tokenizerSource)
            }
            self.tokenizerLoadTime = CFAbsoluteTimeGetCurrent() - tokenizerStart
            Logging.info(String(format: "Tokenizer loaded in %.2fs", tokenizerLoadTime))
        }

        // Load the six CoreML models.
        // Prewarm: sequential to serialize compilation -> lower peak memory.
        // Normal: concurrent since compiled artifacts are already cached.
        let modelLoadStart = CFAbsoluteTimeGetCurrent()

        if prewarmMode {
            Logging.info("Prewarming 6 CoreML models sequentially (serializing compilation)...")
            try await textProjector.loadModel(at: tpURL, computeUnits: embedUnits, prewarmMode: true)
            try await codeEmbedder.loadModel(at: ceURL, computeUnits: embedUnits, prewarmMode: true)
            try await multiCodeEmbedder.loadModel(at: mceURL, computeUnits: embedUnits, prewarmMode: true)
            try await codeDecoder.loadModel(at: cdURL, computeUnits: cdUnits, prewarmMode: true)
            try await multiCodeDecoder.loadModel(at: mcdURL, computeUnits: mcdUnits, prewarmMode: true)
            try await speechDecoder.loadModel(at: sdURL, computeUnits: sdUnits, prewarmMode: true)
            Logging.info(String(format: "Prewarm complete in %.2fs", CFAbsoluteTimeGetCurrent() - modelLoadStart))
        } else {
            Logging.info("Loading 6 CoreML models concurrently...")
            Logging.debug("  TextProjector:     \(tpURL.lastPathComponent)  compute: \(embedUnits.description)")
            Logging.debug("  CodeEmbedder:      \(ceURL.lastPathComponent)  compute: \(embedUnits.description)")
            Logging.debug("  MultiCodeEmbedder: \(mceURL.lastPathComponent) compute: \(embedUnits.description)")
            Logging.debug("  CodeDecoder:       \(cdURL.lastPathComponent)  (\(config.codeDecoderVariant), compute: \(cdUnits.description))")
            Logging.debug("  MultiCodeDecoder:  \(mcdURL.lastPathComponent) (\(config.multiCodeDecoderVariant), compute: \(mcdUnits.description))")
            Logging.debug("  SpeechDecoder:     \(sdURL.lastPathComponent)  (\(config.speechDecoderVariant), compute: \(sdUnits.description))")

            // TODO: naming
            async let tp: Void = textProjector.loadModel(at: tpURL, computeUnits: embedUnits)
            async let ce: Void = codeEmbedder.loadModel(at: ceURL, computeUnits: embedUnits)
            async let mce: Void = multiCodeEmbedder.loadModel(at: mceURL, computeUnits: embedUnits)
            async let cd: Void = codeDecoder.loadModel(at: cdURL, computeUnits: cdUnits)
            async let mcd: Void = multiCodeDecoder.loadModel(at: mcdURL, computeUnits: mcdUnits)
            async let sd: Void = speechDecoder.loadModel(at: sdURL, computeUnits: sdUnits)
            _ = try await (tp, ce, mce, cd, mcd, sd)

            let modelLoad = CFAbsoluteTimeGetCurrent() - modelLoadStart
            self.modelLoadTime = modelLoad

            // Sync audio output sample rate to the loaded speech decoder.
            audioOutput.configure(sampleRate: speechDecoder.sampleRate)

            Logging.info(String(format: "Tokenizer loaded in %.2fs", tokenizerLoadTime))
            Logging.info(String(format: "Total model load: %.2fs", modelLoadTime))
        }
    }

    /// Release all model weights and the tokenizer from memory.
    open func unloadModels() {
        textProjector.unloadModel()
        codeEmbedder.unloadModel()
        multiCodeEmbedder.unloadModel()
        codeDecoder.unloadModel()
        multiCodeDecoder.unloadModel()
        speechDecoder.unloadModel()
        tokenizer = nil
    }

    // MARK: - Task factory

    /// Create a fresh generation task using the current components.
    ///
    /// Each call returns an independent `TTSGenerateTask` with its own sampler seed and
    /// per-task buffers. Override this method to return a custom `any TTSGenerating`
    /// implementation while keeping the orchestration in `generateSpeech`.
    open func makeTask(progress: Progress? = nil) -> any TTSGenerating {
        let derivedSeed: UInt64? = seed.map { $0 ^ taskCounter }
        taskCounter += 1

        return TTSGenerateTask(
            textProjector: textProjector,
            codeEmbedder: codeEmbedder,
            multiCodeEmbedder: multiCodeEmbedder,
            codeDecoder: codeDecoder,
            multiCodeDecoder: multiCodeDecoder,
            speechDecoder: speechDecoder,
            sampler: TTSGreedyTokenSampler(seed: derivedSeed),
            tokenizer: tokenizer!,
            suppressTokenIds: config.suppressTokenIds,
            // TODO: pass timing object instead of specific timings
            modelLoadTime: modelLoadTime,
            tokenizerLoadTime: tokenizerLoadTime,
            progress: progress
        )
    }

    // MARK: - Speech generation

    /// Generate speech from text, returning the complete audio result.
    ///
    /// Handles text chunking, concurrent chunk generation, and crossfade between chunks.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize.
    ///   - speaker: Speaker voice. Passed as `speaker.rawValue` to the generation task.
    ///   - language: Language of the text. Passed as `language.rawValue`.
    ///   - options: Temperature, top-k, chunking strategy, concurrency, etc.
    ///   - callback: Per-step callback receiving decoded audio and timing info.
    ///               `TTSProgress.stepTime` is non-nil only on the first step.
    ///               Return `false` to cancel; `nil` or `true` to continue.
    // TODO: move to Qwen specific class
    open func generateSpeech(
        text: String,
        speaker: Qwen3Speaker = .ryan,
        language: Qwen3Language = .english,
        options: TTSGenerationOptions = TTSGenerationOptions(),
        callback: TTSCallback = nil
    ) async throws -> TTSResult {
        try await generateSpeech(
            text: text,
            voice: speaker.rawValue,
            language: language.rawValue,
            options: options,
            callback: callback
        )
    }

    /// Model-agnostic overload using raw string voice/language identifiers.
    open func generateSpeech(
        text: String,
        voice: String,
        language: String,
        options: TTSGenerationOptions = TTSGenerationOptions(),
        callback: TTSCallback = nil
    ) async throws -> TTSResult {
        let effectiveStrategy = options.chunkingStrategy ?? .sentence
        let textChunks: [String]
        if effectiveStrategy == .none || tokenizer == nil {
            textChunks = [text]
        } else {
            let chunker = TTSTextChunker(
                targetChunkSize: options.targetChunkSize ?? TTSTextChunker.defaultTargetChunkSize,
                minChunkSize: options.minChunkSize ?? TTSTextChunker.defaultMinChunkSize,
                tokenizer: tokenizer! // TODO: dont force unwrap
            )
            let chunks = chunker.chunk(text)
            textChunks = chunks.isEmpty ? [text] : chunks
        }

        // Single chunk, fast path
        if textChunks.count == 1 {
            return try await makeTask().run(
                text: textChunks[0],
                voice: voice,
                language: language,
                options: options,
                callback: callback
            )
        }

        let workerDesc = options.concurrentWorkerCount == nil ? "max" : "\(options.concurrentWorkerCount!)"
        Logging.info("Chunked TTS: \(textChunks.count) chunks, concurrency=\(workerDesc)")
        for (i, chunk) in textChunks.enumerated() {
            let truncated = chunk.count > 60 ? "\(chunk.prefix(60))..." : chunk
            Logging.debug("  Chunk \(i): \"\(truncated)\" (\(chunk.count) chars)")
        }

        let pipelineStart = CFAbsoluteTimeGetCurrent()
        var combinedTimings = TTSTimings()
        combinedTimings.modelLoading = modelLoadTime
        combinedTimings.tokenizerLoading = tokenizerLoadTime

        let crossfadeSamples = speechDecoder.sampleRate / 10 // 100ms crossfade
        var chunkAudioArrays = [[Float]](repeating: [], count: textChunks.count)

        // TODO: cleanup this section
        if options.concurrentWorkerCount == 1 {
            for (i, chunkText) in textChunks.enumerated() {
                Logging.debug(String(format: "  Generating chunk %d/%d...", i + 1, textChunks.count))
                let chunkResult = try await makeTask().run(
                    text: chunkText, voice: voice, language: language,
                    options: options, callback: callback
                )
                chunkAudioArrays[i] = chunkResult.audio
                mergeTimings(&combinedTimings, from: chunkResult.timings)
                if i == 0 { combinedTimings.timeToFirstBuffer = chunkResult.timings.timeToFirstBuffer }
                Logging.debug(String(format: "  Chunk %d done: %.2fs audio (%d steps)",
                                     i + 1, chunkResult.audioDuration, Int(chunkResult.timings.totalDecodingLoops)))
            }
        } else {
            let indexedChunks = textChunks.enumerated().map { (index: $0.offset, text: $0.element) }
            let batchedChunks: [[(index: Int, text: String)]]
            if let sz = options.concurrentWorkerCount, sz > 0 {
                batchedChunks = stride(from: 0, to: indexedChunks.count, by: sz).map {
                    Array(indexedChunks[$0..<min($0 + sz, indexedChunks.count)])
                }
            } else {
                // nil = unlimited: run all chunks in one batch
                batchedChunks = [indexedChunks]
            }

            for batch in batchedChunks {
                let chunkCount = textChunks.count
                let taskItems: [(index: Int, text: String, task: any TTSGenerating)] =
                    batch.map { (index: $0.index, text: $0.text, task: makeTask()) }

                let batchResults: [(index: Int, result: TTSResult)] = try await withThrowingTaskGroup(
                    of: (index: Int, result: TTSResult).self
                ) { group in
                    for item in taskItems {
                        group.addTask {
                            Logging.debug(String(format: "  Starting chunk %d/%d...", item.index + 1, chunkCount))
                            // TODO: chunked inference kv cache handling for stateful
                            let r = try await item.task.run(
                                text: item.text, voice: voice, language: language,
                                options: options, callback: nil
                            )
                            Logging.debug(String(format: "  Chunk %d done: %.2fs audio (%d steps)",
                                                 item.index + 1, r.audioDuration, Int(r.timings.totalDecodingLoops)))
                            return (index: item.index, result: r)
                        }
                    }
                    var results = [(index: Int, result: TTSResult)]()
                    for try await result in group {
                        results.append(result)
                    }
                    return results
                }

                for entry in batchResults {
                    chunkAudioArrays[entry.index] = entry.result.audio
                    mergeTimings(&combinedTimings, from: entry.result.timings)
                    if entry.index == 0 { combinedTimings.timeToFirstBuffer = entry.result.timings.timeToFirstBuffer }
                }
            }

            // Deliver audio in order via callback after concurrent batch completes
            if let callback {
                for (i, chunkAudio) in chunkAudioArrays.enumerated() {
                    let progress = TTSProgress(audio: chunkAudio, timings: combinedTimings, stepTime: i == 0 ? 0 : nil)
                    if callback(progress) == false { break }
                }
            }
        }

        // Crossfade consecutive chunks and assemble final audio
        // TODO: move to helper / TTSAudioOutput
        var allAudio: [Float] = chunkAudioArrays.first ?? []
        for i in 1..<chunkAudioArrays.count {
            let chunk = chunkAudioArrays[i]
            let fadeLen = min(crossfadeSamples, allAudio.count, chunk.count)
            if fadeLen > 0 {
                let overlapStart = allAudio.count - fadeLen
                for j in 0..<fadeLen {
                    let t = Float(j) / Float(fadeLen)
                    allAudio[overlapStart + j] = allAudio[overlapStart + j] * (1 - t) + chunk[j] * t
                }
                if fadeLen < chunk.count { allAudio.append(contentsOf: chunk[fadeLen...]) }
            } else {
                allAudio.append(contentsOf: chunk)
            }
        }

        combinedTimings.fullPipeline = CFAbsoluteTimeGetCurrent() - pipelineStart
        let sr = speechDecoder.sampleRate
        combinedTimings.inputAudioSeconds = Double(allAudio.count) / Double(sr)

        let steps = Int(combinedTimings.totalDecodingLoops)
        let avgMs = steps > 0 ? combinedTimings.generationLoop * 1000 / Double(steps) : 0
        Logging.info(String(format: "Chunked TTS: %d chunks, %d steps, %.1fms avg/step, %.2fs audio",
                            textChunks.count, steps, avgMs,
                            Double(allAudio.count) / Double(sr)))

        return TTSResult(audio: allAudio, timings: combinedTimings, sampleRate: sr)
    }

    // MARK: - Timing accumulation

    // TODO: see if we can avoid inout
    open func mergeTimings(_ combined: inout TTSTimings, from chunk: TTSTimings) {
        combined.tokenize += chunk.tokenize
        combined.prefill += chunk.prefill
        combined.prefillTokens += chunk.prefillTokens
        combined.generationLoop += chunk.generationLoop
        combined.codeDecoder += chunk.codeDecoder
        combined.multiCodeDecoder += chunk.multiCodeDecoder
        combined.multiCodeDecoderPredictions += chunk.multiCodeDecoderPredictions
        combined.multiCodeDecoderSampling += chunk.multiCodeDecoderSampling
        combined.multiCodeDecoderEmbedding += chunk.multiCodeDecoderEmbedding
        combined.multiCodeDecoderKvCache += chunk.multiCodeDecoderKvCache
        combined.totalMultiCodeDecoderPredictions += chunk.totalMultiCodeDecoderPredictions
        combined.speechDecoder += chunk.speechDecoder
        combined.speechDecoderPredictions += chunk.speechDecoderPredictions
        combined.kvCacheUpdate += chunk.kvCacheUpdate
        combined.codeEmbed += chunk.codeEmbed
        combined.codecHidden += chunk.codecHidden
        combined.textProjection += chunk.textProjection
        combined.sampling += chunk.sampling
        combined.totalDecodingLoops += chunk.totalDecodingLoops
    }

    // MARK: - Play Speech

    /// Helper to generate speech and stream it through the audio output in real time.
    ///
    /// Forces sequential chunking (`concurrentWorkerCount = 1`) for per-frame streaming.
    open func playSpeech(
        text: String,
        voice: String,
        language: String,
        options: TTSGenerationOptions = TTSGenerationOptions(),
        playbackStrategy: TTSPlaybackStrategy = .auto,
        callback: TTSCallback = nil
    ) async throws -> TTSResult {
        var playOptions = options
        playOptions.concurrentWorkerCount = 1

        let audioOut = audioOutput
        let maxTokens = playOptions.maxNewTokens

        if case .generateFirst = playbackStrategy {
            let result = try await generateSpeech(
                text: text, voice: voice, language: language,
                options: playOptions, callback: callback
            )
            try audioOut.startPlayback()
            audioOut.setBufferDuration(0)
            audioOut.enqueueAudioChunk(result.audio)
            await audioOut.stopPlayback(waitForCompletion: true)
            return result
        }

        try audioOut.startPlayback()
        switch playbackStrategy {
            case .stream: audioOut.setBufferDuration(0)
            case let .buffered(secs): audioOut.setBufferDuration(secs)
            case .auto: break
            case .generateFirst: break
        }

        let result = try await generateSpeech(
            text: text, voice: voice, language: language,
            options: playOptions,
            callback: { progress in
                // On the first step, stepTime is set - configure adaptive buffer before enqueuing.
                if let stepTime = progress.stepTime, case .auto = playbackStrategy {
                    let spf = self.speechDecoder.samplesPerFrame
                    let sr = self.speechDecoder.sampleRate
                    let buffer = TTSPlaybackStrategy.requiredBuffer(
                        stepTime: stepTime,
                        maxNewTokens: maxTokens,
                        samplesPerFrame: spf,
                        sampleRate: sr
                    )
                    audioOut.setBufferDuration(buffer)
                    let speedRatio = TTSPlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr) / stepTime
                    Logging.info(String(format: "Playback: step %.1fms (%.2fx real-time) -> buffer %.2fs",
                                        stepTime * 1000, speedRatio, buffer))
                }
                audioOut.enqueueAudioChunk(progress.audio)
                return callback?(progress)
            }
        )

        await audioOut.stopPlayback(waitForCompletion: true)
        return result
    }

    /// Generate speech and stream it through the audio output in real time.
    ///
    /// Typed convenience overload. Forces sequential chunking (`concurrentWorkerCount = 1`)
    /// for per-frame streaming.
    // TODO: move to Qwen specific class
    open func playSpeech(
        text: String,
        speaker: Qwen3Speaker = .ryan,
        language: Qwen3Language = .english,
        options: TTSGenerationOptions = TTSGenerationOptions(),
        playbackStrategy: TTSPlaybackStrategy = .auto,
        callback: TTSCallback = nil
    ) async throws -> TTSResult {
        try await playSpeech(
            text: text,
            voice: speaker.rawValue,
            language: language.rawValue,
            options: options,
            playbackStrategy: playbackStrategy,
            callback: callback
        )
    }
}

// MARK: - TTSModel conformance

extension TTSKit: TTSModel {
    /// The output sample rate of the currently loaded speech decoder.
    public var sampleRate: Int { speechDecoder.sampleRate }

    /// Generate speech from text using a plain string voice/language identifier.
    ///
    /// This is the `TTSModel` protocol entry point. Typed `Qwen3Speaker`/`Qwen3Language`
    /// overloads are available directly on `TTSKit`; this method accepts raw strings so
    /// that any model family can be driven through the same `TTSModel` interface.
    public func generate(
        text: String,
        voice: String,
        language: String,
        options: TTSGenerationOptions,
        callback: TTSCallback
    ) async throws -> TTSResult {
        try await generateSpeech(text: text, voice: voice, language: language, options: options, callback: callback)
    }
}
