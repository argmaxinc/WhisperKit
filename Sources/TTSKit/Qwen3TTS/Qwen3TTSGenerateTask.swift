//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation
import Tokenizers

// MARK: - TTSGenerateTask

/// Qwen3 TTS single-chunk generation task.
///
/// The core building block for Qwen3 TTS generation, analogous to `TranscribeTask`
/// in WhisperKit. Each task creates its own KV caches, MLState, and sampler, then
/// runs a complete prefill + autoregressive decode cycle.
///
/// Conforms to `TTSGenerating` so it can be returned by `TTSKit.makeTask(progress:)`
/// and consumed by `TTSKit`'s generic orchestration layer.
///
/// Thread safety: all stored properties are `let` (immutable after init). Each task
/// owns its own sampler (derived seed) so concurrent tasks don't share RNG state.
/// Model components are shared read-only references - `MLModel.prediction()` is
/// thread-safe. The class is `@unchecked Sendable` to permit `open` subclassing.
open class TTSGenerateTask: @unchecked Sendable, TTSGenerating {
    /// Model components
    public let textProjector: any TextProjecting
    public let codeEmbedder: any CodeEmbedding
    public let multiCodeEmbedder: any MultiCodeEmbedding
    public let codeDecoder: any CodeDecoding
    public let multiCodeDecoder: any MultiCodeDecoding
    public let speechDecoder: any SpeechDecoding
    public let sampler: any TTSTokenSampling
    public let tokenizer: any Tokenizer
    public let suppressTokenIds: Set<Int>

    /// Timings from model loading
    public let modelLoadTime: TimeInterval
    public let tokenizerLoadTime: TimeInterval

    /// Progress object for tracking generation. `totalUnitCount` is set to
    /// `maxNewTokens` at the start of generation; `completedUnitCount` is
    /// updated after each decoding step.
    public let progress: Progress

    // MARK: - Initialization

    public init(
        textProjector: any TextProjecting,
        codeEmbedder: any CodeEmbedding,
        multiCodeEmbedder: any MultiCodeEmbedding,
        codeDecoder: any CodeDecoding,
        multiCodeDecoder: any MultiCodeDecoding,
        speechDecoder: any SpeechDecoding,
        sampler: any TTSTokenSampling,
        tokenizer: any Tokenizer,
        suppressTokenIds: Set<Int>,
        modelLoadTime: TimeInterval = 0,
        tokenizerLoadTime: TimeInterval = 0,
        progress: Progress? = nil
    ) {
        self.textProjector = textProjector
        self.codeEmbedder = codeEmbedder
        self.multiCodeEmbedder = multiCodeEmbedder
        self.codeDecoder = codeDecoder
        self.multiCodeDecoder = multiCodeDecoder
        self.speechDecoder = speechDecoder
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.suppressTokenIds = suppressTokenIds
        self.modelLoadTime = modelLoadTime
        self.tokenizerLoadTime = tokenizerLoadTime
        self.progress = progress ?? Progress()
    }

    // MARK: - Run

    /// Generate speech for a single text segment.
    ///
    /// Creates fresh KV caches, runs prefill, then autoregressive generation with
    /// interleaved SpeechDecoder audio output. Safe to call concurrently from
    /// multiple tasks against the same model instances.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Raw string matching `Qwen3Speaker.rawValue`; falls back to `.ryan`.
    ///   - language: Raw string matching `Qwen3Language.rawValue`; falls back to `.english`.
    ///   - options: Generation options (temperature, top-k, etc.)
    ///   - callback: Per-step callback receiving decoded audio and running timings.
    ///               `TTSProgress.stepTime` is non-nil only on the first step.
    ///               Return `false` to cancel; `nil` or `true` to continue.
    /// - Returns: A `TTSResult` containing the complete audio and timings for this chunk.
    // TODO: break up into seperate functions
    open func run(
        text: String,
        voice: String,
        language: String,
        options: TTSGenerationOptions,
        callback: TTSCallback
    ) async throws -> TTSResult {
        let speaker = Qwen3Speaker(rawValue: voice) ?? .ryan
        let lang = Qwen3Language(rawValue: language) ?? .english

        var timings = TTSTimings()
        timings.modelLoading = modelLoadTime
        timings.tokenizerLoading = tokenizerLoadTime
        let pipelineStart = CFAbsoluteTimeGetCurrent()
        var firstBufferEmitted = false

        // Create task-local MLState for stateful decoders (nil for non-stateful)
        let cdState = codeDecoder.makeState()

        progress.totalUnitCount = Int64(options.maxNewTokens)
        progress.completedUnitCount = 0

        // MARK: Tokenize and build dual-track embeddings

        let tokenizeStart = CFAbsoluteTimeGetCurrent()

        let embedDim = codeDecoder.embedSize
        let zeroCodecEmbed = zeroEmbed(dim: embedDim)

        // Build instruction segment (text-only, prepended before the main segment)
        // The instruction acts like a system prompt: <|im_start|>user\n{instruction}<|im_end|>\n
        // It's pure text (no audio coding) that tells the model how to speak (speed, emotion, style)
        var instructTextEmbeds: [EmbedBuffer] = []
        var instructCodecEmbeds: [EmbedBuffer] = []
        if let instruction = options.instruction, !instruction.isEmpty {
            let instructPrompt = "<|im_start|>user\n\(instruction)<|im_end|>\n"
            let instructTokenIds = tokenizer.encode(text: instructPrompt).map { Int32($0) }

            for tokenId in instructTokenIds {
                try instructTextEmbeds.append(textProjector.project(tokenId: tokenId))
                instructCodecEmbeds.append(zeroCodecEmbed)
            }

            Logging.debug("Instruction: \(instructTokenIds.count) tokens")
        }

        let rolePrefix = "<|im_start|>assistant\n"
        let roleTokenIds = tokenizer.encode(text: rolePrefix).map { Int32($0) }
        let textTokenIds = tokenizer.encode(text: text).map { Int32($0) }

        guard !textTokenIds.isEmpty else {
            throw TTSError.emptyText
        }

        let firstTextToken = textTokenIds[0]
        let trailingTextTokens = Array(textTokenIds.dropFirst())

        Logging.debug("Role prefix: \(roleTokenIds.count) tokens")
        Logging.debug("Text: \(textTokenIds.count) tokens (\(trailingTextTokens.count) trailing)")

        // Build text track embeddings: instruction + role + PADs + BOS + first_text
        var textTrackEmbeds: [EmbedBuffer] = instructTextEmbeds
        for tokenId in roleTokenIds {
            try textTrackEmbeds.append(textProjector.project(tokenId: tokenId))
        }
        let textPadEmbed = try textProjector.project(tokenId: Qwen3TTSConstants.textPAD)
        let textBosEmbed = try textProjector.project(tokenId: Qwen3TTSConstants.textBOS)
        let firstTextEmbed = try textProjector.project(tokenId: firstTextToken)

        // Codec track: [THINK, THINK_BOS, LANGUAGE, THINK_EOS, SPEAKER, PAD, BOS]
        let codecIds: [Int32] = [
            Qwen3TTSConstants.codecTHINK,
            Qwen3TTSConstants.codecTHINK_BOS,
            lang.tokenID,
            Qwen3TTSConstants.codecTHINK_EOS,
            speaker.tokenID,
            Qwen3TTSConstants.codecPAD,
            Qwen3TTSConstants.codecBOS,
        ]
        var codecTrackEmbeds: [EmbedBuffer] = []
        for codecId in codecIds {
            try codecTrackEmbeds.append(codeEmbedder.embed(tokenId: codecId))
        }

        // Align text track: (instruction +) role + PADs + BOS + first_text
        let numPads = codecIds.count - 2
        for _ in 0..<numPads {
            textTrackEmbeds.append(textPadEmbed)
        }
        textTrackEmbeds.append(textBosEmbed)
        textTrackEmbeds.append(firstTextEmbed)

        // Align codec track: zeros for instruction + zeros for role prefix + codec tokens
        var fullCodecTrackEmbeds: [EmbedBuffer] = instructCodecEmbeds
        fullCodecTrackEmbeds.append(contentsOf: Array(repeating: zeroCodecEmbed, count: roleTokenIds.count))
        fullCodecTrackEmbeds.append(contentsOf: codecTrackEmbeds)

        assert(
            textTrackEmbeds.count == fullCodecTrackEmbeds.count,
            "Track alignment mismatch: text=\(textTrackEmbeds.count) codec=\(fullCodecTrackEmbeds.count)"
        )

        // Combine tracks via element-wise addition
        var combinedEmbeds: [EmbedBuffer] = []
        for i in 0..<textTrackEmbeds.count {
            combinedEmbeds.append(addEmbeddings(textTrackEmbeds[i], fullCodecTrackEmbeds[i]))
        }

        timings.tokenize = CFAbsoluteTimeGetCurrent() - tokenizeStart

        // MARK: Prefill CodeDecoder

        let prefillStart = CFAbsoluteTimeGetCurrent()
        let cdCache = try TTSKVCache(
            cacheDim: codeDecoder.kvCacheEmbedDim,
            maxSeqLength: codeDecoder.kvCacheMaxSequenceLength,
            isStateful: codeDecoder.isStateful
        )

        var lastCdOutput: CodeDecoderOutput!
        for (i, embed) in combinedEmbeds.enumerated() {
            if i > 0 {
                cdCache.update(keyCacheUpdates: lastCdOutput.keyCacheUpdates, valueCacheUpdates: lastCdOutput.valueCacheUpdates)
            }
            let embedArr = try createEmbedMLArray(embed)
            lastCdOutput = try codeDecoder.decode(inputEmbeds: embedArr, cache: cdCache, state: cdState)
        }

        timings.prefill = CFAbsoluteTimeGetCurrent() - prefillStart
        timings.prefillTokens = Double(combinedEmbeds.count)

        let prefillMs = timings.prefill * 1000
        let prefillTokS = timings.prefill > 0 ? Double(combinedEmbeds.count) / timings.prefill : 0
        Logging.info(String(
            format: "Prefill: %.1fms (%d tokens, %.1f tok/s)",
            prefillMs, combinedEmbeds.count, prefillTokS
        ))

        // MARK: Autoregressive RVQ generation with interleaved audio decode

        var generatedTokens: [Int32] = []
        var allAudio: [Float] = []

        // Set up SpeechDecoder cache with the correct context size for this model variant
        let sdCache = try TTSSpeechDecoderCache(
            cacheDim: speechDecoder.kvCacheEmbedDim,
            maxSeqLength: speechDecoder.kvCacheMaxSequenceLength,
            hiddenDim: speechDecoder.hiddenDim,
            hiddenContextLen: speechDecoder.hiddenContextLen
        )

        // Sample first code0 after prefill
        var code0 = sampler.sampleCodec0(
            logits: lastCdOutput.logits,
            temperature: options.temperature,
            topK: options.topK,
            generatedTokens: generatedTokens,
            repetitionPenalty: options.repetitionPenalty,
            suppressTokenIds: suppressTokenIds
        )
        generatedTokens.append(code0)

        // MARK: Decoding loop

        // Cap output at 6x the total text input length (role prefix + text tokens) to limit hallucinations.
        // Empirically, p98 across 10k samples is 42 text tokens -> 256 audio frames (~6x).
        let maxStepsByPrefill = 6 * (roleTokenIds.count + textTokenIds.count)

        var stepIndex = 0
        while code0 != Qwen3TTSConstants.codecEOS && !cdCache.isFull && stepIndex < options.maxNewTokens && stepIndex < maxStepsByPrefill {
            try Task.checkCancellation()
            let stepStart = CFAbsoluteTimeGetCurrent()
            var t: CFAbsoluteTime

            // Update CodeDecoder cache with previous output
            t = CFAbsoluteTimeGetCurrent()
            cdCache.update(keyCacheUpdates: lastCdOutput.keyCacheUpdates, valueCacheUpdates: lastCdOutput.valueCacheUpdates)
            timings.kvCacheUpdate += CFAbsoluteTimeGetCurrent() - t

            // Embed code0 via CodeEmbedder
            t = CFAbsoluteTimeGetCurrent()
            let code0Embed = try codeEmbedder.embed(tokenId: code0)
            timings.codeEmbed += CFAbsoluteTimeGetCurrent() - t

            // Generate codes 1-15 via MultiCodeDecoder (autoreleasepool for IOSurface memory)
            t = CFAbsoluteTimeGetCurrent()
            let mcdResult = try autoreleasepool {
                try multiCodeDecoder.generateMultiCodes(
                    hiddenStates: lastCdOutput.hiddenStates,
                    code0Embed: code0Embed,
                    multiCodeEmbedder: multiCodeEmbedder,
                    sampler: sampler,
                    options: options
                )
            }
            timings.multiCodeDecoder += CFAbsoluteTimeGetCurrent() - t
            timings.multiCodeDecoderPredictions += mcdResult.timings.multiCodeDecoderPredictions
            timings.multiCodeDecoderSampling += mcdResult.timings.multiCodeDecoderSampling
            timings.multiCodeDecoderEmbedding += mcdResult.timings.multiCodeDecoderEmbedding
            timings.multiCodeDecoderKvCache += mcdResult.timings.multiCodeDecoderKvCache
            timings.totalMultiCodeDecoderPredictions += mcdResult.timings.totalMultiCodeDecoderPredictions

            let rvqFrame = [code0] + mcdResult.codes

            // SpeechDecoder runs concurrently with the next three steps via async let;
            // it uses sdCache while CodeDecoder uses cdCache - no shared state
            async let speechResult = speechDecoder.decodeFrameAsync(codes: rvqFrame, cache: sdCache)

            // Compute codecHidden: sum of all 16 code embeddings
            t = CFAbsoluteTimeGetCurrent()
            var allCodeEmbeds: [EmbedBuffer] = [code0Embed]
            for (i, code) in mcdResult.codes.enumerated() {
                let offsetId = code + Int32(multiCodeDecoder.codecVocabSize * i)
                try allCodeEmbeds.append(multiCodeEmbedder.embed(tokenId: offsetId))
            }
            let codecHidden = sumEmbeddings(allCodeEmbeds)
            timings.codecHidden += CFAbsoluteTimeGetCurrent() - t

            // Text projection for this step
            t = CFAbsoluteTimeGetCurrent()
            let textTokenEmbed: EmbedBuffer
            if stepIndex < trailingTextTokens.count {
                textTokenEmbed = try textProjector.project(tokenId: trailingTextTokens[stepIndex])
            } else {
                textTokenEmbed = textPadEmbed
            }
            let combinedEmbed = addEmbeddings(codecHidden, textTokenEmbed)
            let combinedArr = try createEmbedMLArray(combinedEmbed)
            timings.textProjection += CFAbsoluteTimeGetCurrent() - t

            // CodeDecoder prediction (overlaps with SpeechDecoder)
            t = CFAbsoluteTimeGetCurrent()
            lastCdOutput = try codeDecoder.decode(inputEmbeds: combinedArr, cache: cdCache, state: cdState)
            timings.codeDecoder += CFAbsoluteTimeGetCurrent() - t

            // Await SpeechDecoder result and merge timings
            let sdResult = try await speechResult
            timings.speechDecoderPredictions += sdResult.timings.speechDecoderPredictions
            timings.speechDecoder += sdResult.timings.speechDecoderPredictions

            allAudio.append(contentsOf: sdResult.samples)

            // Build progress: stepTime is populated only on the first step so callers
            // can configure adaptive playback buffers before enqueuing audio.
            let stepTime: TimeInterval?
            if !firstBufferEmitted {
                timings.timeToFirstBuffer = CFAbsoluteTimeGetCurrent() - pipelineStart
                firstBufferEmitted = true
                stepTime = CFAbsoluteTimeGetCurrent() - stepStart
            } else {
                stepTime = nil
            }

            let stepProgress = TTSProgress(audio: sdResult.samples, timings: timings, stepTime: stepTime)
            if callback?(stepProgress) == false { break }

            // Sample next code0
            t = CFAbsoluteTimeGetCurrent()
            code0 = sampler.sampleCodec0(
                logits: lastCdOutput.logits,
                temperature: options.temperature,
                topK: options.topK,
                generatedTokens: generatedTokens,
                repetitionPenalty: options.repetitionPenalty,
                suppressTokenIds: suppressTokenIds
            )
            generatedTokens.append(code0)
            timings.sampling += CFAbsoluteTimeGetCurrent() - t

            timings.generationLoop += CFAbsoluteTimeGetCurrent() - stepStart
            stepIndex += 1
            progress.completedUnitCount = Int64(stepIndex)

            if stepIndex % 10 == 0 {
                let stepMs = (CFAbsoluteTimeGetCurrent() - stepStart) * 1000
                let avgMs = timings.generationLoop * 1000 / Double(stepIndex)
                Logging.debug(String(
                    format: "  Step %d: %.1fms (avg %.1fms/step)",
                    stepIndex, stepMs, avgMs
                ))
            }
        }

        // Loop ended, log reason
        let stopReason: String
        if code0 == Qwen3TTSConstants.codecEOS {
            stopReason = "EOS token"
        } else if cdCache.isFull {
            stopReason = "KV cache full (\(cdCache.cacheLength)/\(cdCache.maxSeqLength))"
        } else if stepIndex >= maxStepsByPrefill {
            stopReason = "Audio token ratio limit (\(stepIndex)/\(maxStepsByPrefill) steps, role=\(roleTokenIds.count) text=\(textTokenIds.count))"
        } else if stepIndex >= options.maxNewTokens {
            stopReason = "maxNewTokens limit (\(options.maxNewTokens))"
        } else {
            stopReason = "unknown"
        }
        Logging.info("Loop stopped: \(stopReason) after \(stepIndex) steps")

        timings.totalDecodingLoops = Double(stepIndex)
        timings.fullPipeline = CFAbsoluteTimeGetCurrent() - pipelineStart
        timings.inputAudioSeconds = Double(allAudio.count) / Double(speechDecoder.sampleRate)

        // Mark progress complete
        progress.completedUnitCount = progress.totalUnitCount

        let genMs = timings.generationLoop * 1000
        let avgMs = stepIndex > 0 ? genMs / Double(stepIndex) : 0
        let stepsPerSec = stepIndex > 0 ? Double(stepIndex) / timings.generationLoop : 0
        Logging.info(String(
            format: "Generation: %d frames in %.1fms (%.1fms/step, %.1f frames/s)",
            stepIndex, genMs, avgMs, stepsPerSec
        ))

        return TTSResult(audio: allAudio, timings: timings, sampleRate: speechDecoder.sampleRate)
    }
}
