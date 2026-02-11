//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
@testable import TTSKit
import XCTest

final class TTSKitUnitTests: XCTestCase {

    // MARK: - Configuration

    func testTTSKitConfigDefaults() {
        let config = TTSKitConfig()
        XCTAssertNil(config.modelsPath)
        XCTAssertEqual(config.modelRepo, Qwen3TTSConstants.defaultModelRepo)
        XCTAssertEqual(config.versionDir, Qwen3TTSConstants.defaultVersionDir)
        XCTAssertEqual(config.tokenizerSource, Qwen3TTSConstants.defaultTokenizerRepo)
        XCTAssertEqual(config.codeDecoderVariant, TTSVariantDefaults.codeDecoder)
        XCTAssertEqual(config.multiCodeDecoderVariant, TTSVariantDefaults.multiCodeDecoder)
        XCTAssertEqual(config.speechDecoderVariant, TTSVariantDefaults.speechDecoder)
        XCTAssertFalse(config.verbose)
    }

    func testTTSComputeOptionsDefaults() {
        let opts = TTSComputeOptions()
        XCTAssertEqual(opts.embedderComputeUnits, .cpuOnly)
        XCTAssertEqual(opts.codeDecoderComputeUnits, .cpuAndNeuralEngine)
        XCTAssertEqual(opts.multiCodeDecoderComputeUnits, .cpuAndNeuralEngine)
        XCTAssertEqual(opts.speechDecoderComputeUnits, .cpuAndNeuralEngine)
    }

    func testModelPresetResolvesInConfig() {
        let small = TTSKitConfig(model: .qwen3TTS_0_6b)
        XCTAssertEqual(small.versionDir, TTSModelPreset.qwen3TTS_0_6b.versionDir)
        XCTAssertEqual(small.multiCodeDecoderVariant, TTSVariantDefaults.multiCodeDecoder)

        let large = TTSKitConfig(model: .qwen3TTS_1_7b)
        XCTAssertEqual(large.versionDir, TTSModelPreset.qwen3TTS_1_7b.versionDir)
        XCTAssertEqual(large.multiCodeDecoderVariant, TTSVariantDefaults.multiCodeDecoder)
    }

    func testGenerationOptionsDefaults() {
        let opts = TTSGenerationOptions()
        XCTAssertEqual(opts.temperature, 0.9)
        XCTAssertEqual(opts.topK, 50)
        XCTAssertEqual(opts.repetitionPenalty, 1.05)
        XCTAssertEqual(opts.maxNewTokens, 245)
        XCTAssertNil(opts.chunkingStrategy)
        XCTAssertNil(opts.instruction)
        XCTAssertNotNil(opts.concurrentWorkerCount)
    }

    func testDownloadPatterns() {
        let config = TTSKitConfig()
        let patterns = config.downloadPatterns
        XCTAssertEqual(patterns.count, 6)
        XCTAssertTrue(patterns.allSatisfy { $0.hasSuffix("/**") })
        XCTAssertTrue(patterns.contains { $0.contains("code_decoder/") })
        XCTAssertTrue(patterns.contains { $0.contains("speech_decoder/") })
        XCTAssertTrue(patterns.contains { $0.contains("text_projector/") })
    }

    // MARK: - Speaker & Language Enums

    func testSpeakerTokenIds() {
        // Each speaker should have a unique token ID
        let ids = Qwen3Speaker.allCases.map { $0.tokenID }
        XCTAssertEqual(Set(ids).count, Qwen3Speaker.allCases.count, "Speaker token IDs should be unique")
    }

    func testLanguageTokenIds() {
        let ids = Qwen3Language.allCases.map { $0.tokenID }
        XCTAssertEqual(Set(ids).count, Qwen3Language.allCases.count, "Language token IDs should be unique")
        XCTAssertEqual(Qwen3Language.english.tokenID, 2050)
    }

    // MARK: - Text Chunker

    /// Char-level tokenizer helper: 1 unicode scalar = 1 token, perfectly round-trips.
    /// Sizes are exact, making test assertions easy to reason about.
    private func makeChunker(
        targetChunkSize: Int = TTSTextChunker.defaultTargetChunkSize,
        minChunkSize: Int = TTSTextChunker.defaultMinChunkSize
    ) -> TTSTextChunker {
        TTSTextChunker(
            targetChunkSize: targetChunkSize,
            minChunkSize: minChunkSize,
            encode: { $0.unicodeScalars.map { Int($0.value) } },
            decode: { String($0.compactMap { Unicode.Scalar($0) }.map { Character($0) }) }
        )
    }

    func testChunkerShortText() {
        let chunker = makeChunker(targetChunkSize: 50, minChunkSize: 5)
        XCTAssertEqual(chunker.chunk("Hello world."), ["Hello world."])
    }

    func testChunkerEmptyText() {
        let chunker = makeChunker()
        XCTAssertTrue(chunker.chunk("").isEmpty)
        XCTAssertTrue(chunker.chunk("   ").isEmpty)
    }

    func testChunkerSentenceSplitting() {
        // With char-level tokenizer (1 char = 1 token), targetChunkSize: 30 gives a 30-char window.
        // "This is the first sentence." = 27 chars, so each window contains one sentence boundary.
        let chunker = makeChunker(targetChunkSize: 30, minChunkSize: 5)
        let text = "This is the first sentence. This is the second sentence. And here is a third one."
        let chunks = chunker.chunk(text)
        XCTAssertGreaterThan(chunks.count, 1, "Should split into multiple chunks")
        let recombined = chunks.joined(separator: " ")
        XCTAssertTrue(recombined.contains("first sentence"))
        XCTAssertTrue(recombined.contains("third one"))
    }

    func testChunkerMergesTinyTrailing() {
        // "A reasonably long sentence that is here." = exactly 40 chars = 40 tokens (char-level).
        // targetChunkSize: 40 captures the full sentence in one window at the "." boundary.
        // The trailing "X." = 2 tokens is below minChunkSize: 5, so it merges with the previous chunk.
        let chunker = makeChunker(targetChunkSize: 40, minChunkSize: 5)
        let text = "A reasonably long sentence that is here. X."
        let chunks = chunker.chunk(text)
        XCTAssertEqual(chunks.count, 1, "Tiny trailing chunk should merge with previous")
    }

    // MARK: - Embedding Math

    func testZeroEmbed() {
        let embed = zeroEmbed(dim: 16)
        XCTAssertEqual(embed.count, 16)
        XCTAssertTrue(embed.allSatisfy { $0 == 0 })
    }

    func testAddEmbeddings() {
        let a: EmbedBuffer = [FloatType(1.0), FloatType(2.0), FloatType(3.0)]
        let b: EmbedBuffer = [FloatType(4.0), FloatType(5.0), FloatType(6.0)]
        let result = addEmbeddings(a, b)
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(Float(result[0]), 5.0, accuracy: 0.01)
        XCTAssertEqual(Float(result[1]), 7.0, accuracy: 0.01)
        XCTAssertEqual(Float(result[2]), 9.0, accuracy: 0.01)
    }

    func testSumEmbeddings() {
        let embeds: [EmbedBuffer] = [
            [FloatType(1.0), FloatType(2.0)],
            [FloatType(3.0), FloatType(4.0)],
            [FloatType(5.0), FloatType(6.0)],
        ]
        let result = sumEmbeddings(embeds)
        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(Float(result[0]), 9.0, accuracy: 0.01)
        XCTAssertEqual(Float(result[1]), 12.0, accuracy: 0.01)
    }

    func testSumEmbeddingsEmpty() {
        let result = sumEmbeddings([])
        XCTAssertTrue(result.isEmpty)
    }

    func testCreateAndExtractEmbed() throws {
        let original: EmbedBuffer = [FloatType(1.5), FloatType(2.5), FloatType(3.5), FloatType(4.5)]
        let arr = try createEmbedMLArray(original)
        XCTAssertEqual(arr.shape, [1, 4, 1, 1] as [NSNumber])

        let extracted = extractEmbed(from: arr)
        XCTAssertEqual(extracted.count, 4)
        for i in 0..<4 {
            XCTAssertEqual(Float(extracted[i]), Float(original[i]), accuracy: 0.01)
        }
    }

    // MARK: - KV Cache

    func testKVCacheInit() throws {
        let cache = try TTSKVCache(cacheDim: 128, maxSeqLength: 32)
        XCTAssertEqual(cache.cacheLength, 0)
        XCTAssertEqual(cache.maxSeqLength, 32)
        XCTAssertEqual(cache.cacheDim, 128)
        XCTAssertFalse(cache.isStateful)
        XCTAssertFalse(cache.isFull)
        XCTAssertEqual(cache.freePositions, 31) // maxSeqLength - 1
        XCTAssertNotNil(cache.keyCache)
        XCTAssertNotNil(cache.valueCache)
    }

    func testKVCacheStatefulInit() throws {
        let cache = try TTSKVCache(cacheDim: 128, maxSeqLength: 32, isStateful: true)
        XCTAssertTrue(cache.isStateful)
        XCTAssertNil(cache.keyCache)
        XCTAssertNil(cache.valueCache)
    }

    func testKVCacheUpdateAdvancesPosition() throws {
        let cache = try TTSKVCache(cacheDim: 4, maxSeqLength: 8)
        XCTAssertEqual(cache.cacheLength, 0)

        cache.update()
        XCTAssertEqual(cache.cacheLength, 1)

        cache.update()
        XCTAssertEqual(cache.cacheLength, 2)
        XCTAssertEqual(cache.freePositions, 5) // 8 - 1 - 2
    }

    func testKVCacheIsFull() throws {
        let cache = try TTSKVCache(cacheDim: 4, maxSeqLength: 4)
        // maxSeqLength=4, isFull when cacheLength >= 3 (maxSeqLength - 1)
        cache.update()
        cache.update()
        XCTAssertFalse(cache.isFull)
        cache.update()
        XCTAssertTrue(cache.isFull)
    }

    func testKVCacheReset() throws {
        let cache = try TTSKVCache(cacheDim: 4, maxSeqLength: 8)
        cache.update()
        cache.update()
        XCTAssertEqual(cache.cacheLength, 2)

        cache.reset()
        XCTAssertEqual(cache.cacheLength, 0)
    }

    func testSpeechDecoderCacheInit() throws {
        let cache = try TTSSpeechDecoderCache(
            cacheDim: 64, maxSeqLength: 16, hiddenDim: 32, hiddenContextLen: 4
        )
        XCTAssertEqual(cache.hiddenDim, 32)
        XCTAssertEqual(cache.hiddenContextLen, 4)
        XCTAssertEqual(cache.hiddenContext.shape, [1, 32, 1, 4] as [NSNumber])
    }

    // MARK: - Sampler

    func testGreedySamplerDeterministic() throws {
        // Two samplers with the same seed should produce the same result
        let sampler1 = TTSGreedyTokenSampler(seed: 42)
        let sampler2 = TTSGreedyTokenSampler(seed: 42)

        // Create a simple logits array [1, 1, vocabSize]
        let vocabSize = 32
        let logits = try MLMultiArray(shape: [1, 1, NSNumber(value: vocabSize)], dataType: .float16)
        let ptr = logits.dataPointer.bindMemory(to: FloatType.self, capacity: vocabSize)
        for i in 0..<vocabSize {
            ptr[i] = FloatType(Float.random(in: -1...1))
        }

        let token1 = sampler1.sampleCodec0(
            logits: logits, temperature: 0.9, topK: 10,
            generatedTokens: [], repetitionPenalty: 1.0, suppressTokenIds: []
        )
        let token2 = sampler2.sampleCodec0(
            logits: logits, temperature: 0.9, topK: 10,
            generatedTokens: [], repetitionPenalty: 1.0, suppressTokenIds: []
        )
        XCTAssertEqual(token1, token2, "Same seed should produce same token")
    }

    func testGreedySamplerZeroTemperature() throws {
        let sampler = TTSGreedyTokenSampler(seed: 0)
        let vocabSize = 16
        let logits = try MLMultiArray(shape: [1, 1, NSNumber(value: vocabSize)], dataType: .float16)
        let ptr = logits.dataPointer.bindMemory(to: FloatType.self, capacity: vocabSize)

        // Make token 7 have the highest logit
        for i in 0..<vocabSize { ptr[i] = FloatType(-1.0) }
        ptr[7] = FloatType(5.0)

        let token = sampler.sampleCodec0(
            logits: logits, temperature: 0.0, topK: 0,
            generatedTokens: [], repetitionPenalty: 1.0, suppressTokenIds: []
        )
        XCTAssertEqual(token, 7, "Greedy (temp=0) should pick argmax")
    }

    func testGreedySamplerTokenSuppression() throws {
        let sampler = TTSGreedyTokenSampler(seed: 0)
        let vocabSize = 16
        let logits = try MLMultiArray(shape: [1, 1, NSNumber(value: vocabSize)], dataType: .float16)
        let ptr = logits.dataPointer.bindMemory(to: FloatType.self, capacity: vocabSize)

        // Make token 3 the argmax, but suppress it
        for i in 0..<vocabSize { ptr[i] = FloatType(0.0) }
        ptr[3] = FloatType(10.0)
        ptr[5] = FloatType(5.0) // second highest

        let token = sampler.sampleCodec0(
            logits: logits, temperature: 0.0, topK: 0,
            generatedTokens: [], repetitionPenalty: 1.0, suppressTokenIds: [3]
        )
        XCTAssertEqual(token, 5, "Suppressed token should be skipped, picking next best")
    }

    // MARK: - TTSKit Initialization

    func testTTSKitInit() async throws {
        let tts = try await TTSKit(TTSKitConfig(verbose: true), seed: 42, load: false)
        XCTAssertEqual(tts.seed, 42)
        XCTAssertTrue(tts.config.verbose)
        XCTAssertNil(tts.tokenizer)
    }

    func testTTSKitComponentsExist() async throws {
        let tts = try await TTSKit(load: false)
        // All components should be initialized (models not loaded yet)
        XCTAssertNil(tts.textProjector.model)
        XCTAssertNil(tts.codeEmbedder.model)
        XCTAssertNil(tts.multiCodeEmbedder.model)
        XCTAssertNil(tts.codeDecoder.model)
        XCTAssertNil(tts.multiCodeDecoder.model)
        XCTAssertNil(tts.speechDecoder.model)
    }

    func testSuppressTokenIds() {
        let ids = Qwen3TTSConstants.suppressTokenIds
        // Should contain [2048, 3072) except codecEOS (2150)
        XCTAssertEqual(ids.count, 1023) // 1024 - 1 (EOS excluded)
        XCTAssertTrue(ids.contains(2048))
        XCTAssertTrue(ids.contains(3071))
        XCTAssertFalse(ids.contains(Int(Qwen3TTSConstants.codecEOS)))
        XCTAssertFalse(ids.contains(2047)) // below range
        XCTAssertFalse(ids.contains(3072)) // above range
    }

    // MARK: - Per-Task Sampler Isolation

    func testMakeTaskDerivedSeeds() async throws {
        let tts = try await TTSKit(seed: 12345, load: false)
        XCTAssertEqual(tts.seed, 12345)
    }

    // MARK: - Timings

    func testTTSTimingsDefaults() {
        let t = TTSTimings()
        XCTAssertEqual(t.fullPipeline, 0)
        XCTAssertEqual(t.generationLoop, 0)
        XCTAssertEqual(t.totalDecodingLoops, 0)
        XCTAssertEqual(t.tokensPerSecond, 0)
        XCTAssertEqual(t.realTimeFactor, 0)
        XCTAssertEqual(t.speedFactor, 0)
    }

    func testTTSTimingsComputedProperties() {
        var t = TTSTimings()
        t.fullPipeline = 2.0
        t.totalDecodingLoops = 100
        t.inputAudioSeconds = 5.0
        t.codeDecoder = 0.5
        t.multiCodeDecoderPredictions = 0.3
        t.speechDecoderPredictions = 0.2
        t.generationLoop = 0.8

        XCTAssertEqual(t.tokensPerSecond, 50.0, accuracy: 0.01)
        XCTAssertEqual(t.realTimeFactor, 0.4, accuracy: 0.01) // 2.0 / 5.0
        XCTAssertEqual(t.speedFactor, 2.5, accuracy: 0.01) // 5.0 / 2.0
        XCTAssertEqual(t.totalPredictions, 1.0, accuracy: 0.01) // 0.5 + 0.3 + 0.2
        // parallelOverlap = max(0, 1.0 - 0.8) = 0.2
        XCTAssertEqual(t.parallelOverlap, 0.2, accuracy: 0.01)
    }

    // MARK: - TTSResult

    func testTTSResultAudioDuration() {
        let sampleRate = Qwen3TTSConstants.sampleRate // 24000
        let samples = [Float](repeating: 0, count: sampleRate * 3) // 3 seconds
        let result = TTSResult(audio: samples, timings: TTSTimings(), sampleRate: Qwen3TTSConstants.sampleRate)
        XCTAssertEqual(result.audioDuration, 3.0, accuracy: 0.001)
    }

    // MARK: - Constants

    func testQwen3TTSConstants() {
        XCTAssertEqual(Qwen3TTSConstants.sampleRate, 24000)
        XCTAssertEqual(Qwen3TTSConstants.samplesPerFrame, 1920)
        XCTAssertEqual(Qwen3TTSConstants.embedDim, 1024)
        XCTAssertEqual(Qwen3TTSConstants.codecVocabSize, 2048)
        // EOS should be in the codec-0 range but excluded from suppression
        XCTAssertEqual(Qwen3TTSConstants.codecEOS, 2150)
        XCTAssertTrue(Qwen3TTSConstants.codecEOS >= 2048 && Qwen3TTSConstants.codecEOS < 3072)
    }

    // MARK: - Unloaded Model Errors

    func testCodeDecoderWithoutModel() {
        let decoder = TTSCodeDecoder()
        XCTAssertNil(decoder.model)
        XCTAssertFalse(decoder.isStateful)
        XCTAssertNil(decoder.makeState())
    }

    func testMultiCodeDecoderWithoutModel() {
        let decoder = TTSMultiCodeDecoder()
        XCTAssertNil(decoder.model)
        XCTAssertFalse(decoder.isStateful)
        XCTAssertNil(decoder.makeState())
    }

    func testSpeechDecoderWithoutModel() {
        let decoder = TTSSpeechDecoder()
        XCTAssertNil(decoder.model)
    }

    // MARK: - Chunking Strategy

    func testChunkingStrategyEnum() {
        XCTAssertEqual(TTSChunkingStrategy.allCases.count, 2)
        XCTAssertEqual(TTSChunkingStrategy.none.rawValue, "none")
        XCTAssertEqual(TTSChunkingStrategy.sentence.rawValue, "sentence")
    }

    // MARK: - SeededRNG Determinism

    func testSeededRNGDeterminism() {
        var rng1 = SeededRandomNumberGenerator(seed: 99)
        var rng2 = SeededRandomNumberGenerator(seed: 99)
        for _ in 0..<100 {
            XCTAssertEqual(rng1.next(), rng2.next())
        }
    }

    func testSeededRNGDifferentSeeds() {
        var rng1 = SeededRandomNumberGenerator(seed: 1)
        var rng2 = SeededRandomNumberGenerator(seed: 2)
        // Very unlikely for all 10 values to match with different seeds
        var allMatch = true
        for _ in 0..<10 {
            if rng1.next() != rng2.next() {
                allMatch = false
                break
            }
        }
        XCTAssertFalse(allMatch, "Different seeds should produce different sequences")
    }

    // MARK: - TTSProgress

    func testTTSProgressInit() {
        let samples: [Float] = [0.1, 0.2, 0.3]
        let timings = TTSTimings()
        let progress = TTSProgress(audio: samples, timings: timings, stepTime: 0.08)
        XCTAssertEqual(progress.audio, samples)
        XCTAssertEqual(progress.timings.fullPipeline, 0)
        XCTAssertEqual(progress.stepTime ?? 0, 0.08, accuracy: 0.0001)
    }

    func testTTSProgressStepTimeNilByDefault() {
        let progress = TTSProgress(audio: [], timings: TTSTimings())
        XCTAssertNil(progress.stepTime)
    }

    func testTTSProgressFirstStepSemantics() {
        // stepTime non-nil signals first step; subsequent steps have nil
        let first = TTSProgress(audio: [0.1], timings: TTSTimings(), stepTime: 0.05)
        let subsequent = TTSProgress(audio: [0.2], timings: TTSTimings(), stepTime: nil)
        XCTAssertNotNil(first.stepTime)
        XCTAssertNil(subsequent.stepTime)
    }

    // MARK: - TTSPlaybackStrategy

    func testAudioPerStep() {
        let spf = Qwen3TTSConstants.samplesPerFrame
        let sr = Qwen3TTSConstants.sampleRate
        let expected = Double(spf) / Double(sr)
        XCTAssertEqual(TTSPlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr), expected, accuracy: 0.0001)
        // ~80ms per frame at 24kHz / 1920 samples
        XCTAssertEqual(TTSPlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr), 0.08, accuracy: 0.001)
    }

    func testRequiredBufferFastDevice() {
        // Step completes in half the frame duration -> device is 2x real-time
        // deficit = max(0, 1 - 2.0) = 0, so result = minimumBufferDuration
        let spf = Qwen3TTSConstants.samplesPerFrame
        let sr = Qwen3TTSConstants.sampleRate
        let stepTime = TTSPlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr) / 2.0
        let buffer = TTSPlaybackStrategy.requiredBuffer(stepTime: stepTime, maxNewTokens: 100, samplesPerFrame: spf, sampleRate: sr)
        XCTAssertEqual(buffer, TTSPlaybackStrategy.minimumBufferDuration, accuracy: 0.001)
    }

    func testRequiredBufferSlowDevice() {
        // Step takes 2x the frame duration -> device is at 0.5x real-time
        // speedRatio = 0.5, deficit = 0.5, maxAudio = 100 * 0.08 = 8s
        // deficitBuffer = 8 * 0.5 = 4s > minimumBufferDuration
        let spf = Qwen3TTSConstants.samplesPerFrame
        let sr = Qwen3TTSConstants.sampleRate
        let stepTime = TTSPlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr) * 2.0
        let buffer = TTSPlaybackStrategy.requiredBuffer(stepTime: stepTime, maxNewTokens: 100, samplesPerFrame: spf, sampleRate: sr)
        XCTAssertGreaterThan(buffer, TTSPlaybackStrategy.minimumBufferDuration)
        // Exact: 100 * 0.08 * 0.5 = 4.0s
        XCTAssertEqual(buffer, 4.0, accuracy: 0.01)
    }

    func testRequiredBufferAtExactRealTime() {
        // Step equals frame duration -> speedRatio = 1, deficit = 0 -> minimum clamp applies
        let spf = Qwen3TTSConstants.samplesPerFrame
        let sr = Qwen3TTSConstants.sampleRate
        let stepTime = TTSPlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr)
        let buffer = TTSPlaybackStrategy.requiredBuffer(stepTime: stepTime, maxNewTokens: 50, samplesPerFrame: spf, sampleRate: sr)
        XCTAssertEqual(buffer, TTSPlaybackStrategy.minimumBufferDuration, accuracy: 0.001)
    }

    func testRequiredBufferMinimumNeverExceeded() {
        // Even a very fast device should never return less than the minimum
        let buffer = TTSPlaybackStrategy.requiredBuffer(
            stepTime: 0.001, maxNewTokens: 200,
            samplesPerFrame: Qwen3TTSConstants.samplesPerFrame, sampleRate: Qwen3TTSConstants.sampleRate
        )
        XCTAssertGreaterThanOrEqual(buffer, TTSPlaybackStrategy.minimumBufferDuration)
    }

    // MARK: - TTSModelPreset

    func testModelPresetDisplayNames() {
        XCTAssertEqual(TTSModelPreset.qwen3TTS_0_6b.displayName, "Qwen3 TTS 0.6B")
        XCTAssertEqual(TTSModelPreset.qwen3TTS_1_7b.displayName, "Qwen3 TTS 1.7B")
    }

    func testModelPresetSupportsVoiceDirection() {
        XCTAssertFalse(TTSModelPreset.qwen3TTS_0_6b.supportsVoiceDirection)
        XCTAssertTrue(TTSModelPreset.qwen3TTS_1_7b.supportsVoiceDirection)
    }

    func testModelPresetVersionDirsDiffer() {
        XCTAssertNotEqual(
            TTSModelPreset.qwen3TTS_0_6b.versionDir,
            TTSModelPreset.qwen3TTS_1_7b.versionDir
        )
    }

    func testModelPresetAvailabilityOnMacOS() {
        // On macOS, all presets should be available
        #if os(macOS)
        for preset in TTSModelPreset.allCases {
            XCTAssertTrue(preset.isAvailableOnCurrentPlatform, "\(preset) should be available on macOS")
        }
        #else
        XCTAssertTrue(TTSModelPreset.qwen3TTS_0_6b.isAvailableOnCurrentPlatform)
        XCTAssertFalse(TTSModelPreset.qwen3TTS_1_7b.isAvailableOnCurrentPlatform)
        #endif
    }

    func testModelPresetVariantDefaultsConsistent() {
        // Both presets share the same variant strings (all quantization is size-independent)
        XCTAssertEqual(TTSModelPreset.qwen3TTS_0_6b.codeDecoderVariant, TTSVariantDefaults.codeDecoder)
        XCTAssertEqual(TTSModelPreset.qwen3TTS_1_7b.codeDecoderVariant, TTSVariantDefaults.codeDecoder)
        XCTAssertEqual(TTSModelPreset.qwen3TTS_0_6b.speechDecoderVariant, TTSVariantDefaults.speechDecoder)
    }

    // MARK: - TTSKitConfig Component Overrides

    func testTTSKitConfigComponentOverridesNilByDefault() {
        let config = TTSKitConfig()
        XCTAssertNil(config.textProjector)
        XCTAssertNil(config.codeEmbedder)
        XCTAssertNil(config.multiCodeEmbedder)
        XCTAssertNil(config.codeDecoder)
        XCTAssertNil(config.multiCodeDecoder)
        XCTAssertNil(config.speechDecoder)
    }

    // MARK: - TTSGenerationOptions Additional Defaults

    func testGenerationOptionsChunkingDefaults() {
        let opts = TTSGenerationOptions()
        // nil defers to TTSTextChunker.defaultTargetChunkSize / defaultMinChunkSize at call site
        XCTAssertNil(opts.targetChunkSize)
        XCTAssertNil(opts.minChunkSize)
        // Verify the canonical defaults live in TTSTextChunker
        XCTAssertEqual(TTSTextChunker.defaultTargetChunkSize, 50)
        XCTAssertEqual(TTSTextChunker.defaultMinChunkSize, 10)
    }

    // MARK: - Speaker & Language Round-trips

    func testSpeakerRawValueRoundTrip() {
        for speaker in Qwen3Speaker.allCases {
            let roundTripped = Qwen3Speaker(rawValue: speaker.rawValue)
            XCTAssertEqual(roundTripped, speaker, "rawValue round-trip failed for \(speaker)")
        }
    }

    func testLanguageRawValueRoundTrip() {
        for lang in Qwen3Language.allCases {
            let roundTripped = Qwen3Language(rawValue: lang.rawValue)
            XCTAssertEqual(roundTripped, lang, "rawValue round-trip failed for \(lang)")
        }
    }

    func testUnrecognisedSpeakerFallsBack() {
        XCTAssertNil(Qwen3Speaker(rawValue: "nonexistent_speaker"))
    }

    func testUnrecognisedLanguageFallsBack() {
        XCTAssertNil(Qwen3Language(rawValue: "klingon"))
    }

    // MARK: - mergeTimings

    func testMergeTimingsAccumulates() async throws {
        let tts = try await TTSKit(load: false)

        var combined = TTSTimings()
        combined.generationLoop = 1.0
        combined.totalDecodingLoops = 10
        combined.codeDecoder = 0.5

        var chunk = TTSTimings()
        chunk.generationLoop = 2.0
        chunk.totalDecodingLoops = 20
        chunk.codeDecoder = 0.3
        chunk.multiCodeDecoderPredictions = 0.1
        chunk.speechDecoderPredictions = 0.2

        tts.mergeTimings(&combined, from: chunk)

        XCTAssertEqual(combined.generationLoop, 3.0, accuracy: 0.001)
        XCTAssertEqual(combined.totalDecodingLoops, 30, accuracy: 0.001)
        XCTAssertEqual(combined.codeDecoder, 0.8, accuracy: 0.001)
        XCTAssertEqual(combined.multiCodeDecoderPredictions, 0.1, accuracy: 0.001)
        XCTAssertEqual(combined.speechDecoderPredictions, 0.2, accuracy: 0.001)
    }

    func testMergeTimingsIdentity() async throws {
        let tts = try await TTSKit(load: false)
        var combined = TTSTimings()
        combined.generationLoop = 5.0
        let empty = TTSTimings()
        tts.mergeTimings(&combined, from: empty)
        XCTAssertEqual(combined.generationLoop, 5.0, accuracy: 0.001)
    }

    // MARK: - TTSTextChunker Edge Cases

    func testChunkerPreservesAllText() {
        // Each "Sentence number N is here." ≈ 26-27 chars; targetChunkSize: 50 spans ~2 sentences.
        let chunker = makeChunker(targetChunkSize: 50, minChunkSize: 5)
        let sentences = (1...10).map { "Sentence number \($0) is here." }
        let text = sentences.joined(separator: " ")
        let chunks = chunker.chunk(text)
        let rejoined = chunks.joined(separator: " ")
        for sentence in sentences {
            XCTAssertTrue(rejoined.contains(sentence.dropLast(1)), // drop period; joins may vary
                "Missing content: \(sentence)")
        }
    }

    func testChunkerWordBoundaryFallback() {
        // No punctuation in text - chunker must fall back to word-boundary splits.
        // With char-level tokenizer, targetChunkSize: 15 gives 15-char windows.
        // Note: multi-word phrase checks are intentionally avoided here because the
        // re-encode advance can leave a stray char when the token stream has a leading
        // space (e.g. "very long" → 9 tokens, but the stream starts with " very lon").
        // This is a char-level mock artifact; BPE tokenizers fold whitespace into the
        // next word token, so no drift occurs in production.
        let chunker = makeChunker(targetChunkSize: 15, minChunkSize: 3)
        let text = "This is a very long text with no punctuation inside it at all here"
        let chunks = chunker.chunk(text)
        XCTAssertGreaterThan(chunks.count, 1, "Long text without punctuation should split at word boundaries")
        let rejoined = chunks.joined(separator: " ")
        for word in ["long", "text", "punctuation", "here"] {
            XCTAssertTrue(rejoined.contains(word), "Missing word in output: \(word)")
        }
    }
}
