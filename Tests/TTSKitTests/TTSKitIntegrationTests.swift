//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import CoreML
import Foundation
@testable import TTSKit
import XCTest

// MARK: - Helpers

extension XCTestCase {
    /// Create a `TTSKit` instance for integration tests.
    ///
    /// Downloads models from the Hub if not already cached on disk, matching the
    /// same pattern as `tinyModelPath()` in WhisperKit's test suite. The test
    /// will fail (not skip) if the download fails.
    func makeCachedTTS(
        model: TTSModelPreset = .qwen3TTS_0_6b,
        seed: UInt64 = 42
    ) async throws -> TTSKit {
        let config = TTSKitConfig(model: model, verbose: true, logLevel: .debug)
        return try await TTSKit(config, seed: seed)
    }
}

// MARK: - Integration Tests

final class TTSKitIntegrationTests: XCTestCase {

    // MARK: - Basic generation

    /// Short sentence -> non-empty audio at 24kHz.
    func testBasicShortGeneration() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let result = try await tts.generateSpeech(
            text: "Hello, this is a basic smoke test of the WhisperKit TTS pipeline.",
            speaker: .ryan,
            language: .english,
            options: TTSGenerationOptions(temperature: 0.9, topK: 50, maxNewTokens: 245)
        )

        XCTAssertGreaterThan(result.audio.count, 0, "Audio samples should be non-empty")
        XCTAssertGreaterThan(result.audioDuration, 1.0, "Expect at least 1s of speech")
        XCTAssertLessThan(result.audioDuration, 30.0, "Short sentence should stay under 30s")
        XCTAssertEqual(result.sampleRate, 24000, "Sample rate should be 24kHz")
    }

    // MARK: - Long text with sentence chunking

    /// 200+ word multi-paragraph text - chunked, all audio produced, no truncation crash.
    func testLongTextSequentialChunks() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let longText = """
        The history of artificial intelligence begins in antiquity, with myths, stories, \
        and rumors of artificial beings endowed with intelligence by master craftsmen. \
        The seeds of modern AI were planted by classical philosophers who attempted to \
        describe the process of human thinking as the mechanical manipulation of symbols. \
        In 1950, Alan Turing published a landmark paper in which he speculated about the \
        possibility of creating machines that think. Fast forward to the 2020s, and large \
        language models have transformed every domain from science to storytelling. \
        Text-to-speech systems now produce voices so natural that listeners struggle to \
        distinguish them from real human speakers.
        """

        let result = try await tts.generateSpeech(
            text: longText,
            speaker: .aiden,
            language: .english,
            options: TTSGenerationOptions(concurrentWorkerCount: 1)
        )

        XCTAssertGreaterThan(result.audioDuration, 20.0, "Long text should produce substantial audio")
        XCTAssertGreaterThan(result.timings.totalDecodingLoops, 100, "Should require many decode steps")
    }

    /// Same long text with unlimited concurrent workers
    func testLongTextUnlimitedConcurrentWorkers() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let longText = """
        First paragraph about science. It covers many interesting topics in depth. \
        Second paragraph switches to history. Many events happened over the centuries. \
        Third paragraph explores music theory. Harmony and rhythm define the art form.
        """

        let result = try await tts.generateSpeech(
            text: longText,
            speaker: .aiden,
            language: .english,
            options: TTSGenerationOptions(concurrentWorkerCount: nil)
        )

        XCTAssertGreaterThan(result.audioDuration, 5.0)
    }

    // MARK: - Concurrent workers

    /// workers=2 on a 3-chunk text - verifies the batching loop handles partial batches.
    func testConcurrentWorkersBatching() async throws {
        let tts = try await makeCachedTTS(seed: 7)
        let threeChunkText = """
        First sentence for chunk one. Second sentence for chunk two. Third sentence for chunk three.
        """
        let result = try await tts.generateSpeech(
            text: threeChunkText,
            speaker: .serena,
            language: .english,
            options: TTSGenerationOptions(
                concurrentWorkerCount: 2,
                targetChunkSize: 10,
                minChunkSize: 2
            )
        )

        XCTAssertGreaterThan(result.audio.count, 0)
        XCTAssertGreaterThan(result.audioDuration, 2.0)
    }

    // MARK: - Reproducibility

    /// Two runs with the same seed and temperature=0 should produce byte-identical audio.
    func testDeterministicOutputWithFixedSeed() async throws {
        let text = "Reproducibility test with a fixed seed."
        let options = TTSGenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 100)

        let tts1 = try await makeCachedTTS(seed: 42)
        let result1 = try await tts1.generateSpeech(
            text: text, speaker: .ryan, language: .english, options: options
        )

        let tts2 = try await makeCachedTTS(seed: 42)
        let result2 = try await tts2.generateSpeech(
            text: text, speaker: .ryan, language: .english, options: options
        )

        XCTAssertEqual(result1.audio.count, result2.audio.count, "Step counts must match")
        // Compare raw float bytes for strict reproducibility
        let data1 = Data(bytes: result1.audio, count: result1.audio.count * MemoryLayout<Float>.size)
        let data2 = Data(bytes: result2.audio, count: result2.audio.count * MemoryLayout<Float>.size)
        XCTAssertEqual(data1, data2, "Audio samples must be byte-identical with the same seed and greedy decoding")
    }

    // MARK: - Speaker voices

    /// All 9 Qwen3Speaker voices should produce non-empty audio without error.
    func testAllSpeakerVoices() async throws {
        let tts = try await makeCachedTTS(seed: 1)
        let text = "Testing this voice."
        let options = TTSGenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 60)

        for speaker in Qwen3Speaker.allCases {
            let result = try await tts.generateSpeech(
                text: text, speaker: speaker, language: .english, options: options
            )
            XCTAssertGreaterThan(result.audio.count, 0, "Speaker \(speaker.rawValue) produced no audio")
        }
    }

    // MARK: - Language support

    /// Korean text with a Korean-native speaker should produce non-empty audio.
    func testKoreanLanguage() async throws {
        let tts = try await makeCachedTTS(seed: 7)
        let result = try await tts.generateSpeech(
            text: "안녕하세요. 저는 한국어를 말할 수 있습니다.",
            speaker: .sohee,
            language: .korean,
            options: TTSGenerationOptions(temperature: 0.9, maxNewTokens: 120)
        )

        XCTAssertGreaterThan(result.audio.count, 0)
        XCTAssertGreaterThan(result.audioDuration, 0.5)
    }

    // MARK: - Edge cases

    /// Single-word input should not crash and should produce short audio.
    func testSingleWordInput() async throws {
        let tts = try await makeCachedTTS(seed: 1)
        let result = try await tts.generateSpeech(
            text: "Hello.",
            speaker: .ryan,
            language: .english,
            options: TTSGenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 60)
        )

        XCTAssertGreaterThan(result.audio.count, 0)
        XCTAssertLessThan(result.audioDuration, 5.0, "Single word should be short")
    }

    /// Text with numbers, punctuation, currency, and percentages should not crash.
    func testNumbersAndPunctuation() async throws {
        let tts = try await makeCachedTTS(seed: 3)
        let result = try await tts.generateSpeech(
            text: "On January 1st, 2025, the price was $4.99 - a 12% discount from $5.67.",
            speaker: .dylan,
            language: .english,
            options: TTSGenerationOptions(maxNewTokens: 245)
        )

        XCTAssertGreaterThan(result.audio.count, 0)
    }

    /// Text with emoji and mixed scripts (Latin + accented) should not crash.
    func testUnicodeAndEmoji() async throws {
        let tts = try await makeCachedTTS(seed: 5)
        let result = try await tts.generateSpeech(
            text: "Today's meeting is at 3pm 🎉. The café serves café au lait. Résumé updated.",
            speaker: .ryan,
            language: .english,
            options: TTSGenerationOptions(maxNewTokens: 200)
        )

        XCTAssertGreaterThan(result.audio.count, 0)
    }

    // MARK: - WAV export

    /// saveAudio round-trip: write WAV then verify the file exists and is readable.
    func testSaveAudioRoundTrip() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let result = try await tts.generateSpeech(
            text: "Audio export round trip test.",
            speaker: .ryan,
            language: .english,
            options: TTSGenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 80)
        )

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("tts_roundtrip_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        try TTSAudioOutput.saveAudio(result.audio, to: tmpURL, sampleRate: result.sampleRate)

        XCTAssertTrue(FileManager.default.fileExists(atPath: tmpURL.path), "WAV file should exist after saveAudio")

        let duration = try await TTSAudioOutput.duration(of: tmpURL)
        XCTAssertEqual(duration, result.audioDuration, accuracy: 0.1, "WAV duration should match result.audioDuration")
    }

    // MARK: - TTSModel protocol

    /// TTSKit should satisfy the TTSModel protocol; generate via the protocol entry point.
    func testTTSModelProtocolConformance() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let model: any TTSModel = tts

        XCTAssertEqual(model.sampleRate, 24000)

        let result = try await model.generate(
            text: "Protocol conformance check.",
            voice: Qwen3Speaker.ryan.rawValue,
            language: Qwen3Language.english.rawValue,
            options: TTSGenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 80),
            callback: nil
        )

        XCTAssertGreaterThan(result.audio.count, 0)
        XCTAssertEqual(result.sampleRate, 24000)
    }

    // MARK: - Performance

    /// Verify timings are populated and the generation loop completed within a reasonable ceiling.
    func testTimingsArePopulated() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let result = try await tts.generateSpeech(
            text: "Performance check: this sentence measures how fast the model runs on device.",
            speaker: .ryan,
            language: .english,
            options: TTSGenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 200)
        )

        XCTAssertGreaterThan(result.timings.totalDecodingLoops, 0, "Should have completed at least one decode step")
        XCTAssertGreaterThan(result.timings.fullPipeline, 0, "Full pipeline duration should be non-zero")
        XCTAssertGreaterThan(result.audioDuration, 0, "Audio duration should be non-zero")
    }
}
