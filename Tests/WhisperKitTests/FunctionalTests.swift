//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
@testable import WhisperKit
import XCTest

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
final class FunctionalTests: XCTestCase {
    func testInitLarge() async throws {
        try await XCTAssertNoThrowAsync(
            await WhisperKit(modelFolder: largev3ModelPath(), logLevel: .error)
        )
    }

    func testOutputAll() async throws {
        let modelPaths = try allModelPaths()

        for modelPath in modelPaths {
            let modelName = modelPath.split(separator: "/").last!
            print("[Integration] Testing model \(modelName)")
            let audioFilePath = try XCTUnwrap(
                Bundle.module.path(forResource: "jfk", ofType: "wav"),
                "Audio file not found"
            )

            let whisperKit = try await WhisperKit(
                modelFolder: modelPath,
                verbose: true,
                logLevel: .debug
            )

            let transcriptionResult: [TranscriptionResult] = try await whisperKit.transcribe(audioPath: audioFilePath)
            let transcriptionResultText = transcriptionResult.text

            print("[Integration] \(transcriptionResultText)")
            XCTAssertEqual(
                transcriptionResultText.normalized,
                " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.".normalized,
                "Transcription result does not match expected result for model \(modelName)"
            )
        }
    }

    func testRealTimeFactorTiny() async throws {
        let modelPath = try tinyModelPath()

        let metrics: [XCTMetric] = [XCTMemoryMetric(), XCTStorageMetric(), XCTClockMetric()]

        let measureOptions = XCTMeasureOptions.default
        measureOptions.iterationCount = 5

        let audioFilePath = try XCTUnwrap(
            Bundle.module.path(forResource: "jfk", ofType: "wav"),
            "Audio file not found"
        )

        let whisperKit = try await WhisperKit(modelFolder: modelPath)

        measure(metrics: metrics, options: measureOptions) {
            let dispatchSemaphore = DispatchSemaphore(value: 0)
            Task {
                let transcriptionResult: [TranscriptionResult] = try await whisperKit.transcribe(audioPath: audioFilePath)
                let transcriptionResultText = transcriptionResult.map(\.text).joined(separator: " ")
                XCTAssertGreaterThan(transcriptionResultText.count, 0)
                dispatchSemaphore.signal()
            }
            dispatchSemaphore.wait()
        }
    }

    func testRealTimeFactorLarge() async throws {
        let modelPath = try largev3ModelPath()

        let metrics: [XCTMetric] = [XCTMemoryMetric(), XCTStorageMetric(), XCTClockMetric()]

        let measureOptions = XCTMeasureOptions.default
        measureOptions.iterationCount = 5

        let audioFilePath = try XCTUnwrap(
            Bundle.module.path(forResource: "jfk", ofType: "wav"),
            "Audio file not found"
        )

        let whisperKit = try await WhisperKit(modelFolder: modelPath, verbose: false)

        measure(metrics: metrics, options: measureOptions) {
            let dispatchSemaphore = DispatchSemaphore(value: 0)
            Task {
                let transcriptionResult: [TranscriptionResult] = try await whisperKit.transcribe(audioPath: audioFilePath)
                XCTAssertGreaterThan(transcriptionResult.text.count, 0)
                dispatchSemaphore.signal()
            }
            dispatchSemaphore.wait()
        }
    }

    func testBaseImplementation() throws {
        let audioFilePath = try XCTUnwrap(
            Bundle.module.path(forResource: "jfk", ofType: "wav"),
            "Audio file not found"
        )

        let dispatchSemaphore = DispatchSemaphore(value: 0)

        Task {
            let whisperKit = try await XCTUnwrapAsync(await WhisperKit(model: "large-v3"))
            let transcriptionResult: [TranscriptionResult] = try await whisperKit.transcribe(audioPath: audioFilePath)
            XCTAssertGreaterThan(transcriptionResult.text.count, 0)
            dispatchSemaphore.signal()
        }

        dispatchSemaphore.wait()
    }

    func testAsyncImplementation() async throws {
        let audioFilePath = try XCTUnwrap(
            Bundle.module.path(forResource: "jfk", ofType: "wav"),
            "Audio file not found"
        )
        let whisperKit = try await WhisperKit(model: "large-v3")
        let transcriptionResult: [TranscriptionResult] = try await whisperKit.transcribe(audioPath: audioFilePath)

        XCTAssertGreaterThan(transcriptionResult.text.count, 0)
    }

    func testBatchTranscribeAudioPaths() async throws {
        let audioPaths = try [
            XCTUnwrap(
                Bundle.module.path(forResource: "jfk", ofType: "wav"),
                "Audio file not found"
            ),
            XCTUnwrap(
                Bundle.module.path(forResource: "es_test_clip", ofType: "wav"),
                "Audio file not found"
            ),
            XCTUnwrap(
                Bundle.module.path(forResource: "ja_test_clip", ofType: "wav"),
                "Audio file not found"
            ),
        ]
        let whisperKit = try await WhisperKit(modelFolder: tinyModelPath())
        let transcriptionResults: [Result<[TranscriptionResult], Swift.Error>] = await whisperKit.transcribe(audioPaths: audioPaths)

        XCTAssertEqual(transcriptionResults.count, 3)
        XCTAssertTrue(transcriptionResults.allSatisfy { $0.isSuccess })
        XCTAssertEqual(
            try transcriptionResults[0].normalizedText(prefix: 5),
            "and so my fellow americans"
        )
        XCTAssertEqual(
            try transcriptionResults[1].normalizedText(prefix: 2),
            "this is"
        )
        XCTAssertEqual(
            try transcriptionResults[2].normalizedText(prefix: 1),
            "tokyo"
        )
    }

    func testBatchTranscribeAudioPathsWithErrors() async throws {
        let audioPaths = try [
            "/path/to/file1.wav",
            XCTUnwrap(
                Bundle.module.path(forResource: "jfk", ofType: "wav"),
                "Audio file not found"
            ),
            "/path/to/file2.wav",
        ]
        let whisperKit = try await WhisperKit(modelFolder: tinyModelPath())
        let transcriptionResults: [Result<[TranscriptionResult], Swift.Error>] = await whisperKit.transcribe(audioPaths: audioPaths)

        XCTAssertEqual(transcriptionResults.count, 3)
        XCTAssertEqual(
            transcriptionResults[0].whisperError(),
            .loadAudioFailed("Resource path does not exist /path/to/file1.wav")
        )
        XCTAssertEqual(
            try transcriptionResults[1].normalizedText(prefix: 5),
            "and so my fellow americans"
        )
        XCTAssertEqual(
            transcriptionResults[2].whisperError(),
            .loadAudioFailed("Resource path does not exist /path/to/file2.wav")
        )
    }

    func testBatchTranscribeAudioArrays() async throws {
        let audioPaths = try [
            XCTUnwrap(
                Bundle.module.path(forResource: "jfk", ofType: "wav"),
                "Audio file not found"
            ),
            XCTUnwrap(
                Bundle.module.path(forResource: "es_test_clip", ofType: "wav"),
                "Audio file not found"
            ),
            XCTUnwrap(
                Bundle.module.path(forResource: "ja_test_clip", ofType: "wav"),
                "Audio file not found"
            ),
        ]
        let audioArrays = try audioPaths
            .map { try AudioProcessor.loadAudio(fromPath: $0) }
            .map { AudioProcessor.convertBufferToArray(buffer: $0) }

        let whisperKit = try await WhisperKit(modelFolder: tinyModelPath())
        let transcriptionResults: [Result<[TranscriptionResult], Swift.Error>] = await whisperKit.transcribe(audioArrays: audioArrays)

        XCTAssertEqual(transcriptionResults.count, 3)
        XCTAssertTrue(transcriptionResults.allSatisfy { $0.isSuccess })
        XCTAssertEqual(
            try transcriptionResults[0].normalizedText(prefix: 5),
            "and so my fellow americans"
        )
        XCTAssertEqual(
            try transcriptionResults[1].normalizedText(prefix: 2),
            "this is"
        )
        XCTAssertEqual(
            try transcriptionResults[2].normalizedText(prefix: 1),
            "tokyo"
        )
    }

    func testModelSearchPathLarge() async throws {
        let audioFilePath = try XCTUnwrap(
            Bundle.module.path(forResource: "jfk", ofType: "wav"),
            "Audio file not found"
        )

        let pipe1 = try await WhisperKit(model: "large-v3", verbose: true, logLevel: .debug)
        let transcriptionResult1: [TranscriptionResult] = try await pipe1.transcribe(audioPath: audioFilePath)
        XCTAssertFalse(transcriptionResult1.text.isEmpty)

        let pipe2 = try await WhisperKit(model: "distil*large-v3", verbose: true, logLevel: .debug)
        let transcriptionResult2: [TranscriptionResult] = try await pipe2.transcribe(audioPath: audioFilePath)
        XCTAssertFalse(transcriptionResult2.text.isEmpty)

        let pipe3 = try await WhisperKit(model: "distil-whisper_distil-large-v3", verbose: true, logLevel: .debug)
        let transcriptionResult3: [TranscriptionResult] = try await pipe3.transcribe(audioPath: audioFilePath)
        XCTAssertFalse(transcriptionResult3.text.isEmpty)
    }
}
