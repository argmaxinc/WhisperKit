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

            let transcriptionResult = try await XCTUnwrapAsync(
                await whisperKit.transcribe(audioPath: audioFilePath),
                "Transcription failed"
            )

            print("[Integration] \(transcriptionResult.text)")
            XCTAssertEqual(
                transcriptionResult.text.normalized,
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
                let transcriptionResult = try await XCTUnwrapAsync(
                    await whisperKit.transcribe(audioPath: audioFilePath),
                    "Transcription failed"
                )
                XCTAssertGreaterThan(transcriptionResult.text.count, 0)
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
                let transcriptionResult = try await XCTUnwrapAsync(
                    await whisperKit.transcribe(audioPath: audioFilePath),
                    "Transcription failed"
                )
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
            let transcriptionResult = try await XCTUnwrapAsync(
                await whisperKit.transcribe(audioPath: audioFilePath),
                "Transcription failed"
            )
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
        let transcriptionResult = try await XCTUnwrapAsync(
            await whisperKit.transcribe(audioPath: audioFilePath),
            "Transcription failed"
        )

        XCTAssertGreaterThan(transcriptionResult.text.count, 0)
    }

    func testBatchTranscribeAudioPaths() async throws {
        let audioPaths = [
            try XCTUnwrap(
                Bundle.module.path(forResource: "jfk", ofType: "wav"),
                "Audio file not found"
            ),
            try XCTUnwrap(
                Bundle.module.path(forResource: "es_test_clip", ofType: "wav"),
                "Audio file not found"
            ),
            try XCTUnwrap(
                Bundle.module.path(forResource: "ja_test_clip", ofType: "wav"),
                "Audio file not found"
            )
        ]
        let whisperKit = try await WhisperKit(modelFolder: try tinyModelPath())
        let transcriptionResults = try await XCTUnwrapAsync(
            await whisperKit.transcribe(audioPaths: audioPaths),
            "Transcription failed"
        )

        XCTAssertEqual(transcriptionResults.count, 3)
        XCTAssertTrue(transcriptionResults.allSatisfy { $0.value.isSuccess })
        XCTAssertEqual(
            try transcriptionResults[audioPaths[0]]?.normalizedText(prefix: 5),
            "and so my fellow americans"
        )
        XCTAssertEqual(
            try transcriptionResults[audioPaths[1]]?.normalizedText(prefix: 2),
            "this is"
        )
        XCTAssertEqual(
            try transcriptionResults[audioPaths[2]]?.normalizedText(prefix: 1),
            "tokyo"
        )
    }

    func testBatchTranscribeAudioArrays() async throws {
        let audioPaths = [
            try XCTUnwrap(
                Bundle.module.path(forResource: "jfk", ofType: "wav"),
                "Audio file not found"
            ),
            try XCTUnwrap(
                Bundle.module.path(forResource: "es_test_clip", ofType: "wav"),
                "Audio file not found"
            ),
            try XCTUnwrap(
                Bundle.module.path(forResource: "ja_test_clip", ofType: "wav"),
                "Audio file not found"
            )
        ]
        let audioArrays = audioPaths
            .compactMap { AudioProcessor.loadAudio(fromPath: $0) }
            .map { AudioProcessor.convertBufferToArray(buffer: $0) }

        let whisperKit = try await WhisperKit(modelFolder: try tinyModelPath())
        let transcriptionResults = try await XCTUnwrapAsync(
            await whisperKit.transcribe(audioArrays: audioArrays),
            "Transcription failed"
        )

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
        guard let audioFilePath = Bundle.module.path(forResource: "jfk", ofType: "wav") else {
            XCTFail("Audio file not found")
            return
        }

        var pipe = try await WhisperKit(model: "large-v3", verbose: true, logLevel: .debug)

        guard let transcriptionResult = try await pipe.transcribe(audioPath: audioFilePath) else {
            XCTFail("Transcription failed")
            return
        }
        XCTAssertGreaterThan(transcriptionResult.text.count, 0)

        pipe = try await WhisperKit(model: "distil*large-v3", verbose: true, logLevel: .debug)

        guard let transcriptionResult = try await pipe.transcribe(audioPath: audioFilePath) else {
            XCTFail("Transcription failed")
            return
        }
        XCTAssertGreaterThan(transcriptionResult.text.count, 0)

        pipe = try await WhisperKit(model: "distil-whisper_distil-large-v3", verbose: true, logLevel: .debug)

        guard let transcriptionResult = try await pipe.transcribe(audioPath: audioFilePath) else {
            XCTFail("Transcription failed")
            return
        }
        XCTAssertGreaterThan(transcriptionResult.text.count, 0)
    }
}
