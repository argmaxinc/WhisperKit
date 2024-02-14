//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
@testable import WhisperKit
import XCTest

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
final class FunctionalTests: XCTestCase {
    func testInitLarge() async {
        let modelPath = largev3ModelPath()
        let whisperKit = try? await WhisperKit(modelFolder: modelPath, logLevel: .error)
        XCTAssertNotNil(whisperKit)
    }

    func testOutputAll() async throws {
        let modelPaths = allModelPaths()

        for modelPath in modelPaths {
            let modelName = modelPath.split(separator: "/").last!
            print("[Integration] Testing model \(modelName)")
            guard let audioFilePath = Bundle.module.path(forResource: "jfk", ofType: "wav") else {
                XCTFail("Audio file not found")
                return
            }

            let whisperKit = try await WhisperKit(modelFolder: modelPath, verbose: true, logLevel: .debug)

            guard let transcriptionResult = try await whisperKit.transcribe(audioPath: audioFilePath) else {
                XCTFail("Transcription failed")
                return
            }
            print("[Integration] \(transcriptionResult.text)")
            XCTAssertEqual(
                transcriptionResult.text.normalized,
                " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.".normalized,
                "Transcription result does not match expected result for model \(modelName)"
            )
        }
    }

    func testRealTimeFactorTiny() async throws {
        let modelPath = tinyModelPath()

        let metrics: [XCTMetric] = [XCTMemoryMetric(), XCTStorageMetric(), XCTClockMetric()]

        let measureOptions = XCTMeasureOptions.default
        measureOptions.iterationCount = 5

        guard let audioFilePath = Bundle.module.path(forResource: "jfk", ofType: "wav") else {
            XCTFail("Audio file not found")
            return
        }

        let whisperKit = try await WhisperKit(modelFolder: modelPath)

        measure(metrics: metrics, options: measureOptions) {
            let dispatchSemaphore = DispatchSemaphore(value: 0)
            Task {
                guard let transcriptionResult = try await whisperKit.transcribe(audioPath: audioFilePath) else {
                    XCTFail("Transcription failed")
                    return
                }
                XCTAssertGreaterThan(transcriptionResult.text.count, 0)
                dispatchSemaphore.signal()
            }
            dispatchSemaphore.wait()
        }
    }

    func testRealTimeFactorLarge() async throws {
        let modelPath = largev3ModelPath()

        let metrics: [XCTMetric] = [XCTMemoryMetric(), XCTStorageMetric(), XCTClockMetric()]

        let measureOptions = XCTMeasureOptions.default
        measureOptions.iterationCount = 5

        guard let audioFilePath = Bundle.module.path(forResource: "jfk", ofType: "wav") else {
            XCTFail("Audio file not found")
            return
        }

        let whisperKit = try await WhisperKit(modelFolder: modelPath, verbose: false)

        measure(metrics: metrics, options: measureOptions) {
            let dispatchSemaphore = DispatchSemaphore(value: 0)
            Task {
                guard let transcriptionResult = try await whisperKit.transcribe(audioPath: audioFilePath) else {
                    XCTFail("Transcription failed")
                    return
                }
                XCTAssertGreaterThan(transcriptionResult.text.count, 0)
                dispatchSemaphore.signal()
            }
            dispatchSemaphore.wait()
        }
    }

    func testBaseImplementation() {
        guard let audioFilePath = Bundle.module.path(forResource: "jfk", ofType: "wav") else {
            XCTFail("Audio file not found")
            return
        }

        let dispatchSemaphore = DispatchSemaphore(value: 0)

        Task {
            let pipe = try? await WhisperKit(model: "large-v3")
            let transcription = try? await pipe!.transcribe(audioPath: audioFilePath)?.text
            XCTAssertGreaterThan(transcription!.count, 0)
            dispatchSemaphore.signal()
        }

        dispatchSemaphore.wait()
    }

    func testAsyncImplementation() async {
        guard let audioFilePath = Bundle.module.path(forResource: "jfk", ofType: "wav") else {
            XCTFail("Audio file not found")
            return
        }

        let pipe = try? await WhisperKit(model: "large-v3")
        let transcription = try? await pipe!.transcribe(audioPath: audioFilePath)?.text
        XCTAssertGreaterThan(transcription!.count, 0)
    }

    func testAsyncThrowingImplementation() async throws {
        guard let audioFilePath = Bundle.module.path(forResource: "jfk", ofType: "wav") else {
            XCTFail("Audio file not found")
            return
        }

        let pipe = try await WhisperKit(model: "large-v3")
        let transcription = try await pipe.transcribe(audioPath: audioFilePath)?.text

        XCTAssertGreaterThan(transcription!.count, 0)
    }
}
