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
    
    func testTinyPerformanceOverTime() async throws{
        let audioFilePath = try XCTUnwrap(
            Bundle.module.path(forResource: "jfk_long", ofType: "mp4"),
            "Audio file not found"
        )
        let startTime = Date()
        var currentMemoryValues = [Float]()
        var currentTPSValues = [Float]()
        var memoryStats = MemoryStats(
            measurements: [], units: "MB",
            totalNumberOfMeasurements: 0,
            preTranscribeMemory: -1,
            postTranscribeMemory: -1
        )
        var latencyStats = LatencyStats(
            measurements: [], units: "Tokens/Sec",
            totalNumberOfMeasurements: 0
        )
        var count: Int = 0
        
        let callback = {
            (result:TranscriptionProgress) -> Bool in
            count += 1
            let currentMemory = memoryFootprint()
            let currentTPS = result.timings.tokensPerSecond
            if currentMemory != 0{
                currentMemoryValues.append(Float(currentMemory))
            }
            if !currentTPS.isNaN{
                currentTPSValues.append(Float(currentTPS))
            }
            if count%100 == 1{
                let timeElapsed = Date().timeIntervalSince(startTime)
                memoryStats.measure(from: currentMemoryValues, timeElapsed: timeElapsed)
                latencyStats.measure(from: currentTPSValues, timeElapsed: timeElapsed)
                currentMemoryValues = []
                currentTPSValues = []
            }
            return true
        }
        
        let whisperKit = try await WhisperKit(model: "tiny")
        let preTranscribeMemory = memoryFootprint()
        
        let transcriptionResult = try await XCTUnwrapAsync(
            await whisperKit.transcribe(audioPath: audioFilePath, callback: callback),
            "Transcription failed"
        )
        memoryStats.preTranscribeMemory = Float(preTranscribeMemory)
        memoryStats.postTranscribeMemory = Float(memoryFootprint())
        let testInfo = TestInfo(
            device: WhisperKit.deviceName(),
            audioFile: audioFilePath,
            model: "tiny",
            date: startTime.formatted(Date.ISO8601FormatStyle().dateSeparator(.dash)),
            timeElapsedInSeconds: Date().timeIntervalSince(startTime)
        )
        let json = RegressionStats(testInfo: testInfo, memoryStats: memoryStats, latencyStats: latencyStats)
        do{
            try writeToFile(text: json.jsonData().prettyPrintedJSONString as! String, fileName: "output.json")
        }
        catch{
            XCTFail("Failed with error: \(error)")
        }
    }
}
