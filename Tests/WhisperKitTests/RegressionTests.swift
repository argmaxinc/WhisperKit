import CoreML
import Hub
@testable import WhisperKit
import XCTest

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
final class RegressionTests: XCTestCase {
    var audioFileURL: URL?

    override func setUp() {
        super.setUp()

        if self.audioFileURL == nil {
            let expectation = XCTestExpectation(description: "Download test audio")
            downloadTestAudio { success in
                if success {
                    expectation.fulfill()
                } else {
                    XCTFail("Downloading audio file for testing failed")
                }
            }
            // Wait for the expectation with a timeout
            wait(for: [expectation], timeout: 30)
        }
    }

    func downloadTestAudio(completion: @escaping (Bool) -> Void) {
        Task {
            do {
                let earnings22CompressedDataset = Hub.Repo(id: "argmaxinc/whisperkit-test-data", type: .datasets)
                let tempPath = FileManager.default.temporaryDirectory
                let downloadBase = tempPath.appending(component: "huggingface")
                let hubApi = HubApi(downloadBase: downloadBase)
                let fileURL = try await hubApi.snapshot(from: earnings22CompressedDataset, matching: ["4484146.mp3"])
                self.audioFileURL = fileURL.appending(component: "4484146.mp3")
                completion(true)
            } catch {
                XCTFail("Async setup failed with error: \(error)")
                completion(false)
            }
        }
    }

    func testAndMeasureModelPerformance(model: String, device: String) async throws {
        let audioFilePath = try XCTUnwrap(
            self.audioFileURL?.path(),
            "Audio file not found"
        )

        let startTime = Date()
        let iso8601DateTimeString = ISO8601DateFormatter().string(from: Date())

        var currentMemoryValues = [Float]()
        var currentTPSValues = [Float]()

        let memoryStats = MemoryStats(
            measurements: [], units: "MB",
            totalNumberOfMeasurements: 0,
            preTranscribeMemory: -1,
            postTranscribeMemory: -1
        )
        let latencyStats = LatencyStats(
            measurements: [], units: "Tokens/Sec",
            totalNumberOfMeasurements: 0
        )
        var count = 0

        let callback = {
            (result: TranscriptionProgress) -> Bool in
            count += 1
            let currentMemory = SystemMemoryChecker.getMemoryUsed()
            let currentTPS = result.timings.tokensPerSecond
            if currentMemory != 0 {
                currentMemoryValues.append(Float(currentMemory))
            }
            if !currentTPS.isNaN {
                currentTPSValues.append(Float(currentTPS))
            }
            if count % 100 == 1 {
                let timeElapsed = Date().timeIntervalSince(startTime)
                memoryStats.measure(from: currentMemoryValues, timeElapsed: timeElapsed)
                latencyStats.measure(from: currentTPSValues, timeElapsed: timeElapsed)
                currentMemoryValues = []
                currentTPSValues = []
            }
            return true
        }

        let whisperKit = try await WhisperKit(model: model)
        memoryStats.preTranscribeMemory = Float(SystemMemoryChecker.getMemoryUsed())

        let transcriptionResult = try await XCTUnwrapAsync(
            await whisperKit.transcribe(audioPath: audioFilePath, callback: callback).first,
            "Transcription failed"
        )
        XCTAssert(transcriptionResult.text.isEmpty == false, "Transcription failed")

        memoryStats.postTranscribeMemory = Float(SystemMemoryChecker.getMemoryUsed())
        let testInfo = TestInfo(
            device: device,
            audioFile: audioFilePath,
            model: model,
            date: startTime.formatted(Date.ISO8601FormatStyle().dateSeparator(.dash)),
            timeElapsedInSeconds: Date().timeIntervalSince(startTime),
            timings: transcriptionResult.timings,
            transcript: transcriptionResult.text
        )
        let json = RegressionStats(testInfo: testInfo, memoryStats: memoryStats, latencyStats: latencyStats)
        do {
            let attachment = try XCTAttachment(data: json.jsonData(), uniformTypeIdentifier: "json")
            attachment.lifetime = .keepAlways
            attachment.name = "\(device)_\(model)_\(iso8601DateTimeString).json"
            add(attachment)
        } catch {
            XCTFail("Failed with error: \(error)")
        }
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

    func testRegressionAndLatencyForAllModels() async throws {
        var allModels: [String] = []
        var failureInfo: [String: String] = [:]
        var currentDevice = WhisperKit.deviceName()
        let iso8601DateTimeString = ISO8601DateFormatter().string(from: Date())

        #if os(macOS) && arch(arm64)
        currentDevice = Process.processor
        #endif

        do {
            allModels = try await WhisperKit.fetchAvailableModels()
        } catch {
            XCTFail("Failed to fetch available models: \(error.localizedDescription)")
        }

        for model in allModels {
            do {
                try await testAndMeasureModelPerformance(model: model, device: currentDevice)
            } catch {
                failureInfo[model] = error.localizedDescription
            }
        }
        let testReport = TestReport(device: currentDevice, modelsTested: allModels, failureInfo: failureInfo)
        do {
            let attachment = try XCTAttachment(data: testReport.jsonData(), uniformTypeIdentifier: "json")
            attachment.lifetime = .keepAlways
            attachment.name = "\(currentDevice)_summary_\(iso8601DateTimeString).json"
            add(attachment)
        } catch {
            XCTFail("Failed with error: \(error)")
        }
    }
}
