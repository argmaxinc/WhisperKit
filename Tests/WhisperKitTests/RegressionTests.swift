import CoreML
import Hub
@testable import WhisperKit
import XCTest

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
final class RegressionTests: XCTestCase {
    
    var audioFileURL: URL?
    
    override func setUp() {
        super.setUp()
        
        if self.audioFileURL == nil{
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
                let earnings22CompressedDataset = Hub.Repo(id: "NagaSaiAbhinay/whisperkit_tests", type: .datasets)
                let tempPath = FileManager.default.temporaryDirectory
                let downloadBase = tempPath.appending(component: "huggingface")
                let hubApi = HubApi(downloadBase: downloadBase)
                var fileURL = try? await hubApi.snapshot(from: earnings22CompressedDataset, matching: ["4484146_24.mp3"])
                fileURL = fileURL?.appending(component: "4484146_24.mp3")
                if let url = fileURL {
                    self.audioFileURL = fileURL
                    completion(true)
                } else {
                    completion(false)
                }
            } catch {
                XCTFail("Async setup failed with error: \(error)")
                completion(false)
            }
        }
    }
    
    func testAndMeasureModelPerformance(model: String) async throws{
        let audioFilePath = try XCTUnwrap(
            Bundle.module.path(forResource:"4484146", ofType:"wav"),
            "Audio file not found"
        )

        let startTime = Date()
        let systemMemoryChecker = SystemMemoryChecker()
        let iso8601DateTimeString = ISO8601DateFormatter().string(from: Date())
        
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
            let currentMemory = systemMemoryChecker.getMemoryUsed()
            let currentTPS = result.timings.tokensPerSecond
            if currentMemory != 0{
                currentMemoryValues.append(Float(currentMemory))
            }
            if !currentTPS.isNaN{
                currentTPSValues.append(Float(currentTPS))
            }
            if count % 100 == 1{
                let timeElapsed = Date().timeIntervalSince(startTime)
                memoryStats.measure(from: currentMemoryValues, timeElapsed: timeElapsed)
                latencyStats.measure(from: currentTPSValues, timeElapsed: timeElapsed)
                currentMemoryValues = []
                currentTPSValues = []
            }
            return true
        }
        
        let whisperKit = try await WhisperKit(model: model)
        memoryStats.preTranscribeMemory = Float(systemMemoryChecker.getMemoryUsed())
        
        let transcriptionResult = try await XCTUnwrapAsync(
            await whisperKit.transcribe(audioPath: audioFilePath, callback: callback),
            "Transcription failed"
        )
        XCTAssert(transcriptionResult.text.isEmpty == false, "Transcription failed")
        
        memoryStats.postTranscribeMemory = Float(systemMemoryChecker.getMemoryUsed())
        let testInfo = TestInfo(
            device: WhisperKit.deviceName(),
            audioFile: audioFilePath,
            model: model,
            date: startTime.formatted(Date.ISO8601FormatStyle().dateSeparator(.dash)),
            timeElapsedInSeconds: Date().timeIntervalSince(startTime),
            timings: transcriptionResult.timings
        )
        let json = RegressionStats(testInfo: testInfo, memoryStats: memoryStats, latencyStats: latencyStats)
        do{
            let attachment = try XCTAttachment(data: json.jsonData())
            attachment.lifetime = .keepAlways
            add(attachment)
        }
        catch{
            XCTFail("Failed with error: \(error)")
        }
    }
    
    //MARK: Distil Whisper
    func testDistilLargeV3PerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "distil-whisper_distil-large-v3")
    }
    
    func testDistilLargeV3_594MBPerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "distil-whisper_distil-large-v3_594MB")
    }
    
    func testDistilLargeV3_TurboPerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "distil-whisper_distil-large-v3_turbo")
    }
    
    func testDistilLargeV3_Turbo_600MBPerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "distil-whisper_distil-large-v3_turbo_600MB")
    }
    
    //MARK: Open AI Whisper
    func testBasePerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "base")
    }
    
    func testBaseEnglishPerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "base.en")
    }
    
    func testLargeV2PerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "large-v2")
    }
    
    func testLargeV2949PerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "large-v2_949")
    }
    
    func testLargeV2TurboPerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "large-v2_turbo")
    }
    
    func testLargeV2Turbo955PerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "large-v2_turbo_955")
    }
    
    func testLargeV3PerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "large-v3")
    }
    
    func testLargeV3947PerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "large-v3_947")
    }
    
    func testLargeV3TurboPerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "large-v3_turbo")
    }
    
    func testLargeV3Turbo954PerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "large-v3_turbo_954")
    }
    
    func testSmallPerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "small")
    }
    
    func testSmallEnglishPerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "small.en")
    }
    
    func testTinyPerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "tiny")
    }
    
    func testTinyEnglishPerformanceOverTime() async throws{
        try await testAndMeasureModelPerformance(model: "tiny.en")
    }
    
}
