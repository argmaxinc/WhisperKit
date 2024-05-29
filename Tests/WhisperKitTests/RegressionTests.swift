import CoreML
import Hub
@testable import WhisperKit
import XCTest
import Foundation

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
final class RegressionTests: XCTestCase {
    var audioFileURL: URL?
    var metadataURL: URL?

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
                
                let earnings22OriginalDataset = Hub.Repo(id: "argmaxinc/earnings22-12hours", type: .datasets)
                let metadataURL = try await hubApi.snapshot(from: earnings22OriginalDataset, matching: ["metadata.json"])
                self.metadataURL = metadataURL.appending(component: "metadata.json")
                completion(true)
            } catch {
                XCTFail("Async setup failed with error: \(error)")
                completion(false)
            }
        }
    }
    
    func getTranscript() -> String?{
        var transcript: String? = nil
        if let metadataURL = self.metadataURL, let data = try? Data(contentsOf: metadataURL){
            if let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [[String: Any]] {
                for audioItem in json{
                    if audioItem["audio"] as? String == self.audioFileURL?.lastPathComponent{
                        transcript = audioItem["transcription"] as? String
                    }
                }
            }
        }
        return transcript
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
        
        if let originalTranscript = getTranscript(){
            let wer = WERUtils.evaluate(
                originalTranscript: originalTranscript,
                generatedTranscript: transcriptionResult.text,
                normalizeOriginal: true
            )
            assert(wer != -Double.infinity)
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
            allModels = ["tiny"]
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
    

    
    func testFractions(){
        assert(Fraction(numerator: 10, denominator: 0) == nil)
        assert(Fraction(numerator: 10, denominator: 10) != nil)
        assert(Fraction("3/7") == Fraction(numerator: 3, denominator: 7))
        assert(Fraction("1/2") == Fraction(numerator: 2, denominator: 4))
        assert(Fraction("100") == Fraction(numerator: 100, denominator: 1))
        assert(Fraction(numerator: 5, denominator: -8) == Fraction(numerator: -5, denominator: 8))
        assert(Fraction(numerator: -5, denominator: -8) == Fraction(numerator: 5, denominator: 8))
        assert(Fraction("3.1415") == Fraction(numerator: 6283, denominator: 2000))
        assert(Fraction("-47e-2") == Fraction(numerator: -47, denominator: 100))
        assert(Fraction(2.25) == Fraction(numerator: 9, denominator: 4))
        assert(Fraction(2.25)! * Fraction(numerator: 100, denominator: 5)! == Fraction(numerator: 45, denominator: 1))
        assert(Fraction(2.25)! * 100 == Fraction(numerator: 225, denominator: 1))
        assert(Fraction(2.25)! + Fraction(1.25)! == Fraction(numerator: 7, denominator: 2))
    }
    
    func testLargeWER(){
        var genText: String?
        var origText: String?
        if let genPath = Bundle.module.path(forResource: "generatedTranscript", ofType: "txt"){
            genText = try? String(contentsOfFile: genPath, encoding: .utf8)
        }
        if let origPath = Bundle.module.path(forResource: "originalTranscript", ofType: "txt"){
            origText = try? String(contentsOfFile: origPath, encoding: .utf8)
        }
        
        let wer = WERUtils.evaluate(originalTranscript: origText!, generatedTranscript: genText!, normalizeOriginal: false)
        
        assert(wer == 0.42448103078024335)
    }
    
    func testHirschberg(){
        let s1 = "With a rumble that echoed through the night, thunder crashed overhead, its raw power shaking the earth beneath it, leaving in its wake an exhilarating sense of awe. As rain poured down in torrents, the thunder boomed with a rhythm that seemed to speak a secret language, intertwining nature's symphony with an innovative melody that captivated all who listened."
        let s2 = "In the midst of a summer storm, thunder erupted with a booming chorus, shaking the earth beneath our feet and electrifying the air with its powerful presence. The crackling symphony of thunderbolts danced across the darkened sky, illuminating the clouds with an innovative display of nature's raw energy."
        let ops = hirschberg(Array(s1.unicodeScalars), Array(s2.unicodeScalars))
        assert(ops.count == 228)
    }
}
