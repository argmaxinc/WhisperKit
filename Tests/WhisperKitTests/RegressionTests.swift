import CoreML
import Hub
@testable import WhisperKit
import XCTest
import Foundation
import UniformTypeIdentifiers

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
final class RegressionTests: XCTestCase {
    var audioFileURLs: [URL]?
    var metadataURL: URL?
    var testWERURLs: [URL]?

    override func setUp() {
        super.setUp()
        if self.audioFileURLs == nil || self.metadataURL == nil || self.testWERURLs == nil{
            let expectation = XCTestExpectation(description: "Download test audio")
            downloadTestAudio { success in
                if success {
                    expectation.fulfill()
                } else {
                    XCTFail("Downloading audio file for testing failed")
                }
            }
            // Wait for the expectation with a timeout
            wait(for: [expectation], timeout: 300)
        }
    }

    private func downloadTestAudio(completion: @escaping (Bool) -> Void) {
        Task {
            do {
                let earnings22CompressedDataset = Hub.Repo(id: "argmaxinc/whisperkit-test-data", type: .datasets)
                let tempPath = FileManager.default.temporaryDirectory
                let downloadBase = tempPath.appending(component: "huggingface")
                let hubApi = HubApi(downloadBase: downloadBase)
                let repoURL = try await hubApi.snapshot(from: earnings22CompressedDataset, matching: ["*.mp3","*.txt"])
        
                var audioFileURLs: [URL] = []
                var testWERURLs: [URL] = []
                for file in try FileManager.default.contentsOfDirectory(atPath: repoURL.path()){
                    if file.hasSuffix(".mp3"){
                        audioFileURLs.append(repoURL.appending(component: file))
                    }else if file.hasSuffix(".txt"){
                        testWERURLs.append(repoURL.appending(component: file))
                    }
                }
                self.audioFileURLs = audioFileURLs
                self.testWERURLs = testWERURLs
                
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
    
    private func getTranscript(filename: String) -> String?{
        var transcript: String? = nil
        if let metadataURL = self.metadataURL, let data = try? Data(contentsOf: metadataURL){
            if let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [[String: Any]] {
                for audioItem in json{
                    if audioItem["audio"] as? String == filename{
                        transcript = audioItem["transcription"] as? String
                    }
                }
            }
        }
        return transcript
    }
    
    private func getWERTestData() -> (String?, String?){
        do{
            let testFileURLs = try XCTUnwrap(
                self.testWERURLs,
                "Test files for WER verification not found"
            )
            var generatedText:String? = nil
            var originalText:String? = nil
            for file in testFileURLs{
                switch file.lastPathComponent{
                case "test_generated_transcript.txt":
                    generatedText = try? String(contentsOf: file)
                case "test_original_transcript.txt":
                    originalText = try? String(contentsOf: file)
                default:
                    continue
                }
            }
            return (originalText, generatedText)
        }
        catch{
            XCTFail("Fetching test data for WER verification failed: \(error)")
        }
        return (nil,nil)
    }

    func testAndMeasureModelPerformance(model: String, device: String, overEntireDataset: Bool) async throws {
        var resultJSON:[RegressionStats] = []
        let iso8601DateTimeString = ISO8601DateFormatter().string(from: Date())
        let audioFilePaths = try XCTUnwrap(
            self.audioFileURLs,
            "Audio files not found"
        ).map({$0.path()})
        
        for audioFilePath in audioFilePaths{
            let startTime = Date()
            
            var currentAppMemoryValues = [Float]()
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
            var lastTimeStamp = CFAbsoluteTimeGetCurrent()
            
            let callback = {
                (result: TranscriptionProgress) -> Bool in
                count += 1
                let currentMemory = AppMemoryChecker.getMemoryUsed()
                let timeTaken = CFAbsoluteTimeGetCurrent() - lastTimeStamp
                lastTimeStamp = CFAbsoluteTimeGetCurrent()
                let currentTPS = Double(1/timeTaken)
                
                if currentMemory != 0 {
                    currentAppMemoryValues.append(Float(currentMemory))
                }
                if !currentTPS.isNaN {
                    currentTPSValues.append(Float(currentTPS))
                }
                if count % 100 == 1 {
                    let timeElapsed = Date().timeIntervalSince(startTime)
                    memoryStats.measure(from: currentAppMemoryValues, timeElapsed: timeElapsed)
                    latencyStats.measure(from: currentTPSValues, timeElapsed: timeElapsed)
                    currentAppMemoryValues = []
                    currentTPSValues = []
                }
                return true
            }
            
            let whisperKit = try await WhisperKit(model: model)
            memoryStats.preTranscribeMemory = Float(AppMemoryChecker.getMemoryUsed())
            
            var systemMemory: [SystemMemoryUsage] = []
            var diskSpace: [DiskSpace] = []
            var batteryLevel: [Float] = []
            var timerTimeElapsed: [TimeInterval] = []
            // DispatchSourceTimer to collect memory usage asynchronously
            let timerQueue = DispatchQueue(label: "com.example.SystemStatTimerQueue")
            let timer = DispatchSource.makeTimerSource(queue: timerQueue)
            timer.schedule(deadline: .now(), repeating: 1.0)
            timer.setEventHandler {
                systemMemory.append(SystemMemoryCheckerAdvanced.getMemoryUsage())
                diskSpace.append(DiskSpaceChecker.getDiskSpace())
                batteryLevel.append(BatteryLevelChecker.getBatteryLevel() ?? -1)
                timerTimeElapsed.append(Date().timeIntervalSince(startTime))
            }
            timer.resume()
            
            let transcriptionResult = try await XCTUnwrapAsync(
                await whisperKit.transcribe(audioPath: audioFilePath, callback: callback).first,
                "Transcription failed"
            )
            XCTAssert(transcriptionResult.text.isEmpty == false, "Transcription failed")
            
            memoryStats.postTranscribeMemory = Float(AppMemoryChecker.getMemoryUsed())
            
            var wer = -Double.infinity
            if let filename = audioFilePath.split(separator: "/").last,let originalTranscript = getTranscript(filename: String(filename)){
                wer = WERUtils.evaluate(
                    originalTranscript: originalTranscript,
                    generatedTranscript: transcriptionResult.text,
                    normalizeOriginal: true
                )
                XCTAssert(wer != -Double.infinity, "Calculating WER failed.")
            }
            
            let testInfo = TestInfo(
                device: device,
                audioFile: audioFilePath,
                model: model,
                date: startTime.formatted(Date.ISO8601FormatStyle().dateSeparator(.dash)),
                timeElapsedInSeconds: Date().timeIntervalSince(startTime),
                timings: transcriptionResult.timings,
                transcript: transcriptionResult.text,
                wer: wer
            )
            let staticAttributes = StaticAttributes(
                encoderCompute: whisperKit.modelCompute.audioEncoderCompute,
                decoderCompute: whisperKit.modelCompute.textDecoderCompute
            )
            let systemMeasurements = SystemMeasurements(
                systemMemory: systemMemory,
                diskSpace: diskSpace,
                batteryLevel: batteryLevel,
                timeElapsed: timerTimeElapsed
            )
            let json = RegressionStats(
                testInfo: testInfo,
                memoryStats: memoryStats,
                latencyStats: latencyStats,
                staticAttributes: staticAttributes,
                systemMeasurements: systemMeasurements
            )
            resultJSON.append(json)
            
            if !overEntireDataset{
                break
            }
        }
        
        do {
            let jsonData = try JSONEncoder().encode(resultJSON)
            let attachment = XCTAttachment(data: jsonData, uniformTypeIdentifier: UTType.json.identifier)
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
        
        //Remove trailing whitespace characters
        while currentDevice.last?.isWhitespace == true { currentDevice = String(currentDevice.dropLast())}
        do {
            allModels = try await WhisperKit.fetchAvailableModels()
            // TODO: Remove after testing
            allModels = ["base"]
        } catch {
            XCTFail("Failed to fetch available models: \(error.localizedDescription)")
        }

        for model in allModels {
            do {
                try await testAndMeasureModelPerformance(
                    model: model,
                    device: currentDevice,
                    overEntireDataset: false
                )
            } catch {
                failureInfo[model] = error.localizedDescription
            }
        }
        let testReport = TestReport(device: currentDevice, modelsTested: allModels, failureInfo: failureInfo)
        do {
            let jsonData = try testReport.jsonData()
            let attachment = XCTAttachment(data: jsonData, uniformTypeIdentifier: UTType.json.identifier)
            attachment.lifetime = .keepAlways
            attachment.name = "\(currentDevice)_summary_\(iso8601DateTimeString).json"
            add(attachment)
        } catch {
            XCTFail("Failed with error: \(error)")
        }
    }
    
    func testLargeWER(){
        let texts = getWERTestData()
        if let originalText = texts.0, let generatedText = texts.1{
            let wer = WERUtils.evaluate(originalTranscript: originalText, generatedTranscript: generatedText, normalizeOriginal: true)
            XCTAssert(wer == 0.18961994278708622, "Expected wer: 0.18961994278708622 but computed \(wer)")
        }else{
            XCTFail("Fetching WER test data failed.")
        }
        
    }
    
    func testHirschberg(){
        let s1 = "With a rumble that echoed through the night, thunder crashed overhead, its raw power shaking the earth beneath it, leaving in its wake an exhilarating sense of awe. As rain poured down in torrents, the thunder boomed with a rhythm that seemed to speak a secret language, intertwining nature's symphony with an innovative melody that captivated all who listened."
        let s2 = "In the midst of a summer storm, thunder erupted with a booming chorus, shaking the earth beneath our feet and electrifying the air with its powerful presence. The crackling symphony of thunderbolts danced across the darkened sky, illuminating the clouds with an innovative display of nature's raw energy."
        let ops = hirschberg(Array(s1.unicodeScalars), Array(s2.unicodeScalars))
        XCTAssert(ops.count == 228)
    }
    
}
