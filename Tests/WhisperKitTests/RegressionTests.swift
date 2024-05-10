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
            await whisperKit.transcribe(audioPath: audioFilePath, callback: callback),
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
            let wer = evaluate(
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
        assert(Fraction("3.1415") == Fraction(numerator: 6823, denominator: 2000))
        assert(Fraction("-47e-2") == Fraction(numerator: -47, denominator: 100))
        assert(Fraction(2.25) == Fraction(numerator: 9, denominator: 4))
        assert(Fraction(2.25)! * Fraction(numerator: 100, denominator: 5)! == Fraction(numerator: 45, denominator: 1))
        assert(Fraction(2.25)! * 100 == Fraction(numerator: 225, denominator: 1))
        assert(Fraction(2.25)! + Fraction(1.25)! == Fraction(numerator: 7, denominator: 2))
    }
    
    func testWER(){
        let test = "This is a test string"
        let origText = "<unk> Welcome to the Gol Airlines Third Quarter 2021 Results Conference Call. This morning, the company made its numbers available, along with three videos with the results presentation, financial review, and preliminary Q & A. Gol hopes that everyone connected has watched them. After the company's brief remarks, we will initiate the Q & A session, when further instructions will be provided. This event is also being broadcast live via webcast, and may be accessed through the company website at www.voegol.com .vr/ir and on the MZIQ platform at www.mziq.com. Those following the presentation via the webcast may post their questions on the platform, and their questions will either be answered by the management during this call, or by the Gol Investor Relations Team after the conference is finished. Before proceeding, let me mention that forward-looking statements are based on the beliefs and assumptions of Gol's management and on information currently available to the company. They involve risks and uncertainties because they relate to future events and therefore depend on circumstances that may or may not occur. Investors and analysts should understand that events related to macroeconomic conditions, industry, and other factors could also cause results to differ materially from those expressed in such forward looking statements. At this time, I will hand you over to Mr. Paul Kakinoff, CEO. Please begin. Good morning everyone, and welcome to Gol Airlines Quarterly Earnings Call. I would like to start by highlighting our most important achievements of this spirit. The first one was the continued recovery in demand, which showed solid growth during the third quarter. At the end of September, Brazil became fourth among all countries with the most vaccines administered against COVID-19. Approximately 56% of Brazil's population is fully vaccinated, and over 74% have received their first dose, a higher percentage than the vast majority of the countries, including the United States. Similar to demand trends in other markets, the rising vaccination rate in the general population is supporting the air markets' ongoing recovery. As a result, Gol's departures in the third quarter grew by 87%, reaching 52% of the levels in 2019. In response to this demand, Gol is expanding its network, and has already announced a new route from Congoines to Bonito. It's starting in this December. We are taking a conservative approach to increasing capacity as travel demand recovers to help maintain high load factors and profitability in our routes. The second important event was a transition of the fleet to Boeing Max 8. In preparation for this strong recovery in air travel that we expect to see in the coming quarters, we sign agreements to accelerate the transformation of our fleet, with the acquisition of 28 additional Boeing 737 Max 8 aircraft. This initiative is expected to reduce the company's unit cost by 8% in 2022. Because of the new contract, we will end 2021 with 28 new Max Aircraft, which represents 20% of the fleet. By the end of 2022, we expect to have 44 Max Aircraft, raising this total to 32%. With purchase commitments, we will meet our 2030 goal of having 75% of the fleet in this new aircraft. And, as is widely recognized, the Max is 15% more fuel- efficient, generates 60% less carbon emissions, and is 40% quieter compared to the MG. This aircraft has positioned us to grow even more competitively, expanding routes to new destinations and providing efficiency gains, all of which will capture more value for all our stakeholders. The full important achievement was the conclusion of the merger with MIOS into GLE. This transaction will generate great value from several operation synergies, as well as new opportunities and strategies that will become even more significant during the airline market recovery. We're optimistic that the synergies from this corporate reorganization, expected to be approximately $3 billion reais in net present value for the next five years, and the subsequent values to our shareholders will be realized in a relatively short period of time. With that, I will hand the floor over to Richard, our CFO, who will present some financial highlights. Thank you, Kak. Our most recent notable event was the success of our liability management program. In September, we issued $150 million in a re-tap at 8% annual interest rate on our senior secured notes, maturing in 2026. Moody's assigned the notes a rating of B2. Proceeds from the offering will be used for general corporate purposes, including aircraft acquisitions and working capital. In October, we re-financed our short-term bank debt, in the amount of $1.2 billion reais by an extension of the seventh series of debentures and the issuance of our eighth series of simple, non-convertible debentures. This re-financing enabled the company to return to its lowest level of short-term debt since 2014 at about a half a billion reais, which will also improve Gol's credit metrics by better matching future assets and liabilities, and reducing the average cost of debt. Our next relevant maturity date for outstanding debt is not until July 2024. Gol's balance sheet is now in a better position in terms of our outstanding debt, versus our peers, which we view to be a competitive advantage in the current market environment. In addition, the company advertised around $518 million reais of debt in this quarter, the average maturity of Gol's long-term debt, excluding aircraft leases and perpetual notes, is approximately 3.4 years, with the mean obligations already addressed in our cash flow. The net debt ratio, excluding exchangeable notes and perpetual bonds, to adjusted last twelve months we've adopted, was 9.7 times on September 30, 2021, representing the lowest financial leverage among peers. Considering the amounts fundable from deposits and unencumbered assets, the company's potential sources of liquidity resulted in approximately $6.1 billion reais of accessible liquidity. The recent capitalization of the balance sheet, with capital increase led by the majority shareholder, represented the recognition of Gol's value as Brazil's largest airline with the best product. The re-financing of our short-term bank debt in October, added to long-term capital of $2.7 billion reais raised in the second and third quarters of this year, totals over $3.9 billion reais in the last seven months. As far as a discussion of financial results for the quarter, it was shared this morning in the video presentation, and we believe you all had a chance to access that. In short, our work to re-establish operating margins that can support the sustained growth of operations is bearing fruit. We ended the third quarter with an EBIT reaching $330 million reais and an operating margin totalling 17.7%. Concurrently, adjusted EBITDA $446 million reais with a 24.3% margin, evidencing our successful efforts at matching supply and demand. I will now return back over to Kakinoff. Thanks Richard. We have seen a recovery in demand for air travel, and we believe that now, with greater population immunization, and the significant expansion of vaccination, we will have a strong fourth quarter, coinciding with the start of the summer season. I would like to close by thanking our employees, the team of RIGOS, who are working with extreme professionalism and commitment. All this adaption puts us in a solid position to expand operations and achieve profitable growth. We reiterate our confidence that Gol will re-emerge as strong and even more resilient as markets normalize. Now, I would like to initiate the Q & A session. Thank you. The conference call is now open for questions. If you have a question, please press * 1 at this or any time. If at any point your question is answered, you may remove yourself from the queue by pressing * 2. We ask that when you ask your questions, speak close to the receiver of the device, so that everyone can hear you clearly. Participants can also send questions via the webcast platform. You need to click on the? In the upper-left hand corner and type in your question. Please hold while we poll for questions. Our first question is from Steven Trent with Citi. Please go ahead. Good morning, gentlemen and thanks very much for taking my questions. I just kind of wanted, you know, your high-level views on international demand to the U.S. spooling up again. You know, now that you're partnered with American Airlines, you know, what sort of bigger opportunities are you seeing on the horizon, and do you see any opportunity as well for American to possibly increase its stake in Gol in some point in the future? Hi Steven, Kakinoff here, good morning. Thank you very much for your question. Let me give you an overview on the North American market specifically. Uh, firstly you know we are, uh, now gradually reaching our international goals. So we have already made available the ticket sales for Cancun in Mexico, Dominicans, for the Dominican Republic, and we are now resuming Montevideo and Buenos Aires."
        
        let normText = EnglishTextNormalizer().normalize(text: origText)
        
        let wer = evaluate(originalTranscript: normText, generatedTranscript: origText)
        
        assert(wer == 0)
    }
}
