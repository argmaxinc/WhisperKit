import AVFoundation
@testable import WhisperKit
import XCTest

final class ResamplingTests: XCTestCase {

    let commonSampleRates: [Double] = [8_000, 16_000, 22_050, 32_000, 44_100, 48_000, 88_200, 96_200]

    override class func setUp() {
        Logging.shared.logLevel = .error
        super.setUp()
    }
    
    func generateSilentAudioFile(sampleRate: Double, inputFrameCount: AVAudioFrameCount) -> URL {
        let outputFileURL = FileManager.default.temporaryDirectory.appendingPathComponent("silence-\(inputFrameCount)-frames.wav")
        let buffer = AVAudioPCMBuffer.silence(sampleRate: sampleRate, frameCount: inputFrameCount)
        let audioFile = try! AVAudioFile(forWriting: outputFileURL, settings: buffer.format.settings)
        try! audioFile.write(from: buffer)
        return outputFileURL
    }

    func testAudioResampleFromSyntheticSilentFiles() {

        func testSilentAudioFile(sampleRate: Double, frameCount: AVAudioFrameCount) throws {
            let audioURL = generateSilentAudioFile(sampleRate: sampleRate, inputFrameCount: frameCount)
            defer { try! FileManager.default.removeItem(at: audioURL) }
            let buffer1 = try AudioProcessor.loadAudio(fromPath: audioURL.path())
            let buffer2 = try AVAudioFile(forReading: audioURL).resampled()
            if sampleRate == AVAudioFormat.whisperKitTargetFormat.sampleRate {
                precondition(buffer2.frameLength == frameCount)
            }
            guard buffer1.frameLength == buffer2.frameLength else {
                throw WhisperError.audioProcessingFailed("Unexpected buffer length: got \(buffer1.frameLength), expected \(buffer2.frameLength)")
            }
        }

        for inputFrameCount in AVAudioFrameCount(0)...100_000 {
            autoreleasepool {
                let sampleRate = Bool.random() ? commonSampleRates.randomElement()! : Double.random(in: 5_000..<100_000)
                //let sampleRate = 44_100.0 // This fails for values of inputFrameCount = 12289 + 1024 * N
                do {
                    try testSilentAudioFile(sampleRate: sampleRate, frameCount: inputFrameCount)
                } catch {
                    print("inputFrameCount \(inputFrameCount): \(error)")
                }
            }
        }
    }

    func testTranscribe() throws {
        let audioPath = Bundle.module.path(forResource: "jfk_441khz-full", ofType: "m4a")!
        let dispatchSemaphore = DispatchSemaphore(value: 0)
        let modelPath = try tinyModelPath()
        Task {
            let whisperKit = try await XCTUnwrapAsync(await WhisperKit(modelFolder: modelPath))
            let decodeOptions = DecodingOptions(temperature: 0, temperatureFallbackCount: 0)
            let transcriptionResult: [TranscriptionResult] = try! await whisperKit.transcribe(audioPath: audioPath, decodeOptions: decodeOptions)
            // Switch to `transcribe_v2` to compare memory usage
            //let transcriptionResult: [TranscriptionResult] = try! await whisperKit.transcribe_v2(audioPath: audioPath, decodeOptions: decodeOptions)
            precondition(!transcriptionResult.text.isEmpty)
            print(transcriptionResult.text)
            dispatchSemaphore.signal()
        }
        dispatchSemaphore.wait()
    }

}


extension AVAudioPCMBuffer {
    class func silence(sampleRate: Double, frameCount: AVAudioFrameCount) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        buffer.frameLength = frameCount
        return buffer
    }
}
