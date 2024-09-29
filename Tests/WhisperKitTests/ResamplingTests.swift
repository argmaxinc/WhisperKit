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
            guard buffer1.frameLength == buffer2.frameLength || buffer1.frameLength == buffer2.frameLength - 1 else {
                throw WhisperError.audioProcessingFailed("Unexpected buffer length: got \(buffer1.frameLength), expected \(buffer2.frameLength)")
            }
        }

        for inputFrameCount in AVAudioFrameCount(0)...100_000 {
            autoreleasepool {
                //let sampleRate = Bool.random() ? commonSampleRates.randomElement()! : Double.random(in: 5_000..<100_000)
                let sampleRate = 44_100.0
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
//            let transcriptionResult: [TranscriptionResult] = try! await whisperKit.transcribe_v2(audioPath: audioPath, decodeOptions: decodeOptions)
            precondition(!transcriptionResult.text.isEmpty)
            print(transcriptionResult.text)
            dispatchSemaphore.signal()
        }
        dispatchSemaphore.wait()
    }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extension WhisperKit {
    func transcribe_v2(
        audioPath: String,
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> [TranscriptionResult] {
        let audioFile = try AVAudioFile(forReading: URL(filePath: audioPath))
        return try await transcribe(
            audioArray: audioFile.resampled().array,
            decodeOptions: decodeOptions,
            callback: callback
        )
    }
}

extension AVAudioFile {

    func read(into buffer: AVAudioPCMBuffer, frameCount frames: AVAudioFrameCount, error outError: inout Error?) -> Bool {
        outError = nil
        // Mark our output buffer as empty
        buffer.frameLength = 0

        // Ignore errors when attempting to read past the EOF -- (Foundation._GenericObjCError error 0.) (nilError), could also check for those via (error as NSError).code == 0
        // Ignore errors when attempting to write past our buffer -- (Code=-1 \"kCFStreamErrorHTTPParseFailure / kCFSocketError / ...)
        let frameCount = AVAudioFrameCount(min(Int64(frames), length - framePosition, Int64(buffer.frameCapacity)))
        precondition(frameCount >= 0)
        guard frameCount != 0 else { return false }

        do {
            try read(into: buffer, frameCount: frameCount)
            return buffer.frameLength > 0
        } catch {
            outError = error
            return false
        }
    }

    func resampled(to outputFormat: AVAudioFormat = .whisperKitTargetFormat,
                   startFrame: AVAudioFramePosition = 0,
                   endFrame: AVAudioFramePosition? = nil
    ) throws -> AVAudioPCMBuffer {
        framePosition = startFrame
        let endFrame = endFrame ?? self.length

        let inputFormat = processingFormat

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw WhisperError.audioProcessingFailed("Failed to create audio converter")
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: Constants.defaultAudioReadFrameSize) else {
            throw WhisperError.audioProcessingFailed("Failed to create temporary buffer")
        }

        // Calculate the upper bound of the expected output length
        let outputLength = AVAudioFrameCount((Double(length) * outputFormat.sampleRate / inputFormat.sampleRate).rounded(.up))

        // Create our result buffer. We add one to the capacity so we can later check for buffer overflow
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputLength + 1) else {
            throw WhisperError.audioProcessingFailed("Failed to create output buffer")
        }

        // Perform the conversion
        var convertError: NSError?
        var readError: Error?
        let status = converter.convert(to: output, error: &convertError) { [self] _, outStatus in
            let remainingFrames = AVAudioFrameCount(endFrame - framePosition)
            if read(into: buffer, frameCount: remainingFrames, error: &readError) {
                outStatus.pointee = .haveData
                return buffer
            } else {
                outStatus.pointee = .endOfStream
                return nil
            }
        }

        if status == .error {
            throw WhisperError.audioProcessingFailed("Error converting audio: \(String(describing: convertError))")
        }

        if let readError {
            throw WhisperError.audioProcessingFailed("Error reading audio: \(readError)")
        }

        guard status == .haveData || (outputLength <= 1 && status == .endOfStream) else {
            throw WhisperError.audioProcessingFailed("Error converting audio, unexpected status: \(status)")
        }

        // Ensure we didn't underestimate output length, which would quietly cause missing frames
        guard output.frameLength < output.frameCapacity else {
            throw WhisperError.audioProcessingFailed("Error converting audio, possible buffer overflow")
        }

        // Ensure our actual output length is within our expected range
        guard output.frameLength == outputLength || output.frameLength == outputLength - 1 else {
            throw WhisperError.audioProcessingFailed("Error converting audio, unexpected output length")
        }

        // Ensure we read the entire file
        guard framePosition == length else {
            throw WhisperError.audioProcessingFailed("Error converting audio, input file was only partially read")
        }

        return output
    }

    // TODO: investigate the impact of these
    //converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
    //converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
    //converter.primeMethod = .normal
    //converter.primeInfo = ?

}

extension AVAudioPCMBuffer {

    var frameBytes: UInt32 { format.channelCount * UInt32(format.streamDescription.pointee.mBitsPerChannel / 8) }
    var frameCapacityBytes: UInt32 { frameCapacity * frameBytes }
    var frameLengthBytes: UInt32 { frameLength * frameBytes }

    class func silence(sampleRate: Double, frameCount: AVAudioFrameCount) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        buffer.frameLength = frameCount
        return buffer
    }

    var array: [Float] {
        precondition(format == .whisperKitTargetFormat)
        precondition(stride == 1)
        return Array(UnsafeBufferPointer(start: floatChannelData![0], count: Int(frameLength)))
    }
}

extension AVAudioFormat {
    static let whisperKitTargetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(WhisperKit.sampleRate), channels: 1, interleaved: false)!
}
