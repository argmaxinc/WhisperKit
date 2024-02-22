//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import AVFoundation
import CoreAudio
import CoreML

public protocol AudioProcessing {
    /// Loads audio data from a specified file path.
    /// - Parameter audioFilePath: The file path of the audio file.
    /// - Returns: An optional `AVAudioPCMBuffer` containing the audio data.
    static func loadAudio(fromPath audioFilePath: String) -> AVAudioPCMBuffer?

    ///  Pad or trim the audio data to the desired length.
    /// - Parameters:
    ///   - audioArray:An array of audio frames to be padded or trimmed.
    ///   - startIndex: The index of the audio frame to start at.
    ///   - frameLength: The desired length of the audio data in frames.
    ///   - saveSegment: Save the segment to disk for debugging.
    /// - Returns: An optional `MLMultiArray` containing the audio data.
    static func padOrTrimAudio(
        fromArray audioArray: [Float],
        startAt startIndex: Int,
        toLength frameLength: Int,
        saveSegment: Bool
    ) -> MLMultiArray?

    /// Stores the audio samples to be transcribed
    var audioSamples: ContiguousArray<Float> { get }

    /// Empties the audio samples array, keeping the last `keep` samples
    func purgeAudioSamples(keepingLast keep: Int)

    /// A measure of current buffer's energy in dB normalized from 0 - 1 based on the quietest buffer's energy in a specified window
    var relativeEnergy: [Float] { get }

    /// How many past buffers of audio to use to calculate relative energy
    /// The lowest average energy value in the buffer within this amount of previous buffers will used as the silence baseline
    var relativeEnergyWindow: Int { get set }

    /// Starts recording audio from the specified input device, resetting the previous state
    func startRecordingLive(callback: (([Float]) -> Void)?) throws

    /// Pause recording
    func pauseRecording()

    /// Stops recording and cleans up resources
    func stopRecording()
}

// Overrideable default methods for AudioProcessing
public extension AudioProcessing {
    // Use default recording device
    func startRecordingLive(callback: (([Float]) -> Void)?) throws {
        try startRecordingLive(callback: callback)
    }

    static func padOrTrimAudio(fromArray audioArray: [Float], startAt startIndex: Int = 0, toLength frameLength: Int = 480_000, saveSegment: Bool = false) -> MLMultiArray? {
        let currentFrameLength = audioArray.count

        if startIndex >= currentFrameLength, startIndex < 0 {
            Logging.error("startIndex is outside the buffer size")
            return nil
        }

        let endFrameIndex = min(audioArray.count, startIndex + frameLength)
        let actualFrameLength = endFrameIndex - startIndex

        let audioSamplesArray = try! MLMultiArray(shape: [NSNumber(value: frameLength)], dataType: .float32)

        if actualFrameLength > 0 {
            audioArray.withUnsafeBufferPointer { sourcePointer in
                audioSamplesArray.dataPointer.assumingMemoryBound(to: Float.self).advanced(by: 0).initialize(from: sourcePointer.baseAddress!.advanced(by: startIndex), count: actualFrameLength)
            }
        }

        // If the buffer is smaller than the desired frameLength, pad the rest with zeros using vDSP
        if actualFrameLength < frameLength {
            vDSP_vclr(audioSamplesArray.dataPointer.assumingMemoryBound(to: Float.self).advanced(by: actualFrameLength), 1, vDSP_Length(frameLength - actualFrameLength))
        }

        // Save the file for debugging
        if saveSegment {
            let desiredFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
            let newBuffer = AVAudioPCMBuffer(pcmFormat: desiredFormat, frameCapacity: AVAudioFrameCount(frameLength))!
            for i in 0..<frameLength {
                newBuffer.floatChannelData?[0][i] = audioSamplesArray[i].floatValue
            }
            newBuffer.frameLength = AVAudioFrameCount(frameLength)

            // Save to file
            let filename = "segment_\(Double(startIndex) / 16000.0).wav"
            var fileURL = getDownloadsDirectory().appendingPathComponent("WhisperKitSegments")
            fileURL = fileURL.appendingPathComponent(filename)
            do {
                try saveBuffer(newBuffer, to: fileURL)
                Logging.debug("Saved audio segment to \(fileURL)")
            } catch {
                Logging.debug("Could not save file: \(error)")
            }
        }

        return audioSamplesArray
    }

    static func createWaveformImage(fromAudio audioBuffer: [Float], withSize size: CGSize) -> CGImage {
        let width = Int(size.width)
        let height = Int(size.height)
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue

        var data = audioBuffer
        let dataProvider = CGDataProvider(data: NSData(bytes: &data, length: data.count * MemoryLayout<Float>.size))
        let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bytesPerPixel * bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: dataProvider!,
            decode: nil,
            shouldInterpolate: false,
            intent: CGColorRenderingIntent.defaultIntent
        )!
        return cgImage
    }
}

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
public class AudioProcessor: NSObject, AudioProcessing {
    public var audioEngine: AVAudioEngine?
    public var audioSamples: ContiguousArray<Float> = []
    public var audioEnergy: [(rel: Float, avg: Float, max: Float, min: Float)] = []
    public var relativeEnergyWindow: Int = 20
    public var relativeEnergy: [Float] {
        return self.audioEnergy.map { $0.rel }
    }

    public var audioBufferCallback: (([Float]) -> Void)?
    public var maxBufferLength = WhisperKit.sampleRate * WhisperKit.chunkLength // 30 seconds of audio at 16,000 Hz
    public var minBufferLength = Int(Double(WhisperKit.sampleRate) * 0.1) // 0.1 second of audio at 16,000 Hz

    // MARK: - Loading and conversion

    public static func loadAudio(fromPath audioFilePath: String) -> AVAudioPCMBuffer? {
        guard FileManager.default.fileExists(atPath: audioFilePath) else {
            Logging.error("Resource path does not exist \(audioFilePath)")
            return nil
        }

        var outputBuffer: AVAudioPCMBuffer?

        do {
            let audioFileURL = URL(fileURLWithPath: audioFilePath)
            let audioFile = try AVAudioFile(forReading: audioFileURL, commonFormat: .pcmFormatFloat32, interleaved: false)

            let sampleRate = audioFile.fileFormat.sampleRate
            let channelCount = audioFile.fileFormat.channelCount
            let frameLength = AVAudioFrameCount(audioFile.length)

            // If the audio file already meets the desired format, read directly into the output buffer
            if sampleRate == 16000 && channelCount == 1 {
                guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameLength) else {
                    Logging.error("Unable to create audio buffer")
                    return nil
                }
                try audioFile.read(into: buffer)
                outputBuffer = buffer
            } else {
                // Audio needs resampling to 16khz
                guard let buffer = resampleAudio(fromFile: audioFile, toSampleRate: 16000, channelCount: 1) else {
                    Logging.error("Unable to resample audio")
                    return nil
                }
                outputBuffer = buffer
            }

            if let buffer = outputBuffer {
                Logging.info("Audio source details - Sample Rate: \(sampleRate) Hz, Channel Count: \(channelCount), Frame Length: \(frameLength), Duration: \(Double(frameLength) / sampleRate)s")
                Logging.info("Audio buffer details - Sample Rate: \(buffer.format.sampleRate) Hz, Channel Count: \(buffer.format.channelCount), Frame Length: \(buffer.frameLength), Duration: \(Double(buffer.frameLength) / buffer.format.sampleRate)s")
            }
        } catch {
            Logging.error("Error loading audio file: \(error)")
            return nil
        }

        return outputBuffer
    }

    public static func resampleAudio(fromFile audioFile: AVAudioFile, toSampleRate sampleRate: Double, channelCount: AVAudioChannelCount) -> AVAudioPCMBuffer? {
        let newFrameLength = Int64((sampleRate / audioFile.fileFormat.sampleRate) * Double(audioFile.length))
        let outputFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: channelCount)!
        guard let converter = AVAudioConverter(from: audioFile.processingFormat, to: outputFormat) else {
            Logging.error("Failed to create audio converter")
            return nil
        }

        let frameCount = AVAudioFrameCount(audioFile.length)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameCount),
              let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: AVAudioFrameCount(newFrameLength))
        else {
            Logging.error("Unable to create buffers, likely due to unsupported file format")
            return nil
        }

        do {
            try audioFile.read(into: inputBuffer, frameCount: frameCount)
        } catch {
            Logging.error("Error reading audio file: \(error)")
            return nil
        }

        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            if inputBuffer.frameLength == 0 {
                outStatus.pointee = .endOfStream
                return nil
            } else {
                outStatus.pointee = .haveData
                return inputBuffer
            }
        }

        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
        switch status {
        case .error:
            if let conversionError = error {
                Logging.error("Error converting audio file: \(conversionError)")
            }
            return nil
        default: break
        }

        return outputBuffer
    }

    public static func resampleBuffer(_ buffer: AVAudioPCMBuffer, with converter: AVAudioConverter) throws -> AVAudioPCMBuffer {
        guard let convertedBuffer = AVAudioPCMBuffer(
            pcmFormat: converter.outputFormat,
            frameCapacity: AVAudioFrameCount(converter.outputFormat.sampleRate * Double(buffer.frameLength) / converter.inputFormat.sampleRate)
        ) else {
            throw WhisperError.audioProcessingFailed("Failed to create converted buffer")
        }

        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }

        converter.convert(to: convertedBuffer, error: nil, withInputFrom: inputBlock)
        return convertedBuffer
    }

    // MARK: - Utility

    public static func calculateEnergy(of signal: [Float]) -> (avg: Float, max: Float, min: Float) {
        var rmsEnergy: Float = 0.0
        var minEnergy: Float = 0.0
        var maxEnergy: Float = 0.0

        // Calculate the root mean square of the signal
        vDSP_rmsqv(signal, 1, &rmsEnergy, vDSP_Length(signal.count))

        // Calculate the maximum sample value of the signal
        vDSP_maxmgv(signal, 1, &maxEnergy, vDSP_Length(signal.count))

        // Calculate the minimum sample value of the signal
        vDSP_minmgv(signal, 1, &minEnergy, vDSP_Length(signal.count))

        return (rmsEnergy, maxEnergy, minEnergy)
    }

    public static func calculateRelativeEnergy(of signal: [Float], relativeTo reference: Float?) -> Float {
        let signalEnergy = calculateEnergy(of: signal).avg

        // Make sure reference is greater than 0
        // Default 1e-3 measured empirically in a silent room
        let referenceEnergy = max(1e-8, reference ?? 1e-3)

        // Convert to dB
        let dbEnergy = 20 * log10(signalEnergy)
        let refEnergy = 20 * log10(referenceEnergy)

        // Normalize based on reference
        // Note: since signalEnergy elements are floats from 0 to 1, max (full volume) is always 0dB
        let normalizedEnergy = rescale(value: dbEnergy, min: refEnergy, max: 0)

        // Clamp from 0 to 1
        return max(0, min(normalizedEnergy, 1))
    }

    public static func convertBufferToArray(buffer: AVAudioPCMBuffer) -> [Float] {
        let start = buffer.floatChannelData?[0]
        let count = Int(buffer.frameLength)
        let convertedArray = Array(UnsafeBufferPointer(start: start, count: count))
        return convertedArray
    }

    public static func requestRecordPermission() async -> Bool {
        await AVAudioApplication.requestRecordPermission()
    }

    deinit {
        stopRecording()
    }
}

// MARK: - Streaming

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
public extension AudioProcessor {
    /// We have a new buffer, process and store it.
    /// Note: Assumes audio is 16khz mono
    func processBuffer(_ buffer: [Float]) {
        audioSamples.append(contentsOf: buffer)

        // Find the lowest average energy of the last 20 buffers ~2 seconds
        let minAvgEnergy = self.audioEnergy.suffix(20).reduce(Float.infinity) { min($0, $1.avg) }
        let relativeEnergy = Self.calculateRelativeEnergy(of: buffer, relativeTo: minAvgEnergy)

        // Update energy for buffers with valid data
        let signalEnergy = Self.calculateEnergy(of: buffer)
        let newEnergy = (relativeEnergy, signalEnergy.avg, signalEnergy.max, signalEnergy.min)
        self.audioEnergy.append(newEnergy)

        // Call the callback with the new buffer
        audioBufferCallback?(buffer)

        // Print the current size of the audio buffer
        if self.audioSamples.count % (minBufferLength * Int(relativeEnergyWindow)) == 0 {
            Logging.debug("Current audio size: \(self.audioSamples.count) samples, most recent buffer: \(buffer.count) samples, most recent energy: \(newEnergy)")
        }
    }

    func setupEngine() throws -> AVAudioEngine {
        let audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Desired format (16,000 Hz, 1 channel)
        guard let desiredFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(WhisperKit.sampleRate),
            channels: AVAudioChannelCount(1),
            interleaved: false
        ) else {
            throw WhisperError.audioProcessingFailed("Failed to create desired format")
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: desiredFormat) else {
            throw WhisperError.audioProcessingFailed("Failed to create audio converter")
        }

        let bufferSize = AVAudioFrameCount(minBufferLength) // 100ms - 400ms supported
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] (buffer: AVAudioPCMBuffer, _: AVAudioTime) in
            guard let self = self else { return }
            var buffer = buffer
            if !buffer.format.sampleRate.isEqual(to: Double(WhisperKit.sampleRate)) {
                do {
                    buffer = try Self.resampleBuffer(buffer, with: converter)
                } catch {
                    Logging.error("Failed to resample buffer: \(error)")
                    return
                }
            }

            let newBufferArray = Self.convertBufferToArray(buffer: buffer)
            self.processBuffer(newBufferArray)
        }

        audioEngine.prepare()
        try audioEngine.start()

        return audioEngine
    }

    func purgeAudioSamples(keepingLast keep: Int) {
        if audioSamples.count > keep {
            audioSamples.removeFirst(audioSamples.count - keep)
        }
    }

    func startRecordingLive(callback: (([Float]) -> Void)? = nil) throws {
        audioSamples = []
        audioEnergy = []

        // TODO: implement selecting input device

        audioEngine = try setupEngine()

        // Set the callback
        audioBufferCallback = callback
    }

    func pauseRecording() {
        audioEngine?.pause()
    }

    func stopRecording() {
        // Remove the tap on any attached node
        audioEngine?.attachedNodes.forEach { node in
            node.removeTap(onBus: 0)
        }

        // Stop the audio engine
        audioEngine?.stop()
        audioEngine = nil
    }
}
