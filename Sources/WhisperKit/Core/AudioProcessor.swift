//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import AVFoundation
import CoreAudio
import CoreML

// Core Audio Device
#if os(macOS)
public typealias DeviceID = AudioDeviceID
#else
public typealias DeviceID = String
#endif

public struct AudioDevice: Identifiable, Hashable {
    public let id: DeviceID
    public let name: String
}

public protocol AudioProcessing {
    /// Loads audio data from a specified file path.
    /// - Parameter audioFilePath: The file path of the audio file.
    /// - Returns: `AVAudioPCMBuffer` containing the audio data.
    static func loadAudio(fromPath audioFilePath: String) throws -> AVAudioPCMBuffer

    /// Loads and converts audio data from a specified file paths.
    /// - Parameter audioPaths: The file paths of the audio files.
    /// - Returns: Array of `.success` if the file was loaded and converted correctly, otherwise `.failure`
    static func loadAudio(at audioPaths: [String]) async -> [Result<[Float], Swift.Error>]

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
    func startRecordingLive(inputDeviceID: DeviceID?, callback: (([Float]) -> Void)?) throws

    /// Pause recording
    func pauseRecording()

    /// Stops recording and cleans up resources
    func stopRecording()
}

/// Overrideable default methods for AudioProcessing
public extension AudioProcessing {
    func startRecordingLive(inputDeviceID: DeviceID? = nil, callback: (([Float]) -> Void)?) throws {
        try startRecordingLive(inputDeviceID: inputDeviceID, callback: callback)
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

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
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

    public static func loadAudio(fromPath audioFilePath: String) throws -> AVAudioPCMBuffer {
        guard FileManager.default.fileExists(atPath: audioFilePath) else {
            throw WhisperError.loadAudioFailed("Resource path does not exist \(audioFilePath)")
        }

        let audioFileURL = URL(fileURLWithPath: audioFilePath)
        let audioFile = try AVAudioFile(forReading: audioFileURL, commonFormat: .pcmFormatFloat32, interleaved: false)

        let sampleRate = audioFile.fileFormat.sampleRate
        let channelCount = audioFile.fileFormat.channelCount
        let frameLength = AVAudioFrameCount(audioFile.length)

        let outputBuffer: AVAudioPCMBuffer
        // If the audio file already meets the desired format, read directly into the output buffer
        if sampleRate == 16000 && channelCount == 1 {
            guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameLength) else {
                throw WhisperError.loadAudioFailed("Unable to create audio buffer")
            }
            try audioFile.read(into: buffer)
            outputBuffer = buffer
        } else {
            // Audio needs resampling to 16khz
            guard let buffer = resampleAudio(fromFile: audioFile, toSampleRate: 16000, channelCount: 1) else {
                throw WhisperError.loadAudioFailed("Unable to resample audio")
            }
            outputBuffer = buffer
        }
        Logging.debug("Audio source details - Sample Rate: \(sampleRate) Hz, Channel Count: \(channelCount), Frame Length: \(frameLength), Duration: \(Double(frameLength) / sampleRate)s")
        Logging.debug("Audio buffer details - Sample Rate: \(outputBuffer.format.sampleRate) Hz, Channel Count: \(outputBuffer.format.channelCount), Frame Length: \(outputBuffer.frameLength), Duration: \(Double(outputBuffer.frameLength) / outputBuffer.format.sampleRate)s")
        return outputBuffer
    }

    public static func loadAudio(at audioPaths: [String]) async -> [Result<[Float], Swift.Error>] {
        await withTaskGroup(of: [(index: Int, result: Result<[Float], Swift.Error>)].self) { taskGroup -> [Result<[Float], Swift.Error>] in
            for (index, audioPath) in audioPaths.enumerated() {
                taskGroup.addTask {
                    do {
                        let audioBuffer = try AudioProcessor.loadAudio(fromPath: audioPath)
                        let audio = AudioProcessor.convertBufferToArray(buffer: audioBuffer)
                        return [(index: index, result: .success(audio))]
                    } catch {
                        return [(index: index, result: .failure(error))]
                    }
                }
            }
            var batchResult = [(index: Int, result: Result<[Float], Swift.Error>)]()
            for await result in taskGroup {
                batchResult.append(contentsOf: result)
            }
            batchResult.sort(by: { $0.index < $1.index })
            return batchResult.map { $0.result }
        }
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
        // NOTE: since signalEnergy elements are floats from 0 to 1, max (full volume) is always 0dB
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
        if #available(macOS 14, iOS 17, *) {
            return await AVAudioApplication.requestRecordPermission()
        } else {
            #if os(watchOS)
            // watchOS does not support AVCaptureDevice
            return true
            #else
            let microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)
            switch microphoneStatus {
                case .notDetermined:
                return await withCheckedContinuation { continuation in
                    AVCaptureDevice.requestAccess(for: .audio) { granted in
                        continuation.resume(returning: granted)
                    }
                }
                case .restricted, .denied:
                Logging.error("Microphone access denied")
                return false
                case .authorized:
                return true
                @unknown default:
                Logging.error("Unknown authorization status")
                return false
            }
            #endif
        }
    }

    #if os(macOS)
    public static func getAudioDevices() -> [AudioDevice] {
        var devices = [AudioDevice]()

        var propertySize: UInt32 = 0
        var status: OSStatus = noErr

        // Get the number of devices
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        status = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &propertySize
        )
        if status != noErr {
            Logging.error("Error: Unable to get the number of audio devices.")
            return devices
        }

        // Get the device IDs
        let deviceCount = Int(propertySize) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
        status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &propertySize,
            &deviceIDs
        )
        if status != noErr {
            Logging.error("Error: Unable to get the audio device IDs.")
            return devices
        }

        // Get device info for each device
        for deviceID in deviceIDs {
            var deviceName = ""
            var inputChannels = 0

            // Get device name
            var propertySize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
            var name: Unmanaged<CFString>?
            propertyAddress.mSelector = kAudioDevicePropertyDeviceNameCFString

            status = AudioObjectGetPropertyData(
                deviceID,
                &propertyAddress,
                0,
                nil,
                &propertySize,
                &name
            )
            if status == noErr, let deviceNameCF = name?.takeUnretainedValue() as String? {
                deviceName = deviceNameCF
            }

            // Get input channels
            propertyAddress.mSelector = kAudioDevicePropertyStreamConfiguration
            propertyAddress.mScope = kAudioDevicePropertyScopeInput
            status = AudioObjectGetPropertyDataSize(deviceID, &propertyAddress, 0, nil, &propertySize)
            if status == noErr {
                let bufferListPointer = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
                defer { bufferListPointer.deallocate() }
                status = AudioObjectGetPropertyData(deviceID, &propertyAddress, 0, nil, &propertySize, bufferListPointer)
                if status == noErr {
                    let bufferList = UnsafeMutableAudioBufferListPointer(bufferListPointer)
                    for buffer in bufferList {
                        inputChannels += Int(buffer.mNumberChannels)
                    }
                }
            }

            if inputChannels > 0 {
                devices.append(AudioDevice(id: deviceID, name: deviceName))
            }
        }

        return devices
    }
    #endif

    deinit {
        stopRecording()
    }
}

// MARK: - Streaming

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public extension AudioProcessor {
    /// We have a new buffer, process and store it.
    /// NOTE: Assumes audio is 16khz mono
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

    #if os(macOS)
    func assignAudioInput(inputNode: AVAudioInputNode, inputDeviceID: AudioDeviceID) {
        guard let audioUnit = inputNode.audioUnit else {
            Logging.error("Failed to access the audio unit of the input node.")
            return
        }

        var inputDeviceID = inputDeviceID

        let error = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &inputDeviceID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )

        if error != noErr {
            Logging.error("Error setting Audio Unit property: \(error)")
        } else {
            Logging.info("Successfully set input device.")
        }
    }
    #endif

    /// Attempts to setup the shared audio session if available on the device's OS
    func setupAudioSessionForDevice() throws {
        #if !os(macOS) // AVAudioSession is not available on macOS

        #if !os(watchOS) // watchOS does not support .defaultToSpeaker
        let options: AVAudioSession.CategoryOptions = [.defaultToSpeaker, .allowBluetooth]
        #else
        let options: AVAudioSession.CategoryOptions = .mixWithOthers
        #endif

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, options: options)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch let error as NSError {
            throw WhisperError.audioProcessingFailed("Failed to set up audio session: \(error)")
        }
        #endif
    }

    func setupEngine(inputDeviceID: DeviceID? = nil) throws -> AVAudioEngine {
        let audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode

        #if os(macOS)
        if let inputDeviceID = inputDeviceID {
            assignAudioInput(inputNode: inputNode, inputDeviceID: inputDeviceID)
        }
        #endif

        let hardwareSampleRate = audioEngine.inputNode.inputFormat(forBus: 0).sampleRate
        let inputFormat = inputNode.outputFormat(forBus: 0)

        guard let nodeFormat = AVAudioFormat(commonFormat: inputFormat.commonFormat, sampleRate: hardwareSampleRate, channels: inputFormat.channelCount, interleaved: inputFormat.isInterleaved) else {
            throw WhisperError.audioProcessingFailed("Failed to create node format")
        }

        // Desired format (16,000 Hz, 1 channel)
        guard let desiredFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(WhisperKit.sampleRate), channels: AVAudioChannelCount(1), interleaved: false) else {
            throw WhisperError.audioProcessingFailed("Failed to create desired format")
        }

        guard let converter = AVAudioConverter(from: nodeFormat, to: desiredFormat) else {
            throw WhisperError.audioProcessingFailed("Failed to create audio converter")
        }

        let bufferSize = AVAudioFrameCount(minBufferLength) // 100ms - 400ms supported
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: nodeFormat) { [weak self] (buffer: AVAudioPCMBuffer, _: AVAudioTime) in
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

    func startRecordingLive(inputDeviceID: DeviceID? = nil, callback: (([Float]) -> Void)? = nil) throws {
        audioSamples = []
        audioEnergy = []

        try? setupAudioSessionForDevice()

        audioEngine = try setupEngine(inputDeviceID: inputDeviceID)

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
