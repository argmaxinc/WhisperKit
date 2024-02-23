//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

extension AudioStreamTranscriber {
    public struct State {
        public var isRecording: Bool = false
        public var currentFallbacks: Int = 0
        public var lastBufferSize: Int = 0
        public var lastConfirmedSegmentEndSeconds: Float = 0
        public var bufferEnergy: [Float] = []
        public var currentText: String = ""
        public var confirmedSegments: [TranscriptionSegment] = []
        public var unconfirmedSegments: [TranscriptionSegment] = []
        public var unconfirmedText: [String] = []
    }
}

public typealias AudioStreamTranscriberCallback = (AudioStreamTranscriber.State, AudioStreamTranscriber.State) -> Void

/// Responsible for streaming audio from the microphone, processing it, and transcribing it in real-time.
public actor AudioStreamTranscriber {
    private var state: AudioStreamTranscriber.State = .init() {
        didSet {
            stateChangeCallback?(oldValue, state)
        }
    }
    private let stateChangeCallback: AudioStreamTranscriberCallback?

    private let requiredSegmentsForConfirmation: Int
    private let useVAD: Bool
    private let silenceThreshold: Float
    private let compressionCheckWindow: Int
    private let audioProcessor: any AudioProcessing
    private let transcriber: any Transcriber
    private let decodingOptions: DecodingOptions

    public init(
        audioProcessor: any AudioProcessing,
        transcriber: any Transcriber,
        decodingOptions: DecodingOptions,
        requiredSegmentsForConfirmation: Int = 2,
        silenceThreshold: Float = 0.3,
        compressionCheckWindow: Int = 20,
        useVAD: Bool = true,
        stateChangeCallback: AudioStreamTranscriberCallback?
    ) {
        self.audioProcessor = audioProcessor
        self.transcriber = transcriber
        self.decodingOptions = decodingOptions
        self.requiredSegmentsForConfirmation = requiredSegmentsForConfirmation
        self.silenceThreshold = silenceThreshold
        self.compressionCheckWindow = compressionCheckWindow
        self.useVAD = useVAD
        self.stateChangeCallback = stateChangeCallback
    }

    public func startStreamTranscription() async throws {
        guard !state.isRecording else { return }
        guard await AudioProcessor.requestRecordPermission() else {
            Logging.error("Microphone access was not granted.")
            return
        }
        state.isRecording = true
        try audioProcessor.startRecordingLive { [weak self] _ in
            Task { [weak self] in
                await self?.onAudioBufferCallback()
            }
        }
        await realtimeLoop()
        Logging.info("Realtime transcription has started")
    }

    public func stopStreamTranscription() {
        state.isRecording = false
        audioProcessor.stopRecording()
        Logging.info("Realtime transcription has ended")
    }

    private func realtimeLoop() async {
        while state.isRecording {
            do {
                try await transcribeCurrentBuffer()
            } catch {
                Logging.error("Error: \(error.localizedDescription)")
                break
            }
        }
    }

    private func onAudioBufferCallback() {
        state.bufferEnergy = audioProcessor.relativeEnergy
    }

    private func onProgressCallback(_ progress: TranscriptionProgress) {
        let fallbacks = Int(progress.timings.totalDecodingFallbacks)
        if progress.text.count < state.currentText.count {
            if fallbacks == state.currentFallbacks {
                state.unconfirmedText.append(state.currentText)
            } else {
                Logging.info("Fallback occured: \(fallbacks)")
            }
        }
        state.currentText = progress.text
        state.currentFallbacks = fallbacks
    }

    private func transcribeCurrentBuffer() async throws {
        // Retrieve the current audio buffer from the audio processor
        let currentBuffer = audioProcessor.audioSamples

        // Calculate the size and duration of the next buffer segment
        let nextBufferSize = currentBuffer.count - state.lastBufferSize
        let nextBufferSeconds = Float(nextBufferSize) / Float(WhisperKit.sampleRate)

        // Only run the transcribe if the next buffer has at least 1 second of audio
        guard nextBufferSeconds > 1 else {
            if state.currentText == "" {
                state.currentText = "Waiting for speech..."
            }
            return try await Task.sleep(nanoseconds: 100_000_000) // sleep for 100ms for next buffer
        }

        if useVAD {
            // Retrieve the current relative energy values from the audio processor
            let currentRelativeEnergy = audioProcessor.relativeEnergy

            // Calculate the number of energy values to consider based on the duration of the next buffer
            // Each energy value corresponds to 1 buffer length (100ms of audio), hence we divide by 0.1
            let energyValuesToConsider = Int(nextBufferSeconds / 0.1)

            // Extract the relevant portion of energy values from the currentRelativeEnergy array
            let nextBufferEnergies = currentRelativeEnergy.suffix(energyValuesToConsider)

            // Determine the number of energy values to check for voice presence
            // Considering up to the last 1 second of audio, which translates to 10 energy values
            let numberOfValuesToCheck = max(10, nextBufferEnergies.count - 10)

            // Check if any of the energy values in the considered range exceed the silence threshold
            // This indicates the presence of voice in the buffer
            let voiceDetected = nextBufferEnergies.prefix(numberOfValuesToCheck).contains { $0 > Float(silenceThreshold) }

            // Only run the transcribe if the next buffer has voice
            if !voiceDetected {
                Logging.debug("No voice detected, skipping transcribe")
                if state.currentText == "" {
                    state.currentText = "Waiting for speech..."
                }
                // Sleep for 100ms and check the next buffer
                return try await Task.sleep(nanoseconds: 100_000_000)
            }
        }

        // Run transcribe
        state.lastBufferSize = currentBuffer.count

        let transcription = try await transcribeAudioSamples(Array(currentBuffer))

        state.currentText = ""
        state.unconfirmedText = []
        guard let segments = transcription?.segments else {
            return
        }

        // Logic for moving segments to confirmedSegments
        if segments.count > requiredSegmentsForConfirmation {
            // Calculate the number of segments to confirm
            let numberOfSegmentsToConfirm = segments.count - requiredSegmentsForConfirmation

            // Confirm the required number of segments
            let confirmedSegmentsArray = Array(segments.prefix(numberOfSegmentsToConfirm))
            let remainingSegments = Array(segments.suffix(requiredSegmentsForConfirmation))

            // Update lastConfirmedSegmentEnd based on the last confirmed segment
            if let lastConfirmedSegment = confirmedSegmentsArray.last, lastConfirmedSegment.end > state.lastConfirmedSegmentEndSeconds {
                state.lastConfirmedSegmentEndSeconds = lastConfirmedSegment.end

                // Add confirmed segments to the confirmedSegments array
                if !state.confirmedSegments.contains(confirmedSegmentsArray) {
                    state.confirmedSegments.append(contentsOf: confirmedSegmentsArray)
                }
            }

            // Update transcriptions to reflect the remaining segments
            state.unconfirmedSegments = remainingSegments
        } else {
            // Handle the case where segments are fewer or equal to required
            state.unconfirmedSegments = segments
        }
    }

    private func transcribeAudioSamples(_ samples: [Float]) async throws -> TranscriptionResult? {
        var options = decodingOptions
        options.clipTimestamps = [state.lastConfirmedSegmentEndSeconds]
        let checkWindow = compressionCheckWindow
        return try await transcriber.transcribe(audioArray: samples, decodeOptions: options) { [weak self] progress in
            Task { [weak self] in
                await self?.onProgressCallback(progress)
            }
            return AudioStreamTranscriber.shouldStopEarly(progress: progress, options: options, compressionCheckWindow: checkWindow)
        }
    }

    private static func shouldStopEarly(
        progress: TranscriptionProgress,
        options: DecodingOptions,
        compressionCheckWindow: Int
    ) -> Bool? {
        let currentTokens = progress.tokens
        if currentTokens.count > compressionCheckWindow {
            let checkTokens: [Int] = currentTokens.suffix(compressionCheckWindow)
            let compressionRatio = compressionRatio(of: checkTokens)
            if compressionRatio > options.compressionRatioThreshold ?? 0.0 {
                return false
            }
        }
        if let avgLogprob = progress.avgLogprob, let logProbThreshold = options.logProbThreshold {
            if avgLogprob < logProbThreshold {
                return false
            }
        }
        return nil
    }
}
