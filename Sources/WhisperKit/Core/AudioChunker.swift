//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import Accelerate
import AVFoundation

/// Responsible for chunking audio into smaller pieces
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
protocol AudioChunking {
    func chunkAll(audioArray: [Float], maxSampleLength: Int, decodeOptions: DecodingOptions?) async throws -> [[Float]]
}

/// A audio chunker that splits audio into smaller pieces based on voice activity detection
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
final class VADAudioChunker {
    // prevent hallucinations at the end of the clip by stopping up to 1.0s early
    private let windowPadding: Int
    private let vad = EnergyVAD()

    init(windowPadding: Int = 16000) {
        self.windowPadding = windowPadding
    }

    private func splitOnMiddleOfLongestSilence(audioArray: [Float], startIndex: Int, endIndex: Int) -> Int {
        // NOTE: we want to check just the 2nd part for the silence to attempt to get closest to a max length chunk
        let audioMidIndex = startIndex + (endIndex - startIndex) / 2
        let vadAudioSlice = Array(audioArray[audioMidIndex..<endIndex])
        let voiceActivity = vad.voiceActivity(in: vadAudioSlice)
        if let silence = vad.findLongestSilence(in: voiceActivity) {
            // if silence is detected we take the middle point of the silent chunk
            let silenceMidIndex = silence.startIndex + (silence.endIndex - silence.startIndex) / 2
            return audioMidIndex + vad.voiceActivityIndexToAudioSampleIndex(silenceMidIndex)
        }
        return endIndex
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
extension VADAudioChunker {
    func chunkAll(audioArray: [Float], maxChunkLength: Int, decodeOptions: DecodingOptions?) async throws -> [[Float]] {
        // If the audio array length is less than or equal to maxLength, return it as a single chunk
        if audioArray.count <= maxChunkLength {
            return [audioArray]
        }

        let seekClips = prepareSeekClips(contentFrames: audioArray.count, decodeOptions: decodeOptions)
        var chunkedAudioSamples = [[Float]]()
        for (seekClipStart, seekClipEnd) in seekClips {
            // Loop through the current clip until we reach the end
            // Typically this will be the full audio file, unless seek points are explicitly provided
            var startIndex = seekClipStart
            while startIndex < seekClipEnd - windowPadding {
                let currentFrameLength = audioArray.count
                if startIndex >= currentFrameLength, startIndex < 0 {
                    throw WhisperError.audioProcessingFailed("startIndex is outside the buffer size")
                }
                // Adjust the end index based on VAD
                let endIndex = splitOnMiddleOfLongestSilence(
                    audioArray: audioArray,
                    startIndex: startIndex,
                    endIndex: min(audioArray.count, startIndex + maxChunkLength)
                )
                guard endIndex > startIndex else {
                    break
                }
                Logging.debug("Found chunk from \(formatTimestamp(Float(startIndex) / Float(WhisperKit.sampleRate))) to \(formatTimestamp(Float(endIndex) / Float(WhisperKit.sampleRate)))")
                let audioSlice = Array(audioArray[startIndex..<endIndex])
                chunkedAudioSamples.append(audioSlice)
                startIndex = endIndex
            }
        }
        return chunkedAudioSamples
    }
}
