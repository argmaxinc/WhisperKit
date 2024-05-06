//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import Accelerate
import AVFoundation

/// Responsible for chunking audio into smaller pieces
protocol AudioChunking {
    func chunkAll(audioArray: [Float], frameLength: Int, decodeOptions: DecodingOptions?) async throws -> [[Float]]
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

    private func prepareSeekClips(contentFrames: Int, decodeOptions: DecodingOptions?) -> [(start: Int, end: Int)] {
        let options = decodeOptions ?? DecodingOptions()
        var seekPoints: [Int] = options.clipTimestamps.map { Int(round($0 * Float(WhisperKit.sampleRate))) }
        if seekPoints.count == 0 {
            seekPoints.append(0)
        }
        if seekPoints.count % 2 == 1 {
            seekPoints.append(contentFrames)
        }
        var seekClips: [(start: Int, end: Int)] = []
        for i in stride(from: 0, to: seekPoints.count, by: 2) {
            let start = seekPoints[i]
            let end = i + 1 < seekPoints.count ? seekPoints[i + 1] : contentFrames
            seekClips.append((start, end))
        }
        return seekClips
    }

    private func adjustEndIndex(audioArray: [Float], startIndex: Int, endIndex: Int) -> Int {
        // NOTE: we want to check just the 2nd part for the silecne
        let audioMidIndex = startIndex + (endIndex - startIndex) / 2
        let vadAudioSlice = Array(audioArray[audioMidIndex..<endIndex])
        let voiceActivity = vad.voiceActivity(in: vadAudioSlice)
        if let silence = vad.findLongestSilence(in: voiceActivity) {
            // if silence is detected we take the middle point of the silent chunk
            let silenceMidIndex = silence.startIndex + (silence.endIndex - silence.startIndex) / 2
            return audioMidIndex + vad.voiceActivityIndexToAudioIndex(silenceMidIndex)
        }
        return endIndex
    }
}

extension VADAudioChunker {
    func chunkAll(audioArray: [Float], frameLength: Int, decodeOptions: DecodingOptions?) async throws -> [[Float]] {
        let seekClips = prepareSeekClips(contentFrames: audioArray.count, decodeOptions: decodeOptions)
        var result = [[Float]]()
        for (seekClipStart, seekClipEnd) in seekClips {
            // Loop through the current clip until we reach the end
            // Typically this will be the full audio file, unless seek points are explicitly provided
            var startIndex = seekClipStart
            while startIndex < seekClipEnd - windowPadding {
                let currentFrameLength = audioArray.count
                if startIndex >= currentFrameLength, startIndex < 0 {
                    throw WhisperError.audioProcessingFailed("startIndex is outside the buffer size")
                }
                // adjust the end index based on VAD
                let endIndex = adjustEndIndex(
                    audioArray: audioArray,
                    startIndex: startIndex,
                    endIndex: min(audioArray.count, startIndex + frameLength)
                )
                guard endIndex > startIndex else {
                    break
                }
                var audioSlice = Array(audioArray[startIndex..<endIndex])
                // If the buffer is smaller than the desired frameLength, pad the rest with zeros
                if audioSlice.count < frameLength {
                    audioSlice.append(contentsOf: Array(repeating: 0, count: frameLength - audioSlice.count))
                }
                result.append(audioSlice)
                startIndex = endIndex
            }
        }
        return result
    }
}
