//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import Accelerate

public struct AudioChunk {
    var audioArray: [Float]
    var startIndex: Int
    var endIndex: Int
}

/// Responsible for chunking audio into smaller pieces
public protocol AudioChunking {
    func chunkAll(audioArray: [Float], frameLength: Int, decodeOptions: DecodingOptions?) async throws -> [AudioChunk]
}

/// A simple audio chunker that splits audio into even 30s pieces, if audio is too short, it pads with zeros
public final class SimpleAudioChunker {
    // prevent hallucinations at the end of the clip by stopping up to 1.0s early
    private let windowPadding: Int

    public init(windowPadding: Int = 16000) {
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
}

public extension SimpleAudioChunker {
    func chunkAll(audioArray: [Float], frameLength: Int, decodeOptions: DecodingOptions?) async throws -> [AudioChunk] {
        let seekClips = prepareSeekClips(contentFrames: audioArray.count, decodeOptions: decodeOptions)
        var result = [AudioChunk]()
        for (seekClipStart, seekClipEnd) in seekClips {
            // Loop through the current clip until we reach the end
            // Typically this will be the full audio file, unless seek points are explicitly provided
            var startIndex = seekClipStart
            while startIndex < seekClipEnd - windowPadding {
                let currentFrameLength = audioArray.count
                if startIndex >= currentFrameLength, startIndex < 0 {
                    throw WhisperError.audioProcessingFailed("startIndex is outside the buffer size")
                }
                let endIndex = min(audioArray.count, startIndex + frameLength)
                guard endIndex > startIndex else {
                    break
                }
                var audioSlice = Array(audioArray[startIndex..<endIndex])
                // If the buffer is smaller than the desired frameLength, pad the rest with zeros
                if audioSlice.count < frameLength {
                    audioSlice.append(contentsOf: Array(repeating: 0, count: frameLength - audioSlice.count))
                }
                result.append(AudioChunk(audioArray: audioSlice, startIndex: startIndex, endIndex: endIndex))
                startIndex = endIndex
            }
        }
        return result
    }
}
