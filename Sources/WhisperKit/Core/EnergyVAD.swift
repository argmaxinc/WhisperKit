//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import Foundation

/// Voice activity detection based on energy threshold
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
final class EnergyVAD {
    var sampleRate: Int
    var frameLengthSamples: Int
    var frameOverlapSamples: Int
    var energyThreshold: Float

    /// Initialize a new EnergyVAD instance
    /// - Parameters:
    ///   - sampleRate: Audio sample rate
    ///   - frameLength: Frame length in seconds
    ///   - frameOverlap: frame overlap in seconds, this will include `frameOverlap` length audio into the `frameLength` and is helpful to catch audio that starts exactly at chunk boundaries
    ///   - energyThreshold: minimal energy threshold
    convenience init(
        sampleRate: Int = WhisperKit.sampleRate,
        frameLength: Float = 0.1,
        frameOverlap: Float = 0.0,
        energyThreshold: Float = 0.02
    ) {
        self.init(
            sampleRate: sampleRate,
            // Compute frame length and overlap in number of samples
            frameLengthSamples: Int(frameLength * Float(sampleRate)),
            frameOverlapSamples: Int(frameOverlap * Float(sampleRate)),
            energyThreshold: energyThreshold
        )
    }

    required init(
        sampleRate: Int = 16000,
        frameLengthSamples: Int,
        frameOverlapSamples: Int = 0,
        energyThreshold: Float = 0.02
    ) {
        self.sampleRate = sampleRate
        self.frameLengthSamples = frameLengthSamples
        self.frameOverlapSamples = frameOverlapSamples
        self.energyThreshold = energyThreshold
    }

    func voiceActivity(in waveform: [Float]) -> [Bool] {
        let chunkRatio = Double(waveform.count) / Double(frameLengthSamples)

        // Round up if uneven, the final chunk will not be a full `frameLengthSamples` long
        let count = Int(chunkRatio.rounded(.up))

        let chunkedVoiceActivity = AudioProcessor.calculateVoiceActivityInChunks(
            of: waveform,
            chunkCount: count,
            frameLengthSamples: frameLengthSamples,
            frameOverlapSamples: frameOverlapSamples,
            energyThreshold: energyThreshold
        )

        return chunkedVoiceActivity
    }

    func calculateActiveChunks(in waveform: [Float]) -> [(startIndex: Int, endIndex: Int)] {
        let vad: [Bool] = voiceActivity(in: waveform)
        var result = [(startIndex: Int, endIndex: Int)]()

        // Temporary variables to hold the start of the current non-silent segment
        var currentStartIndex: Int?

        for (index, vadChunk) in vad.enumerated() {
            if vadChunk {
                let chunkStart = index * frameLengthSamples
                let chunkEnd = min(chunkStart + frameLengthSamples, waveform.count)

                if currentStartIndex != nil {
                    // If we already have a starting point, just update the end point in the last added segment
                    result[result.count - 1].endIndex = chunkEnd
                } else {
                    // If there is no current start, this is a new segment
                    currentStartIndex = chunkStart
                    result.append((startIndex: chunkStart, endIndex: chunkEnd))
                }
            } else {
                // Reset currentStartIndex when encountering a silent chunk
                currentStartIndex = nil
            }
        }

        return result
    }

    func voiceActivityClipTimestamps(in waveform: [Float]) -> [Float] {
        let nonSilentChunks = calculateActiveChunks(in: waveform)
        var clipTimestamps = [Float]()

        for chunk in nonSilentChunks {
            let startTimestamp = Float(chunk.startIndex) / Float(sampleRate)
            let endTimestamp = Float(chunk.endIndex) / Float(sampleRate)

            clipTimestamps.append(contentsOf: [startTimestamp, endTimestamp])
        }

        return clipTimestamps
    }

    func calculateNonSilentSeekClips(in waveform: [Float]) -> [(start: Int, end: Int)] {
        let clipTimestamps = voiceActivityClipTimestamps(in: waveform)
        let options = DecodingOptions(clipTimestamps: clipTimestamps)
        let seekClips = prepareSeekClips(contentFrames: waveform.count, decodeOptions: options)
        return seekClips
    }

    func calculateSeekTimestamps(in waveform: [Float]) -> [(startTime: Float, endTime: Float)] {
        let nonSilentChunks = calculateActiveChunks(in: waveform)
        var seekTimestamps = [(startTime: Float, endTime: Float)]()

        for chunk in nonSilentChunks {
            let startTimestamp = Float(chunk.startIndex) / Float(sampleRate)
            let endTimestamp = Float(chunk.endIndex) / Float(sampleRate)

            seekTimestamps.append(contentsOf: [(startTime: startTimestamp, endTime: endTimestamp)])
        }

        return seekTimestamps
    }

    func voiceActivityIndexToAudioSampleIndex(_ index: Int) -> Int {
        return index * frameLengthSamples
    }

    func voiceActivityIndexToSeconds(_ index: Int) -> Float {
        return Float(voiceActivityIndexToAudioSampleIndex(index)) / Float(sampleRate)
    }

    func findLongestSilence(in vadResult: [Bool]) -> (startIndex: Int, endIndex: Int)? {
        var longestStartIndex: Int?
        var longestEndIndex: Int?
        var longestCount = 0
        var index = 0
        while index < vadResult.count {
            let value = vadResult[index]
            if value {
                // found non-silence, skip
                index += 1
            } else {
                // found beginning of silence, find the end
                var endIndex = index
                while endIndex < vadResult.count, !vadResult[endIndex] {
                    endIndex += 1
                }
                let count = endIndex - index
                if count > longestCount {
                    longestCount = count
                    longestStartIndex = index
                    longestEndIndex = endIndex
                }
                index = endIndex
            }
        }
        if let longestStartIndex, let longestEndIndex {
            return (startIndex: longestStartIndex, endIndex: longestEndIndex)
        } else {
            return nil
        }
    }
}
