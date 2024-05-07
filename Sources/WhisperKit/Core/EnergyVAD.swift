//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import Accelerate

/// Voice activity detection based on energy threshold
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
final class EnergyVAD {
    var sampleRate: Int
    var frameLength: Int
    var frameShift: Int
    var energyThreshold: Float

    /// Initialize a new EnergyVAD instance
    /// - Parameters:
    ///   - sampleRate: audio sample rate
    ///   - frameLength: frame length in milliseconds
    ///   - frameShift: frame shift in milliseconds
    ///   - energyThreshold: minimal energy threshold
    init(
        sampleRate: Int = 16000,
        frameLength: Int = 25,
        frameShift: Int = 20,
        energyThreshold: Float = 0.05
    ) {
        self.sampleRate = sampleRate
        // Compute frame length and frame shift in number of samples (not milliseconds)
        self.frameLength = frameLength * sampleRate / 1000
        self.frameShift = frameShift * sampleRate / 1000
        self.energyThreshold = energyThreshold
    }

    func voiceActivity(in waveform: [Float]) -> [Bool] {
        let count = (waveform.count - frameLength + frameShift) / frameShift
        if count < 0 {
            return []
        }
        return AudioProcessor.calculateVoiceActivityInChunks(
            of: waveform,
            chunkCount: count,
            frameLength: frameLength,
            frameShift: frameShift,
            energyThreshold: energyThreshold
        )
    }

    func calculateNonSilentChunks(in waveform: [Float]) -> [(startIndex: Int, endIndex: Int)] {
        let vad = voiceActivity(in: waveform)
        var result = [(startIndex: Int, endIndex: Int)]()
        for (index, vadChunk) in vad.enumerated() {
            if vadChunk {
                result.append((startIndex: index * frameShift, endIndex: index * frameShift + frameShift))
            }
        }
        return result
    }

    func voiceActivityIndexToAudioIndex(_ index: Int) -> Int {
        return index * frameShift + frameShift
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
                while endIndex < vadResult.count && !vadResult[endIndex] {
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
