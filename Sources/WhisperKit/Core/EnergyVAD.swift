//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import Accelerate

/// Voice activity detection based on energy threshold
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
        return (0..<count).map { vDSP.sumOfSquares(waveform[$0 * frameShift..<($0 * frameShift + frameLength)]) > energyThreshold }
    }

    func voiceActivityIndexToAudioIndex(_ index: Int) -> Int {
        return index * frameShift + frameShift
    }

    func findLongestSilence(in array: [Bool]) -> (startIndex: Int, endIndex: Int)? {
        var longestStartIndex: Int?
        var longestEndIndex: Int?
        var longestCount = 0
        var index = 0
        while index < array.count {
            let value = array[index]
            if value {
                // found non-silence, skip
                index += 1
            } else {
                // found beginning of silence, find the end
                var endIndex = index
                while endIndex < array.count && !array[endIndex] {
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
