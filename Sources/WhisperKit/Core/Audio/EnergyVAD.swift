//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// Voice activity detection based on energy threshold
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public struct EnergyVAD: VoiceActivityDetectable {
    public let sampleRate: Int
    public let frameLengthSamples: Int
    public let frameOverlapSamples: Int
    public var energyThreshold: Float

    public init(
        sampleRate: Int = WhisperKit.sampleRate,
        frameLength: Float = 0.1,
        frameOverlap: Float = 0.0,
        energyThreshold: Float = 0.02
    ) {
        self.sampleRate = sampleRate
        self.frameLengthSamples = Int(frameLength * Float(sampleRate))
        self.frameOverlapSamples = Int(frameOverlap * Float(sampleRate))
        self.energyThreshold = energyThreshold
    }
    
    public func voiceActivity(in waveform: [Float]) -> [Bool] {
        let chunkRatio = Double(waveform.count) / Double(frameLengthSamples)
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
}
