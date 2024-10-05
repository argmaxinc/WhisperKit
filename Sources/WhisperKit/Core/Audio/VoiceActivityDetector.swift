//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// A base class for Voice Activity Detection (VAD), used to identify and separate segments of audio that contain human speech from those that do not.
/// Subclasses must implement the `voiceActivity(in:)` method to provide specific voice activity detection functionality.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public struct VoiceActivityDetector: VoiceActivityDetectable {
    public let sampleRate: Int
    public let frameLengthSamples: Int
    public let frameOverlapSamples: Int
    
    public init(
        sampleRate: Int = 16000,
        frameLengthSamples: Int,
        frameOverlapSamples: Int = 0
    ) {
        self.sampleRate = sampleRate
        self.frameLengthSamples = frameLengthSamples
        self.frameOverlapSamples = frameOverlapSamples
    }
    
    public func voiceActivity(in waveform: [Float]) -> [Bool] {
        fatalError("voiceActivity(in:) must be implemented by conforming types")
    }
}
