//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import MLX
import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class MLXAudioEncoder: AudioEncoding {

    public var embedSize: Int? {
        return 1234
    }

    public var sequenceLength: Int? {
        return 1234
    }

    public init() {}

    public func encodeFeatures(_ features: MLMultiArray) async throws -> MLMultiArray? {
        // Make sure features is shape MultiArray (Float32 1 × {80,128} × 3000)
//        guard let model else {
//            throw WhisperError.modelsUnavailable()
//        }
        try Task.checkCancellation()

//        let interval = Logging.beginSignpost("EncodeAudio", signposter: Logging.AudioEncoding.signposter)
//        defer { Logging.endSignpost("EncodeAudio", interval: interval, signposter: Logging.AudioEncoding.signposter) }

        let modelInputs = MLXArray()
        let outputFeatures =  MLXArray()
        let output =  MLMultiArray()
        return output
    }
}
