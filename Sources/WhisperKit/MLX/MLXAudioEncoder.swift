//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import MLX
import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class MLXAudioEncoder: AudioEncoding {

    public var embedSize: Int? {
        fatalError("Not implemented")
    }

    public var sequenceLength: Int? {
        fatalError("Not implemented")
    }

    public init() {}

    public func encodeFeatures(_ features: MLMultiArray) async throws -> MLMultiArray? {
        // Make sure features is shape MultiArray (Float32 1 × {80,128} × 3000)
        try Task.checkCancellation()
        fatalError("Not implemented")
    }
}
