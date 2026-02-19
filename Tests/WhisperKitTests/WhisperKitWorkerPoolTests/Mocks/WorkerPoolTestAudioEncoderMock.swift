//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
@testable import WhisperKit

final class WorkerPoolTestAudioEncoderMock: AudioEncoding {
    var embedSize: Int? = 512

    func encodeFeatures(_ features: any FeatureExtractorOutputType) async throws -> (any AudioEncoderOutputType)? {
        try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
    }
}
