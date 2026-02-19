//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
@testable import WhisperKit

final class WorkerPoolTestFeatureExtractorMock: FeatureExtracting {
    var melCount: Int? = 80
    var windowSamples: Int? = 32_000

    func logMelSpectrogram(fromAudio inputAudio: any AudioProcessorOutputType) async throws -> (any FeatureExtractorOutputType)? {
        try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
    }
}
