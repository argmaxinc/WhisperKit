//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
@testable import WhisperKit

final class WorkerPoolTestDecodingInputs: DecodingInputsType {
    var initialPrompt: [Int]
    var inputIds: MLMultiArray
    var cacheLength: MLMultiArray

    init(initialPrompt: [Int]) throws {
        self.initialPrompt = initialPrompt
        self.inputIds = try MLMultiArray(shape: [1], dataType: .int32)
        self.cacheLength = try MLMultiArray(shape: [1], dataType: .int32)
        self.inputIds[0] = NSNumber(value: initialPrompt.last ?? 0)
        self.cacheLength[0] = 0
    }

    func reset(prefilledCacheSize: Int, maxTokenContext: Int) {
        cacheLength[0] = NSNumber(value: prefilledCacheSize)
    }
}
