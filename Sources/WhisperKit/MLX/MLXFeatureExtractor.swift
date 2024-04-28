//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import MLX
import CoreML
import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class MLXFeatureExtractor: FeatureExtracting {
    public let melCount: Int?
    private let nFFT: Int
    private let hopLength: Int
    private let filters: MLXArray

    public init(
        melCount: Int = 80,
        nFFT: Int = 400,
        hopLength: Int = 160
    ) {
        self.melCount = melCount
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.filters = loadMelFilters(nMels: melCount)
    }

    public func logMelSpectrogram(fromAudio inputAudio: MLMultiArray) async throws -> MLMultiArray? {
        try Task.checkCancellation()
        let input = inputAudio.withUnsafeBytes { ptr in
            MLXArray(ptr, inputAudio.shape.map { $0.intValue }, type: Float.self)
        }
        let logMelSpectrogram = WhisperKitMLX.logMelSpectrogram(
            audio: input,
            filters: filters,
            nMels: melCount ?? 80,
            nFFT: nFFT,
            hopLength: hopLength
        )
        let logMelSpectrogramData = logMelSpectrogram.asArray(Float.self)
        let result = try MLMultiArray(shape: logMelSpectrogram.shape.map { NSNumber(value: $0) }, dataType: .float32)
        let maxRow = logMelSpectrogram.shape[0]
        let maxCol = logMelSpectrogram.shape[1]
        for row in 0..<maxRow {
            for col in 0..<maxCol {
                let key = [row, col] as [NSNumber]
                result[key] = NSNumber(value: logMelSpectrogramData[row * maxCol + col])
            }
        }
        return result
    }
}
