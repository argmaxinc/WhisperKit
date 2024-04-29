//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import XCTest
import MLX
@testable import WhisperKit
@testable import WhisperKitMLX

final class MLXUnitTests: XCTestCase {

    // MARK: - Feature Extractor Tests

    func testLogmelOutput() async throws {
        let audioSamples = [Float](repeating: 0.0, count: 16000)
        let paddedSamples = try XCTUnwrap(
            AudioProcessor.padOrTrimAudio(fromArray: audioSamples, startAt: 0, toLength: 480_000),
            "Failed to pad audio samples"
        )
        let featureExtractor = MLXFeatureExtractor()
        let extractedFeature = try await featureExtractor.logMelSpectrogram(fromAudio: paddedSamples)
        let melSpectrogram = try XCTUnwrap(extractedFeature, "Failed to produce Mel spectrogram from audio samples")

        let expectedShape: [NSNumber] = [3000, 80]
        XCTAssertNotNil(melSpectrogram, "Failed to produce Mel spectrogram from audio samples")
        XCTAssertEqual(melSpectrogram.shape, expectedShape, "Mel spectrogram shape is not as expected")
    }

    // MARK: - Utils Tests

    func testAsMLMultiArray() throws {
        let count = 24
        let input = (0..<count).map { Float($0) }
        let arr1 = MLXArray(input, [count])
        let multiArray1 = try arr1.asMLMultiArray()

        XCTAssertEqual(arr1.shape, multiArray1.shape.map { $0.intValue })
        for col in 0..<count {
            let v1 = multiArray1[[col] as [NSNumber]].floatValue
            let v2 = arr1[col]
            XCTAssertEqual(v1, v2.item(Float.self), accuracy: 0.00001)
        }

        let arr2 = MLXArray(input, [4, 6])
        let multiArray2 = try arr2.asMLMultiArray()

        XCTAssertEqual(arr2.shape, multiArray2.shape.map { $0.intValue })
        for row in 0..<4 {
            for col in 0..<6 {
                let v1 = multiArray2[[row, col] as [NSNumber]].floatValue
                let v2 = arr2[row, col]
                XCTAssertEqual(v1, v2.item(Float.self), accuracy: 0.00001)
            }
        }

        let arr3 = MLXArray(input, [4, 3, 2])
        let multiArray3 = try arr3.asMLMultiArray()

        XCTAssertEqual(arr3.shape, multiArray3.shape.map { $0.intValue })
        for dim1 in 0..<4 {
            for dim2 in 0..<3 {
                for dim3 in 0..<2 {
                    let v1 = multiArray3[[dim1, dim2, dim3] as [NSNumber]].floatValue
                    let v2 = arr3[dim1, dim2, dim3]
                    XCTAssertEqual(v1, v2.item(Float.self), accuracy: 0.00001)
                }
            }
        }
    }
}
