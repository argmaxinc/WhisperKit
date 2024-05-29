//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import XCTest
import MLX
import WhisperKitTestsUtils
import CoreML
@testable import WhisperKit
@testable import WhisperKitMLX

final class MLXUnitTests: XCTestCase {

    private let accuracy: Float = 0.00001

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

        let expectedShape: [NSNumber] = [1, 80, 1, 3000]
        XCTAssertNotNil(melSpectrogram, "Failed to produce Mel spectrogram from audio samples")
        XCTAssertEqual(melSpectrogram.shape, expectedShape, "Mel spectrogram shape is not as expected")
    }

    // MARK: - Encoder Tests

    func testEncoderOutput() async throws {
        let audioEncoder = MLXAudioEncoder()
        let modelPath = try URL(filePath: tinyMLXModelPath())
        try await audioEncoder.loadModel(at: modelPath)

        let encoderInput = try MLMultiArray(shape: [1, 80, 1, 3000], dataType: .float16)
        let expectedShape: [NSNumber] = [1, 384, 1, 1500]

        let encoderOutput = try await audioEncoder.encodeFeatures(encoderInput)
        XCTAssertNotNil(encoderOutput, "Failed to encode features")
        XCTAssertEqual(encoderOutput?.shape, expectedShape, "Encoder output shape is not as expected")
    }

    // MARK: - Utils Tests

    func testArrayConversion() throws {
        let count = 16
        let arr1 = MLXArray(0..<count, [2, count / 2]).asType(Int32.self)
        let arr2 = try arr1.asMLMultiArray().asMLXArray(Int32.self)
        XCTAssertTrue(MLX.allClose(arr1, arr2).item(), "Array conversion failed")

        let arr3 = arr1.asMLXOutput().asMLXInput()
        XCTAssertTrue(MLX.allClose(arr1, arr3).item(), "Input output conversion failed")

        let arr4 = try arr1.asMLXOutput().asMLMultiArray().asMLXArray(Int32.self).asMLXInput()
        XCTAssertTrue(MLX.allClose(arr1, arr4).item(), "Complex conversion failed")
    }

    func testAsMLMultiArray() throws {
        let count = 24
        let input = (0..<count).map { Float($0) }
        let arr1 = MLXArray(input, [count])
        let multiArray1 = try arr1.asMLMultiArray()

        XCTAssertEqual(arr1.shape, multiArray1.shape.map { $0.intValue })
        for col in 0..<count {
            let v1 = multiArray1[[col] as [NSNumber]].floatValue
            let v2 = arr1[col]
            XCTAssertEqual(v1, v2.item(Float.self), accuracy: accuracy)
        }

        let arr2 = MLXArray(input, [4, 6])
        let multiArray2 = try arr2.asMLMultiArray()

        XCTAssertEqual(arr2.shape, multiArray2.shape.map { $0.intValue })
        for row in 0..<4 {
            for col in 0..<6 {
                let v1 = multiArray2[[row, col] as [NSNumber]].floatValue
                let v2 = arr2[row, col]
                XCTAssertEqual(v1, v2.item(Float.self), accuracy: accuracy)
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
                    XCTAssertEqual(v1, v2.item(Float.self), accuracy: accuracy)
                }
            }
        }
    }

    func testSinusoids() {
        let result1 = sinusoids(length: 0, channels: 0).asType(Float.self)
        XCTAssertEqual(result1.shape, [0, 0])
        XCTAssertEqual(result1.count, 0)

        let result2 = sinusoids(length: 2, channels: 4).asType(Float.self)
        XCTAssertEqual(result2.shape, [2, 4])
        XCTAssertEqual(result2.count, 2)
        XCTAssertEqual(result2[0].asArray(Float.self), [0.0, 0.0, 1.0, 1.0], accuracy: accuracy)
        XCTAssertEqual(result2[1].asArray(Float.self), [0.841471, 0.0001, 0.540302, 1.0], accuracy: accuracy)

        let result3 = sinusoids(length: 4, channels: 8).asType(Float.self)
        XCTAssertEqual(result3.shape, [4, 8])
        XCTAssertEqual(result3.count, 4)

        XCTAssertEqual(result3[0].asArray(Float.self), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], accuracy: accuracy)
        XCTAssertEqual(result3[1].asArray(Float.self), [0.841471, 0.0463992, 0.00215443, 0.0001, 0.540302, 0.998923, 0.999998, 1], accuracy: accuracy)
        XCTAssertEqual(result3[2].asArray(Float.self), [0.909297, 0.0926985, 0.00430886, 0.0002, -0.416147, 0.995694, 0.999991, 1.0], accuracy: accuracy)
        XCTAssertEqual(result3[3].asArray(Float.self), [0.14112, 0.138798, 0.00646326, 0.0003, -0.989992, 0.990321, 0.999979, 1.0], accuracy: accuracy)
    }
}
