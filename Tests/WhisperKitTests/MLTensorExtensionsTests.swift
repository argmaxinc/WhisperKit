//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

#if canImport(CoreML.MLState)
import CoreML
@testable import WhisperKit
import XCTest

@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
final class MLTensorExtensionsTests: XCTestCase {
    func testAsIntArrayReturnsExpectedScalars() async {
        let tensor = MLTensor(MLShapedArray<Int32>(scalars: [1, -2, 42], shape: [3]))

        let result = await tensor.toIntArray()

        XCTAssertEqual(result, [1, -2, 42])
    }

    func testAsFloatArraySupportsFloat32Tensor() async {
        let tensor = MLTensor(MLShapedArray<Float32>(scalars: [0.25, -1.5, 2.0], shape: [3]))

        let result = await tensor.toFloatArray()

        assertEqual(result, [0.25, -1.5, 2.0], accuracy: 0.0001)
    }

    func testAsFloatArraySupportsFloatTypeTensor() async {
        let expected = [FloatType(0.125), FloatType(-0.75), FloatType(3.5)]
        let tensor = MLTensor(MLShapedArray<FloatType>(scalars: expected, shape: [3]))

        let result = await tensor.toFloatArray()

        assertEqual(result, expected.map(Float.init), accuracy: 0.0001)
    }

    func testAsFloatArraySupportsInt32Tensor() async {
        let tensor = MLTensor(MLShapedArray<Int32>(scalars: [-3, 0, 7], shape: [3]))

        let result = await tensor.toFloatArray()

        assertEqual(result, [-3, 0, 7], accuracy: 0.0001)
    }

    func testAsMLMultiArrayRoundTripsFloatTypeTensor() async {
        let expected = [FloatType(1.25), FloatType(-0.5), FloatType(3.75)]
        let tensor = MLTensor(MLShapedArray<FloatType>(scalars: expected, shape: [3]))

        let result = await tensor.toMLMultiArray()
        let shapedArray = MLShapedArray<FloatType>(result)

        XCTAssertEqual(result.shape, [3])
        XCTAssertEqual(shapedArray.scalars.count, expected.count)
        assertEqual(shapedArray.scalars.map(Float.init), expected.map(Float.init), accuracy: 0.0001)
    }

    func testAsMLMultiArrayRoundTripsInt32Tensor() async {
        let expected: [Int32] = [-9, 4, 12]
        let tensor = MLTensor(MLShapedArray<Int32>(scalars: expected, shape: [3]))

        let result = await tensor.toMLMultiArray()
        let shapedArray = MLShapedArray<Int32>(result)

        XCTAssertEqual(result.shape, [3])
        XCTAssertEqual(shapedArray.scalars, expected)
    }

    private func assertEqual(_ lhs: [Float], _ rhs: [Float], accuracy: Float) {
        XCTAssertEqual(lhs.count, rhs.count)
        for (actual, expected) in zip(lhs, rhs) {
            XCTAssertEqual(actual, expected, accuracy: accuracy)
        }
    }
}
#endif
