//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
import MLX
import MLXNN
import WhisperKit

// MARK: - Extensions

extension MLMultiArray {
    func asMLXArray<T: MLShapedArrayScalar & HasDType>(_ type: T.Type) -> MLXArray {
        let shape = shape.map(\.intValue)
        let strides = strides.map(\.intValue)
        return withUnsafeBufferPointer(ofType: T.self) { ptr in
            let buffer = UnsafeBufferPointer(start: ptr.baseAddress, count: shape.reduce(1, *))
            return asStrided(MLXArray(buffer, shape), shape, strides: strides)
        }
    }
}

extension MLXArray {
    /// Adapts the shape of the output array so MLX is compatible with CoreML
    ///
    /// Remove empty dimensions, swap axes and add empty dimensions, result: [1, n, 1, m]
    func asMLXOutput() -> MLXArray {
        squeezed().swappedAxes(0, 1).expandedDimensions(axes: [0, 2])
    }

    /// Adapts the shape of the input array so MLX is compatible with CoreML
    ///
    /// Remove empty dimensions, swap axes, result: [1, n, m]
    func asMLXInput() -> MLXArray {
        squeezed().swappedAxes(0, 1).expandedDimensions(axes: [0])
    }
}


extension MLXArray {
    func asMLMultiArray() throws -> MLMultiArray {
        let dataType = multiArrayDataType()
        // a buffer to be passed to CoreML
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: nbytes, alignment: 8)
        // copy the data from the MLXArray backing into buffer
        let dataStartTime = CFAbsoluteTimeGetCurrent()
        asData(access: .noCopy).data.withUnsafeBytes { ptr in
            let destination = UnsafeMutableRawBufferPointer(start: buffer, count: nbytes)
            ptr.copyBytes(to: destination)
        }
        let time = Date()
        let outputArray = try MLMultiArray(
            dataPointer: buffer,
            shape: shape.map { NSNumber(value: $0) },
            dataType: dataType,
            strides: strides.map { NSNumber(value: $0) },
            deallocator: { $0.deallocate() }
        )

        return outputArray
    }
}

public extension MLXArray {
    func multiArrayDataType() -> MLMultiArrayDataType {
        switch dtype {
            case .bool, .bfloat16, .complex64,
                 .uint8, .uint16, .uint32, .uint64,
                 .int8, .int16, .int64:
                fatalError("Unsupported type: \(dtype)")
            case .int32:
                return .int32
            case .float16:
                return .float16
            case .float32:
                return .float32
        }
    }
}

extension Embedding {
    func asLinear(_ x: MLXArray) -> MLXArray {
        x.matmul(weight.T)
    }
}

// MARK: - Functions

func additiveCausalMask(_ n: Int, dType: MLX.DType = .float32) -> MLXArray {
    let indices = MLXArray(Array(0..<n))
    let mask = indices[0..., .newAxis] .< indices[.newAxis]
    return mask.asType(dType) * -1e9
}

func sinusoids(length: Int, channels: Int, maxTimescale: Int = 10000) -> MLXArray {
    assert(channels % 2 == 0)
    let logTimescaleIncrement = log(Float(maxTimescale)) / Float(channels / 2 - 1)
    let invTimescales = MLX.exp(-logTimescaleIncrement * MLXArray(Array(0..<(channels / 2))))
    let scaledTime = MLXArray(Array(0..<length)).reshaped([length, 1]) * invTimescales.reshaped([1, channels / 2])
    return MLX.concatenated([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: 1)
}

func loadParameters(at url: URL) throws -> NestedDictionary<String, MLXArray> {
    let arrays = try MLX.loadArrays(url: url)
    return ModuleParameters.unflattened(arrays)
}

func loadConfig(at configPath: URL?) throws -> MLXModelConfig {
    guard let url = configPath else {
        throw WhisperError.modelsUnavailable("Config path must be specified for MLX models")
    }
    let configDecoder = JSONDecoder()
    configDecoder.keyDecodingStrategy = .convertFromSnakeCase
    return try configDecoder.decode(MLXModelConfig.self, from: Data(contentsOf: url))
}
