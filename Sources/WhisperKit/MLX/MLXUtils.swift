//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import MLX
import MLXNN
import CoreML

// MARK: - Extensions

extension MLXArray {
    func asMLMultiArray() throws -> MLMultiArray {
        let dataType = multiArrayDataType()
        // a buffer to be passed to CoreML
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: nbytes, alignment: 8)
        // copy the data from the MLXArray backing into buffer
        asData(noCopy: true).withUnsafeBytes { ptr in
            let destination = UnsafeMutableRawBufferPointer(start: buffer, count: nbytes)
            ptr.copyBytes(to: destination)
        }
        return try MLMultiArray(
            dataPointer: buffer,
            shape: shape.map { NSNumber(value: $0) },
            dataType: dataType,
            strides: strides.map { NSNumber(value: $0) },
            deallocator: { $0.deallocate() }
        )
    }
}

extension MLXArray {
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

// MARK: - Functions

func sinusoids(length: Int, channels: Int, maxTimescale: Int = 10000) -> MLXArray {
    assert(channels % 2 == 0)
    let logTimescaleIncrement = log(Float(maxTimescale)) / Float(channels / 2 - 1)
    let invTimescales = MLX.exp(-logTimescaleIncrement * MLXArray(Array(0..<(channels / 2))))
    let scaledTime = MLXArray(Array(0..<length)).reshaped([length, 1]) * invTimescales.reshaped([1, channels / 2])
    return MLX.concatenated([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: 1)
}

func loadParameters(at url: URL, forKey key: String? = nil) throws -> NestedDictionary<String, MLXArray> {
    let arrays = try MLX.loadArrays(url: url)
    let params = ModuleParameters.unflattened(arrays)
    guard let key else {
        return params
    }
    guard let keyParams = params[key] else {
        throw CocoaError.error(.coderValueNotFound)
    }
    return NestedDictionary(item: keyParams)
}

func loadConfig(at url: URL) throws -> ModelConfig {
    let configDecoder = JSONDecoder()
    configDecoder.keyDecodingStrategy = .convertFromSnakeCase
    return try configDecoder.decode(ModelConfig.self, from: Data(contentsOf: url))
}
