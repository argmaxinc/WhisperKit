//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import MLX
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
