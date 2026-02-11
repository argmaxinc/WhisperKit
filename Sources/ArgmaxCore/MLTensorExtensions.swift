//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML

// MARK: - MLTensor Conversions

#if canImport(CoreML.MLState)
@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
public extension MLTensor {
    /// Converts the tensor to an [Int] array synchronously.
    func asIntArray() -> [Int] {
        let semaphore = DispatchSemaphore(value: 0)
        var result: [Int] = []

        Task(priority: .high) {
            result = await self.shapedArray(of: Int32.self).scalars.map { Int($0) }
            semaphore.signal()
        }

        semaphore.wait()
        return result
    }

    /// Converts the tensor to a [Float] array synchronously.
    func asFloatArray() -> [Float] {
        let semaphore = DispatchSemaphore(value: 0)
        let tensorType = self.scalarType

        var result: [Float] = []

        Task(priority: .high) {
            switch tensorType {
                case is Float32.Type:
                    result = await self.shapedArray(of: Float32.self).scalars.map { Float($0) }
                case is FloatType.Type:
                    result = await self.shapedArray(of: FloatType.self).scalars.map { Float($0) }
                case is Float.Type:
                    result = await self.shapedArray(of: Float.self).scalars.map { Float($0) }
                case is Int32.Type:
                    result = await self.shapedArray(of: Int32.self).scalars.map { Float($0) }
                default:
                    fatalError("Unsupported data type")
            }
            semaphore.signal()
        }

        semaphore.wait()
        return result
    }

    /// Converts the tensor to an MLMultiArray synchronously.
    func asMLMultiArray() -> MLMultiArray {
        let semaphore = DispatchSemaphore(value: 0)
        let tensorType = self.scalarType

        var result = try! MLMultiArray(shape: [1], dataType: .float16, initialValue: 0.0)

        Task(priority: .high) {
            switch tensorType {
                case is Float32.Type:
                    result = MLMultiArray(await self.shapedArray(of: Float32.self))
                case is FloatType.Type:
                    result = MLMultiArray(await self.shapedArray(of: FloatType.self))
                case is Float.Type:
                    result = MLMultiArray(await self.shapedArray(of: Float.self))
                case is Int32.Type:
                    result = MLMultiArray(await self.shapedArray(of: Int32.self))
                default:
                    fatalError("Unsupported data type")
            }
            semaphore.signal()
        }

        semaphore.wait()
        return result
    }
}
#endif
