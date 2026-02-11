//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
import ArgmaxCore
import CoreML
import Foundation

// MARK: - Token Sampling

/// Protocol for TTS token sampling strategies.
public protocol TTSTokenSampling {
    /// Sample a single token from codec-0 logits.
    func sampleCodec0(
        logits: MLMultiArray,
        temperature: Float,
        topK: Int,
        generatedTokens: [Int32],
        repetitionPenalty: Float,
        suppressTokenIds: Set<Int>
    ) -> Int32

    /// Sample a single token from one head of multi-code logits.
    func sampleMultiHead(
        allLogits: MLMultiArray,
        headIndex: Int,
        temperature: Float,
        topK: Int
    ) -> Int32
}

// MARK: - Greedy / Top-k Sampler

/// Greedy / top-k / temperature token sampler with a seedable RNG.
///
/// Thread safety: each `TTSGenerateTask` owns its own `TTSGreedyTokenSampler`
/// instance (created per-task in `TTSKit.makeTask()` with a derived seed).
/// The `var rng` is never accessed concurrently because it's single-owner.
public class TTSGreedyTokenSampler: TTSTokenSampling, @unchecked Sendable {
    private var rng: any RandomNumberGenerator

    /// Create a sampler with an optional seed for reproducibility.
    /// - Parameter seed: If provided, uses a deterministic RNG. If nil, uses system RNG.
    public init(seed: UInt64? = nil) {
        if let seed {
            self.rng = SeededRandomNumberGenerator(seed: seed)
        } else {
            self.rng = SystemRandomNumberGenerator()
        }
    }

    public func sampleCodec0(
        logits: MLMultiArray,
        temperature: Float,
        topK: Int,
        generatedTokens: [Int32],
        repetitionPenalty: Float,
        suppressTokenIds: Set<Int>
    ) -> Int32 {
        return sampleCodec0WithVDSP(
            logits: logits,
            temperature: temperature,
            topK: topK,
            generatedTokens: generatedTokens,
            repetitionPenalty: repetitionPenalty,
            suppressTokenIds: suppressTokenIds
        )
    }

    public func sampleMultiHead(
        allLogits: MLMultiArray,
        headIndex: Int,
        temperature: Float,
        topK: Int
    ) -> Int32 {
        // Logits shape [1, 15, vocabSize] - extract one head using vDSP batch conversion
        let vocabSize = allLogits.shape[2].intValue
        let ptr = allLogits.dataPointer.bindMemory(to: FloatType.self, capacity: allLogits.count)
        let stride1 = allLogits.strides[1].intValue
        let stride2 = allLogits.strides[2].intValue
        var logitsF = [Float](repeating: 0, count: vocabSize)
        let base = headIndex * stride1

        if stride2 == 1 {
            // Contiguous last dim - use vectorized Float16 -> Float32 conversion
            let src = UnsafeBufferPointer(start: ptr.advanced(by: base), count: vocabSize)
            convertToFloat(src, to: &logitsF)
        } else {
            for i in 0..<vocabSize {
                logitsF[i] = Float(ptr[base + i * stride2])
            }
        }
        return sampleFromLogits(&logitsF, temperature: temperature, topK: topK)
    }

    // MARK: - Sampling implementations

    // TODO: use mltensor/bnns similar to whisperkit

    private func sampleCodec0WithVDSP(
        logits: MLMultiArray,
        temperature: Float,
        topK: Int,
        generatedTokens: [Int32],
        repetitionPenalty: Float,
        suppressTokenIds: Set<Int>
    ) -> Int32 {
        let vocabSize = logits.shape.last!.intValue
        var logitsF = extractFloat32Logits(logits, count: vocabSize)
        for id in suppressTokenIds where id < vocabSize {
            logitsF[id] = -.infinity
        }
        if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
            for tokenId in Set(generatedTokens) {
                let idx = Int(tokenId)
                guard idx < vocabSize else { continue }
                logitsF[idx] = logitsF[idx] > 0 ? logitsF[idx] / repetitionPenalty : logitsF[idx] * repetitionPenalty
            }
        }
        return sampleFromLogits(&logitsF, temperature: temperature, topK: topK)
    }

    // MARK: - Private helpers

    // TODO: remove, prefer no copy/conversion implementation
    private func extractFloat32Logits(_ arr: MLMultiArray, count: Int) -> [Float] {
        let ptr = arr.dataPointer.bindMemory(to: FloatType.self, capacity: arr.count)
        let lastStride = arr.strides.last!.intValue
        var result = [Float](repeating: 0, count: count)
        if lastStride == 1 {
            // Contiguous last dim
            let src = UnsafeBufferPointer(start: ptr, count: count)
            convertToFloat(src, to: &result)
        } else {
            for i in 0..<count {
                result[i] = Float(ptr[i * lastStride])
            }
        }
        return result
    }

    private func sampleFromLogits(_ logits: inout [Float], temperature: Float, topK: Int) -> Int32 {
        let vocabSize = logits.count

        // Greedy: use vDSP for vectorised argmax
        if temperature == 0 {
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(logits, 1, &maxVal, &maxIdx, vDSP_Length(vocabSize))
            return Int32(maxIdx)
        }

        // Temperature scaling
        var temp = temperature
        vDSP_vsdiv(logits, 1, &temp, &logits, 1, vDSP_Length(vocabSize))

        // Top-k filtering: partial sort is faster than sorting the full vocab
        if topK > 0 && topK < vocabSize {
            var sorted = logits
            vDSP_vsort(&sorted, vDSP_Length(vocabSize), -1) // descending
            let threshold = sorted[topK - 1]
            for i in 0..<vocabSize where logits[i] < threshold {
                logits[i] = -.infinity
            }
        }

        // Softmax (numerically stable: subtract max before exp)
        var maxVal: Float = 0
        vDSP_maxv(logits, 1, &maxVal, vDSP_Length(vocabSize))
        var negMax = -maxVal
        vDSP_vsadd(logits, 1, &negMax, &logits, 1, vDSP_Length(vocabSize))
        var count = Int32(vocabSize)
        vvexpf(&logits, logits, &count)
        var sum: Float = 0
        vDSP_sve(logits, 1, &sum, vDSP_Length(vocabSize))
        if sum > 0 {
            vDSP_vsdiv(logits, 1, &sum, &logits, 1, vDSP_Length(vocabSize))
        }

        // Categorical sample using the (possibly seeded) RNG
        let r = Float.random(in: 0..<1, using: &rng)
        var cumSum: Float = 0
        for i in 0..<vocabSize {
            cumSum += logits[i]
            if cumSum >= r { return Int32(i) }
        }
        return Int32(vocabSize - 1)
    }
}

// MARK: - Seedable RNG

/// A seedable random number generator using xoshiro256 algorithm.
/// Produces deterministic sequences for a given seed.
public struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: (UInt64, UInt64, UInt64, UInt64)

    public init(seed: UInt64) {
        // Initialize state from a single seed
        var z = seed &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        let s0 = z ^ (z >> 31)

        z = (seed &+ 2 &* 0x9E37_79B9_7F4A_7C15)
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        let s1 = z ^ (z >> 31)

        z = (seed &+ 3 &* 0x9E37_79B9_7F4A_7C15)
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        let s2 = z ^ (z >> 31)

        z = (seed &+ 4 &* 0x9E37_79B9_7F4A_7C15)
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        let s3 = z ^ (z >> 31)

        state = (s0, s1, s2, s3)
    }

    public mutating func next() -> UInt64 {
        let result = rotl(state.1 &* 5, 7) &* 9
        let t = state.1 << 17
        state.2 ^= state.0
        state.3 ^= state.1
        state.1 ^= state.2
        state.0 ^= state.3
        state.2 ^= t
        state.3 = rotl(state.3, 45)
        return result
    }

    private func rotl(_ x: UInt64, _ k: Int) -> UInt64 {
        (x << k) | (x >> (64 - k))
    }
}
