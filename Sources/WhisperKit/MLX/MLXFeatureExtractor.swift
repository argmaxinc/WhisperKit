//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
import MLX
import MLXFFT
import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class MLXFeatureExtractor: FeatureExtracting, WhisperMLXModel {
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
        self.filters = MLXFeatureExtractor.loadMelFilters(nMels: melCount)
    }

    public func logMelSpectrogram(fromAudio inputAudio: MLMultiArray) async throws -> MLMultiArray? {
        try Task.checkCancellation()
        let input = inputAudio.asMLXArray(Float.self)
        let output = MLXFeatureExtractor.logMelSpectrogram(
            audio: input,
            filters: filters,
            nMels: melCount ?? 80,
            nFFT: nFFT,
            hopLength: hopLength
        )
        return try output.asType(FloatType.self).asMLXOutput().asMLMultiArray()
    }

    // Stubs for WhisperMLXModel protocol, not needed
    public typealias MLXModuleType = NSObject
    public var model: NSObject?

    public func loadModel(at modelPath: URL, configPath: URL?) async throws {}
    public func unloadModel() {}
}

public extension MLXFeatureExtractor {
    /// Return the Hanning window.
    /// Taken from [numpy](https://numpy.org/doc/stable/reference/generated/numpy.hanning.html) implementation
    static func hanningNumpy(_ size: Int) -> MLXArray {
        if size < 1 {
            return MLXArray([Float]())
        }
        if size == 1 {
            return MLXArray([1.0] as [Float])
        }
        let n = MLXArray(Array(stride(from: 1 - size, to: size, by: 2)))
        return 0.5 + 0.5 * MLX.cos(MLXArray(.pi) * n / Float(size - 1))
    }

    static func hanning(_ size: Int) -> MLXArray {
        hanningNumpy(size + 1)[..<(-1)]
    }

    static func pad(
        _ x: MLXArray,
        padding: Int,
        padMode: PadMode = .constant
    ) -> MLXArray {
        switch padMode {
            case .constant:
                return MLX.padded(x, widths: [IntOrPair((padding, padding))])
            case .reflect:
                let prefix = x[1..<padding + 1][.stride(by: -1)]
                let suffix = x[-(padding + 1) ..< -1][.stride(by: -1)]
                return MLX.concatenated([prefix, x, suffix])
        }
    }

    static func stft(
        _ x: MLXArray,
        window: MLXArray,
        nPerSeg: Int = 256,
        nOverlap: Int? = nil,
        nFFT: Int? = nil,
        axis: Int = -1,
        padMode: PadMode = .reflect
    ) -> MLXArray {
        let nFFT = nFFT ?? nPerSeg
        let nOverlap = nOverlap ?? nFFT / 4

        let padding = nPerSeg / 2
        let x = pad(x, padding: padding, padMode: padMode)

        let strides = [nOverlap, 1]
        let t = (x.count - nPerSeg + nOverlap) / nOverlap
        let shape = [t, nFFT]
        return MLXFFT.rfft(MLX.asStrided(x, shape, strides: strides) * window)
    }

    /// Compute the log mel spectrogram of audio
    /// Taken from [MLX](https://github.com/ml-explore/mlx-examples/blob/c012eb173f0f632e369ec71f08be777df3aede08/whisper/whisper/audio.py#L130) implementation
    static func logMelSpectrogram(
        audio: MLXArray,
        filters: MLXArray,
        nMels: Int = 80,
        padding: Int = 0,
        nFFT: Int = 400,
        hopLength: Int = 160
    ) -> MLXArray {
        let window = hanning(nFFT)
        let freqs = stft(audio, window: window, nPerSeg: nFFT, nOverlap: hopLength)
        let magnitudes = freqs[..<(-1)].abs().square()

        let melSpec = magnitudes.matmul(filters.T)

        var logSpec = MLX.maximum(melSpec, 1e-10).log10()
        logSpec = MLX.maximum(logSpec, logSpec.max() - 8.0)
        logSpec = (logSpec + 4.0) / 4.0
        return logSpec
    }

    /// Load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    /// Allows decoupling librosa dependency.
    ///
    /// Saved using:
    /// ```python
    /// n80 = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
    /// n128 = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)
    /// with open('mel_filters_80.npy', 'wb') as f:
    ///   np.save(f, n80)
    /// with open('mel_filters_128.npy', 'wb') as f:
    ///   np.save(f, n128)
    /// ```
    static func loadMelFilters(nMels: Int) -> MLXArray {
        precondition(nMels == 80 || nMels == 128, "Unsupported nMels: \(nMels)")
        let fileUrl = Bundle.module.url(forResource: "mel_filters_\(nMels)", withExtension: "npy")!
        return try! MLX.loadArray(url: fileUrl)
    }
}
