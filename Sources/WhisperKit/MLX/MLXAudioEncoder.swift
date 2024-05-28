//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import MLX
import WhisperKit
import MLXNN

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class MLXAudioEncoder: AudioEncoding {
    public var embedSize: Int? {
        encoder?.nState
    }
    private var encoder: AudioEncoder?

    public init() {}

    public func encodeFeatures(_ features: MLMultiArray) async throws -> MLMultiArray? {
        guard let encoder else {
            throw WhisperError.modelsUnavailable()
        }
        try Task.checkCancellation()
        let input = features.withUnsafeBytes { ptr in
            MLXArray(ptr, features.shape.map { $0.intValue }, type: FloatType.self)
        }
        let ouput = encoder(input)
        return try ouput.asMLMultiArray()
    }
}

extension MLXAudioEncoder: WhisperMLXModel {
    public func loadModel(at modelPath: URL) async throws {
        let parameters = try loadParameters(at: modelPath.appending(path: "weights.safetensors"), forKey: "encoder")
        let config = try loadConfig(at: modelPath.appending(path: "config.json"))
        let encoder = AudioEncoder(
            nMels: config.nMels,
            nCtx: config.nAudioCtx,
            nState: config.nAudioState,
            nHead: config.nAudioHead,
            nLayer: config.nAudioLayer,
            dType: .float16
        )
        let loadedEncoder = try encoder.update(parameters: parameters, verify: [.noUnusedKeys])
        MLX.eval(loadedEncoder)
        self.encoder = encoder
    }

    public func unloadModel() {
        encoder = nil
    }
}

final class AudioEncoder: Module {
    let nMels: Int
    let nCtx: Int
    let nState: Int
    let nHead: Int
    let nLayer: Int
    let dType: MLX.DType

    private let conv1: Conv1d
    private let conv2: Conv1d
    private let positionalEmbedding: MLXArray
    private let blocks: [ResidualAttentionBlock]
    private let ln_post: LayerNorm

    init(
        nMels: Int,
        nCtx: Int,
        nState: Int,
        nHead: Int,
        nLayer: Int,
        dType: MLX.DType = .float16
    ) {
        self.nMels = nMels
        self.nCtx = nCtx
        self.nState = nState
        self.nHead = nHead
        self.nLayer = nLayer
        self.dType = dType

        self.conv1 = Conv1d(inputChannels: nMels, outputChannels: nState, kernelSize: 3, padding: 1)
        self.conv2 = Conv1d(inputChannels: nState, outputChannels: nState, kernelSize: 3, stride: 2, padding: 1)
        self.positionalEmbedding = sinusoids(length: nCtx, channels: nState).asType(dType)
        self.blocks = (0..<nLayer).map { _ in ResidualAttentionBlock(nState: nState, nHead: nHead) }
        self.ln_post = LayerNorm(dimensions: nState)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = MLXNN.gelu(conv1(x))
        x = MLXNN.gelu(conv2(x))
        assert(Array(x.shape[1...]) == positionalEmbedding.shape, "incorrect audio shape")
        x = x + positionalEmbedding
        for block in blocks {
            x = block(x).x
        }
        x = ln_post(x)
        return x
    }
}
