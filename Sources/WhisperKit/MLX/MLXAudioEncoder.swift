//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import MLX
import MLXNN
import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class MLXAudioEncoder: AudioEncoding, WhisperMLXModel {
    public var model: AudioEncoderModule?

    public var embedSize: Int? {
        model?.nState
    }

    public init() {}

    public func encodeFeatures(_ features: MLMultiArray) async throws -> MLMultiArray? {
        guard let model else {
            throw WhisperError.modelsUnavailable()
        }
        try Task.checkCancellation()
        let inputArray = features.asMLXArray(FloatType.self)
        let input = inputArray.asMLXInput()
        let output = model(input)
        return try output.asMLXOutput().asMLMultiArray()
    }
}

extension MLXAudioEncoder {
    public func loadModel(at modelPath: URL, configPath: URL?) async throws {
        let parameters = try loadParameters(at: modelPath)
        let config = try loadConfig(at: configPath)
        let encoder = AudioEncoderModule(
            nMels: config.nMels,
            nCtx: config.nAudioCtx,
            nState: config.nAudioState,
            nHead: config.nAudioHead,
            nLayer: config.nAudioLayer,
            dType: .float16
        )
        let loadedEncoder = try encoder.update(parameters: parameters, verify: [.noUnusedKeys])
        MLX.eval(loadedEncoder)
        self.model = encoder
    }

    public func unloadModel() {
        model = nil
    }

    public var modelState: ModelState {
        return model == nil ? .unloaded : .loaded
    }
}

public class AudioEncoderModule: MLXNN.Module {
    let nMels: Int
    let nCtx: Int
    let nState: Int
    let nHead: Int
    let nLayer: Int
    let dType: MLX.DType

    @ModuleInfo(key: "conv1") private var conv1: Conv1d
    @ModuleInfo(key: "conv2") private var conv2: Conv1d
    @ModuleInfo(key: "blocks") private var blocks: [ResidualAttentionBlock]
    @ModuleInfo(key: "ln_post") private var lnPost: LayerNorm
    private let _positionalEmbedding: MLXArray

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

        self._conv1.wrappedValue = Conv1d(inputChannels: nMels, outputChannels: nState, kernelSize: 3, padding: 1)
        self._conv2.wrappedValue = Conv1d(inputChannels: nState, outputChannels: nState, kernelSize: 3, stride: 2, padding: 1)
        self._blocks.wrappedValue = (0..<nLayer).map { _ in ResidualAttentionBlock(nState: nState, nHead: nHead) }
        self._lnPost.wrappedValue = LayerNorm(dimensions: nState)
        self._positionalEmbedding = sinusoids(length: nCtx, channels: nState).asType(dType)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = MLXNN.gelu(conv1(x))
        x = MLXNN.gelu(conv2(x))
        assert(Array(x.shape[1...]) == _positionalEmbedding.shape, "incorrect audio shape")
        x = x + _positionalEmbedding
        for block in blocks {
            x = block(x).x
        }
        x = lnPost(x)
        return x
    }
}
