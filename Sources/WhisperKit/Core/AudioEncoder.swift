//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML

/// AudioEncoding protocol defines the requirements for an audio encoding implementation.
public protocol AudioEncoding {
    /// The size of the embedding produced by the encoder.
    var embedSize: Int? { get }

    /// Encodes the given audio features asynchronously.
    /// - Parameter features: The audio features to be encoded.
    /// - Returns: An optional `MLMultiArray` containing the encoded features.
    func encodeFeatures(_ features: MLMultiArray) async throws -> MLMultiArray?
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class AudioEncoder: AudioEncoding, WhisperMLModel {
    public var model: MLModel?

    public var embedSize: Int? {
        guard let inputDescription = model?.modelDescription.outputDescriptionsByName["encoder_output_embeds"] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[1]
    }

    public var sequenceLength: Int? {
        guard let inputDescription = model?.modelDescription.outputDescriptionsByName["encoder_output_embeds"] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[3]
    }

    public init() {}

    public func encodeFeatures(_ features: MLMultiArray) async throws -> MLMultiArray? {
        // Make sure features is shape MultiArray (Float32 1 × {80,128} × 3000)
        guard let model else {
            throw WhisperError.modelsUnavailable()
        }
        try Task.checkCancellation()

        let interval = Logging.beginSignpost("EncodeAudio", signposter: Logging.AudioEncoding.signposter)
        defer { Logging.endSignpost("EncodeAudio", interval: interval, signposter: Logging.AudioEncoding.signposter) }

        let modelInputs = AudioEncoderInput(melspectrogram_features: features)
        let outputFeatures = try await model.asyncPrediction(from: modelInputs, options: MLPredictionOptions())
        let output = AudioEncoderOutput(features: outputFeatures)
        return output.encoder_output_embeds
    }
}
