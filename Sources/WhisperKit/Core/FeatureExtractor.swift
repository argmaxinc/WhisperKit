//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import AVFoundation
import CoreGraphics
import CoreML
import Foundation

public protocol FeatureExtractorOutputType {}
extension MLMultiArray: FeatureExtractorOutputType {}

public protocol FeatureExtracting {
    associatedtype OutputType: FeatureExtractorOutputType

    var melCount: Int? { get }
    var windowSamples: Int? { get }
    func logMelSpectrogram(fromAudio inputAudio: MLMultiArray) async throws -> OutputType?
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class FeatureExtractor: FeatureExtracting, WhisperMLModel {
    public var model: MLModel?

    public init() {}

    public var melCount: Int? {
        guard let inputDescription = model?.modelDescription.outputDescriptionsByName["melspectrogram_features"] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[1]
    }

    public var windowSamples: Int? {
        guard let inputDescription = model?.modelDescription.inputDescriptionsByName["audio"] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[0]  // The audio input is a 1D array
    }

    public func logMelSpectrogram(fromAudio inputAudio: MLMultiArray) async throws -> MLMultiArray? {
        guard let model else {
            throw WhisperError.modelsUnavailable()
        }
        try Task.checkCancellation()

        let interval = Logging.beginSignpost("ExtractAudioFeatures", signposter: Logging.FeatureExtractor.signposter)
        defer { Logging.endSignpost("ExtractAudioFeatures", interval: interval, signposter: Logging.FeatureExtractor.signposter) }

        let modelInputs = MelSpectrogramInput(audio: inputAudio)
        let outputFeatures = try await model.asyncPrediction(from: modelInputs, options: MLPredictionOptions())
        let output = MelSpectrogramOutput(features: outputFeatures)
        return output.melspectrogramFeatures
    }

}
