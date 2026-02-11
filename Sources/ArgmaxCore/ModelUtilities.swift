//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

// MARK: - Model Dimension Introspection

/// Read a dimension from a model input's multiarray constraint shape.
/// - Parameters:
///   - model: The loaded MLModel to inspect
///   - named: The input feature name
///   - position: The index in the shape array
/// - Returns: The dimension value, or nil if unavailable
public func modelInputDim(_ model: MLModel?, named: String, position: Int) -> Int? {
    guard let desc = model?.modelDescription.inputDescriptionsByName[named],
          desc.type == .multiArray,
          let constraint = desc.multiArrayConstraint else { return nil }
    let shape = constraint.shape.map { $0.intValue }
    guard position < shape.count else { return nil }
    return shape[position]
}

/// Read a dimension from a model output's multiarray constraint shape.
/// - Parameters:
///   - model: The loaded MLModel to inspect
///   - named: The output feature name
///   - position: The index in the shape array
/// - Returns: The dimension value, or nil if unavailable
public func modelOutputDim(_ model: MLModel?, named: String, position: Int) -> Int? {
    guard let desc = model?.modelDescription.outputDescriptionsByName[named],
          desc.type == .multiArray,
          let constraint = desc.multiArrayConstraint else { return nil }
    let shape = constraint.shape.map { $0.intValue }
    guard position < shape.count else { return nil }
    return shape[position]
}

// MARK: - Model URL Detection

/// Detects the best available CoreML model URL in a folder, preferring
/// compiled `.mlmodelc` over `.mlpackage`.
/// - Parameters:
///   - path: The folder containing the model
///   - modelName: The base name of the model (without extension)
/// - Returns: The URL to the detected model file
public func detectModelURL(inFolder path: URL, named modelName: String) -> URL {
    let compiledUrl = path.appending(path: "\(modelName).mlmodelc")
    let packageUrl = path.appending(path: "\(modelName).mlpackage/Data/com.apple.CoreML/model.mlmodel")

    let compiledModelExists = FileManager.default.fileExists(atPath: compiledUrl.path)
    let packageModelExists = FileManager.default.fileExists(atPath: packageUrl.path)

    // Prefer .mlmodelc; fall back to .mlpackage only when compiled model is absent
    if packageModelExists && !compiledModelExists {
        return packageUrl
    }
    return compiledUrl
}

/// Scans a folder for the first CoreML model bundle when the filename is not known in advance.
///
/// Prefers a compiled `.mlmodelc` bundle over an `.mlpackage` source bundle.
/// - Parameter path: The folder to scan
/// - Returns: The URL to the first detected model bundle, or `nil` if the folder is empty
///   or does not exist
public func detectModelURL(inFolder path: URL) -> URL? {
    guard let contents = try? FileManager.default.contentsOfDirectory(
        at: path, includingPropertiesForKeys: nil
    ) else { return nil }

    if let compiled = contents.first(where: { $0.pathExtension == "mlmodelc" }) {
        return compiled
    }
    if let package = contents.first(where: { $0.pathExtension == "mlpackage" }) {
        return package.appending(path: "Data/com.apple.CoreML/model.mlmodel")
    }
    return nil
}
