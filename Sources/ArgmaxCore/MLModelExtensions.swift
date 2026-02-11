//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML

// MARK: - Async Prediction

public extension MLModel {
    /// Async wrapper for `MLModel.prediction` that uses native async prediction
    /// on macOS 14+ / iOS 17+ and falls back to a Task-wrapped call on older OS.
    func asyncPrediction(
        from input: MLFeatureProvider,
        options: MLPredictionOptions
    ) async throws -> MLFeatureProvider {
        if #available(macOS 14, iOS 17, watchOS 10, visionOS 1, *) {
            return try await prediction(from: input, options: options)
        } else {
            return try await Task {
                try prediction(from: input, options: options)
            }.value
        }
    }
}

// MARK: - Compute Units Description

public extension MLComputeUnits {
    var description: String {
        switch self {
            case .cpuOnly:
                return "cpuOnly"
            case .cpuAndGPU:
                return "cpuAndGPU"
            case .all:
                return "all"
            case .cpuAndNeuralEngine:
                return "cpuAndNeuralEngine"
            @unknown default:
                return "unknown"
        }
    }
}
