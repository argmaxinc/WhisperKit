//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML

// MARK: - TTS Embed Type Protocols

/// These protocols allow embed tensors to be represented as either MLMultiArray (all platforms)
/// or MLTensor (macOS 15+ / iOS 18+) without changing the calling convention.

/// Marker protocol for a raw TTS embedding tensor emitted by a CoreML model.
public protocol TTSEmbedTensorType {}

/// Marker protocol for a TTS embedding value that can be used as CoreML model input.
public protocol TTSEmbedInputType {}

extension MLMultiArray: TTSEmbedTensorType {}
extension MLMultiArray: TTSEmbedInputType {}

#if canImport(CoreML.MLState)
@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension MLTensor: TTSEmbedTensorType {}

@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension MLTensor: TTSEmbedInputType {}
#endif
