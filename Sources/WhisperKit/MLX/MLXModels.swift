//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import MLX
import MLXNN
import WhisperKit

public enum PadMode {
    case constant
    case reflect
}

struct MLXModelConfig: Codable {
    let nMels: Int
    let nAudioCtx: Int
    let nAudioState: Int
    let nAudioHead: Int
    let nAudioLayer: Int
    let nVocab: Int
    let nTextCtx: Int
    let nTextState: Int
    let nTextHead: Int
    let nTextLayer: Int
}

struct KV {
    var k: MLXArray
    var v: MLXArray
}

struct TextDecoderResult {
    var logits: MLXArray
    var kvCache: [KV]
}

struct ResidualAttentionBlockResult {
    var x: MLXArray
    var kv: KV
    var crossKv: KV?
    var crossQk: MLXArray?
}

struct MultiHeadAttentionResult {
    var x: MLXArray
    var kv: KV
    var qk: MLXArray
}
