//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import MLX

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

struct ResidualAttentionBlockResult {
    var x: MLXArray
    var kv: (MLXArray, MLXArray)
    var crossKv: MLXArray?
    var crossQk: MLXArray?
}
