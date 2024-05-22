//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import MLX
import MLXNN

final class MultiHeadAttention: Module {
    let nHead: Int
    let query: Linear
    let key: Linear
    let value: Linear
    let out: Linear

    init(nState: Int, nHead: Int) {
        self.nHead = nHead
        self.query = Linear(nState, nState)
        self.key = Linear(nState, nState, bias: false)
        self.value = Linear(nState, nState)
        self.out = Linear(nState, nState)
    }

    func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: MLXArray? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), MLXArray) {
        let q = query(x)

        var k: MLXArray
        var v: MLXArray
        if let xa {
            k = key(xa)
            v = value(xa)
        } else {
            k = key(x)
            v = value(x)
            if let kvCache {
                k = MLX.concatenated([kvCache[0], k], axis: 1)
                v = MLX.concatenated([kvCache[1], v], axis: 1)
            }
        }

        let (wv, qk) = qkvAttention(q, k, v, mask)
        return (out(wv), (k, v), qk)
    }

    private func qkvAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ mask: MLXArray?) -> (MLXArray, MLXArray) {
        let (nBatch, nCtx, nState) = (q.shape[0], q.shape[1], q.shape[2])
        let scale = pow(Float(nState / nHead), -0.25)
        let q = q.reshaped([q.shape[0], q.shape[1], nHead, -1]).transposed(0, 2, 1, 3) * scale
        let k = k.reshaped([k.shape[0], k.shape[1], nHead, -1]).transposed(0, 2, 3, 1) * scale
        let v = v.reshaped([v.shape[0], v.shape[1], nHead, -1]).transposed(0, 2, 1, 3)
        var qk = q.matmul(k)
        if let mask {
            qk = qk + mask[0..<nCtx, 0..<nCtx]
        }
        qk = qk.asType(.float32)
        let w = MLX.softmax(qk, axis: -1).asType(q.dtype)
        var out = w.matmul(v).transposed(0, 2, 1, 3)
        out = out.reshaped([nBatch, nCtx, nState])
        return (out, qk)
    }
}

final class ResidualAttentionBlock: Module {
    let attn: MultiHeadAttention
    let attn_ln: LayerNorm
    let mlp1: Linear
    let mlp2: Linear
    let mlp_ln: LayerNorm

    init(nState: Int, nHead: Int) {
        self.attn = MultiHeadAttention(nState: nState, nHead: nHead)
        self.attn_ln = LayerNorm(dimensions: nState)
        let nMlp = nState * 4
        self.mlp1 = Linear(nState, nMlp)
        self.mlp2 = Linear(nMlp, nState)
        self.mlp_ln = LayerNorm(dimensions: nState)
    }

    func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> ResidualAttentionBlockResult {
        let (kvCache, crossKv) = kvCache ?? (nil, nil)
        let (y, kv, _) = attn(attn_ln(x), mask: mask, kvCache: kvCache)
        var x = x + y
        x = x + mlp2(gelu(mlp1(mlp_ln(x))))
        return ResidualAttentionBlockResult(x: x, kv: kv, crossKv: crossKv, crossQk: nil)
    }
}
