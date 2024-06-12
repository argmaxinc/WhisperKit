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
        kvCache: KV? = nil
    ) -> MultiHeadAttentionResult {
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
                k = MLX.concatenated([kvCache.k, k], axis: 1)
                v = MLX.concatenated([kvCache.v, v], axis: 1)
            }
        }

        let (wv, qk) = qkvAttention(q, k, v, mask)
        return MultiHeadAttentionResult(
            x: out(wv),
            kv: KV(k: k, v: v),
            qk: qk
        )
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
    let cross_attn: MultiHeadAttention?
    let cross_attn_ln: LayerNorm?

    init(nState: Int, nHead: Int, crossAttention: Bool = false) {
        self.attn = MultiHeadAttention(nState: nState, nHead: nHead)
        self.attn_ln = LayerNorm(dimensions: nState)
        self.cross_attn = crossAttention ? MultiHeadAttention(nState: nState, nHead: nHead) : nil
        self.cross_attn_ln = crossAttention ? LayerNorm(dimensions: nState) : nil
        let nMlp = nState * 4
        self.mlp1 = Linear(nState, nMlp)
        self.mlp2 = Linear(nMlp, nState)
        self.mlp_ln = LayerNorm(dimensions: nState)
    }

    func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: KV? = nil,
        crossKvCache: KV? = nil
    ) -> ResidualAttentionBlockResult {
        let attnResult = attn(attn_ln(x), mask: mask, kvCache: kvCache)
        var x = x + attnResult.x
        if let cross_attn, let cross_attn_ln {
            let crossAttnResult = cross_attn(cross_attn_ln(x), xa: xa, kvCache: crossKvCache)
            x = x + crossAttnResult.x
            x = x + mlp2(gelu(mlp1(mlp_ln(x))))
            return ResidualAttentionBlockResult(
                x: x,
                kv: attnResult.kv,
                crossKv: crossAttnResult.kv,
                crossQk: crossAttnResult.qk
            )
        } else {
            x = x + mlp2(gelu(mlp1(mlp_ln(x))))
            return ResidualAttentionBlockResult(
                x: x,
                kv: attnResult.kv,
                crossKv: crossKvCache,
                crossQk: nil
            )
        }
    }
}
