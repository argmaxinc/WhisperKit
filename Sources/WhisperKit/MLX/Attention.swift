//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import MLX
import MLXNN

final class MultiHeadAttention: Module {
    let nHead: Int
    @ModuleInfo(key: "query") private var query: Linear
    @ModuleInfo(key: "key") private var key: Linear
    @ModuleInfo(key: "value") private var value: Linear
    @ModuleInfo(key: "out") private var out: Linear

    init(nState: Int, nHead: Int) {
        self.nHead = nHead
        self._query.wrappedValue = Linear(nState, nState)
        self._key.wrappedValue = Linear(nState, nState, bias: false)
        self._value.wrappedValue = Linear(nState, nState)
        self._out.wrappedValue = Linear(nState, nState)
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
    @ModuleInfo(key: "attn") private var attn: MultiHeadAttention
    @ModuleInfo(key: "attn_ln") private var attnLn: LayerNorm
    @ModuleInfo(key: "mlp1") private var mlp1: Linear
    @ModuleInfo(key: "mlp2") private var mlp2: Linear
    @ModuleInfo(key: "mlp_ln") private var mlpLn: LayerNorm
    @ModuleInfo(key: "cross_attn") private var crossAttn: MultiHeadAttention?
    @ModuleInfo(key: "cross_attn_ln") private var crossAttnLn: LayerNorm?

    init(nState: Int, nHead: Int, crossAttention: Bool = false) {
        let nMlp = nState * 4
        self._attn.wrappedValue = MultiHeadAttention(nState: nState, nHead: nHead)
        self._attnLn.wrappedValue = LayerNorm(dimensions: nState)
        self._crossAttn.wrappedValue = crossAttention ? MultiHeadAttention(nState: nState, nHead: nHead) : nil
        self._crossAttnLn.wrappedValue = crossAttention ? LayerNorm(dimensions: nState) : nil
        self._mlp1.wrappedValue = Linear(nState, nMlp)
        self._mlp2.wrappedValue = Linear(nMlp, nState)
        self._mlpLn.wrappedValue = LayerNorm(dimensions: nState)
    }

    func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: KV? = nil,
        crossKvCache: KV? = nil
    ) -> ResidualAttentionBlockResult {
        let attnResult = attn(attnLn(x), mask: mask, kvCache: kvCache)
        var x = x + attnResult.x
        if let crossAttn, let crossAttnLn {
            let crossAttnResult = crossAttn(crossAttnLn(x), xa: xa, kvCache: crossKvCache)
            x = x + crossAttnResult.x
            x = x + mlp2(gelu(mlp1(mlpLn(x))))
            return ResidualAttentionBlockResult(
                x: x,
                kv: attnResult.kv,
                crossKv: crossAttnResult.kv,
                crossQk: crossAttnResult.qk
            )
        } else {
            x = x + mlp2(gelu(mlp1(mlpLn(x))))
            return ResidualAttentionBlockResult(
                x: x,
                kv: attnResult.kv,
                crossKv: crossKvCache,
                crossQk: nil
            )
        }
    }
}
