// Copyright 2024 Moondream AI
// Quantized Language model attention with Tau modulation

import Foundation
import MLX
import MLXFast
import MLXNN

/// Quantized text attention with tau modulation
/// Uses int4 quantized QKV and projection layers
internal class QuantizedLanguageAttention: Module {
    let dim: Int
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float
    let qkvDim: Int

    @ModuleInfo var qkv: QuantizedLinear
    @ModuleInfo var proj: QuantizedLinear

    @ModuleInfo(key: "tau_wq") var tauWq: MLXArray
    @ModuleInfo(key: "tau_wv") var tauWv: MLXArray
    @ModuleInfo(key: "tau_alpha") var tauAlpha: MLXArray

    init(dim: Int, numHeads: Int, numKvHeads: Int) {
        self.dim = dim
        self.numHeads = numHeads
        self.numKvHeads = numKvHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)

        let kvDim = dim * numKvHeads / numHeads
        self.qkvDim = dim + 2 * kvDim

        self.qkv = QuantizedLinear(dim, qkvDim, bias: true, groupSize: 64, bits: 4)
        self.proj = QuantizedLinear(dim, dim, bias: true, groupSize: 64, bits: 4)

        self._tauWq.wrappedValue = MLXArray.zeros([numHeads, qkvDim])
        self._tauWv.wrappedValue = MLXArray.zeros([numHeads, qkvDim])
        self._tauAlpha.wrappedValue = MLXArray.zeros([numHeads])
    }

    func callAsFunction(
        _ x: MLXArray,
        freqsCis: MLXArray,
        positions: MLXArray,
        mask: MLXArray?,
        cache: (MLXArray, MLXArray)?,
        cachePos: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))
        let qkvOut = qkv(x)

        let qDim = numHeads * headDim
        let kvDim = numKvHeads * headDim

        let q = qkvOut[0..., 0..., ..<qDim]
        let k = qkvOut[0..., 0..., qDim..<(qDim + kvDim)]
        let v = qkvOut[0..., 0..., (qDim + kvDim)...]

        var queries = q.reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = k.reshaped(B, T, numKvHeads, headDim).transposed(0, 2, 1, 3)
        var values = v.reshaped(B, T, numKvHeads, headDim).transposed(0, 2, 1, 3)

        let tokFeat = gelu(qkvOut)
        let tokQ = tanh(matmul(tokFeat, tauWq.T)).transposed(0, 2, 1)
        let tokV = tanh(matmul(tokFeat, tauWv.T)).transposed(0, 2, 1)

        let pos = positions.asType(.float32) + 1
        let logPos = log(pos)
        let tauPosScaled = expandedDimensions(tauAlpha, axis: 1) * logPos
        let tauPos = 1 + (sigmoid(tauPosScaled) - 0.5)

        let tauQ = expandedDimensions(tokQ + tauPos, axis: -1)
        let tauV = expandedDimensions(tokV + tauPos, axis: -1)

        queries = queries * tauQ
        values = values * tauV

        queries = applyRotaryEmb(queries, freqsCis: freqsCis, positions: positions)
        keys = applyRotaryEmb(keys, freqsCis: freqsCis, positions: positions)

        // KV cache handling using shared manager
        let (newCache, keysForAttn, valuesForAttn) = KVCacheManager.updateCache(
            keys: keys,
            values: values,
            cache: cache,
            cachePos: cachePos
        )
        keys = keysForAttn
        values = valuesForAttn

        if numKvHeads != numHeads {
            let nRep = numHeads / numKvHeads
            keys = repeated(keys, count: nRep, axis: 1)
            values = repeated(values, count: nRep, axis: 1)
        }

        var output: MLXArray
        if let mask = mask {
            let qLen = queries.dim(2)
            let kvLen = keys.dim(2)
            let maskSlice = mask[0..., 0..., ..<qLen, ..<kvLen]

            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: .array(maskSlice)
            )
        } else {
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: .none
            )
        }

        output = output.transposed(0, 2, 1, 3).reshaped(B, T, C)
        return (proj(output), newCache)
    }
}
