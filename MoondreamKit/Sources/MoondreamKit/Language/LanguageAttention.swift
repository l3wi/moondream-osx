import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Language Attention (Standard)

/// Text attention with tau modulation for attention scaling
/// Q, K, V all have same dimension (32 heads x 64 = 2048 each)
/// Total qkv = 6144 = 2048 + 2048 + 2048
internal class LanguageAttention: Module {
    let dim: Int
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float
    let qkvDim: Int

    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear

    // Tau parameters for attention modulation
    @ModuleInfo(key: "tau_wq") var tauWq: MLXArray
    @ModuleInfo(key: "tau_wv") var tauWv: MLXArray
    @ModuleInfo(key: "tau_alpha") var tauAlpha: MLXArray

    init(dim: Int, numHeads: Int, numKvHeads: Int) {
        self.dim = dim
        self.numHeads = numHeads
        self.numKvHeads = numKvHeads
        self.headDim = dim / numHeads  // 64
        self.scale = pow(Float(headDim), -0.5)

        // Python: qkv_dim = dim + 2 * (dim * n_kv_heads // n_heads)
        // With n_heads=32, n_kv_heads=32: qkv_dim = 2048 + 2*2048 = 6144
        let kvDim = dim * numKvHeads / numHeads
        self.qkvDim = dim + 2 * kvDim

        self.qkv = Linear(dim, qkvDim, bias: true)
        self.proj = Linear(dim, dim, bias: true)

        // Tau parameters - shape (n_heads, qkv_dim) = (32, 6144)
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
        let qkvOut = qkv(x)  // [B, T, 6144]

        // Python: q_dim = n_heads * head_dim = 32 * 64 = 2048
        // Python: kv_dim = n_kv_heads * head_dim = 32 * 64 = 2048
        // Split: [q_dim, q_dim + kv_dim] = [2048, 4096]
        let qDim = numHeads * headDim  // 2048
        let kvDim = numKvHeads * headDim  // 2048

        // q, k, v = mx.split(qkv_out, [q_dim, q_dim + kv_dim], axis=-1)
        let q = qkvOut[0..., 0..., ..<qDim]  // [B, T, 2048]
        let k = qkvOut[0..., 0..., qDim..<(qDim + kvDim)]  // [B, T, 2048]
        let v = qkvOut[0..., 0..., (qDim + kvDim)...]  // [B, T, 2048]

        // Reshape to multi-head format
        // q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        var queries = q.reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)  // [B, 32, T, 64]
        var keys = k.reshaped(B, T, numKvHeads, headDim).transposed(0, 2, 1, 3)   // [B, 32, T, 64]
        var values = v.reshaped(B, T, numKvHeads, headDim).transposed(0, 2, 1, 3) // [B, 32, T, 64]

        // Tau modulation
        // tok_feat = nn.gelu(qkv_out)
        let tokFeat = gelu(qkvOut)

        // tok_q = mx.tanh(tok_feat @ self.tau_wq.T).transpose(0, 2, 1)
        let tokQ = tanh(matmul(tokFeat, tauWq.T)).transposed(0, 2, 1)  // [B, n_heads, T]
        let tokV = tanh(matmul(tokFeat, tauWv.T)).transposed(0, 2, 1)  // [B, n_heads, T]

        // pos = positions.astype(mx.float32) + 1
        let pos = positions.asType(.float32) + 1

        // tau_pos = 1 + (mx.sigmoid(self.tau_alpha[:, None] * mx.log(pos)) - 0.5)
        let logPos = log(pos)  // [T]
        let tauPosScaled = expandedDimensions(tauAlpha, axis: 1) * logPos  // [n_heads, T]
        let tauPos = 1 + (sigmoid(tauPosScaled) - 0.5)  // [n_heads, T]

        // tau_q = (tok_q + tau_pos[None])[..., None]
        let tauQ = expandedDimensions(tokQ + tauPos, axis: -1)  // [B, n_heads, T, 1]
        let tauV = expandedDimensions(tokV + tauPos, axis: -1)  // [B, n_heads, T, 1]

        // Apply tau modulation to Q and V
        queries = queries * tauQ
        values = values * tauV

        // Apply RoPE to Q and K
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

        // Handle GQA if n_kv_heads != n_heads (not needed here since both are 32)
        if numKvHeads != numHeads {
            let nRep = numHeads / numKvHeads
            keys = repeated(keys, count: nRep, axis: 1)
            values = repeated(values, count: nRep, axis: 1)
        }

        // Standard scaled dot-product attention (NOT differential attention)
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

        // out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        output = output.transposed(0, 2, 1, 3).reshaped(B, T, C)
        return (proj(output), newCache)
    }
}

// MARK: - Language Attention (Quantized)

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
