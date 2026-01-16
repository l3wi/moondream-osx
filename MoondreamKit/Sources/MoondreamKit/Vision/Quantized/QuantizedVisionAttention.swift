// Copyright 2024 Moondream AI
// Quantized Vision Attention layer

import Foundation
import MLX
import MLXFast
import MLXNN

/// Quantized vision attention layer with int4 QKV projection
/// Used in the fully quantized model variant (md3p-int4-smol)
internal class QuantizedVisionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var qkv: QuantizedLinear
    @ModuleInfo var proj: QuantizedLinear

    init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)

        self.qkv = QuantizedLinear(dim, dim * 3, bias: true, groupSize: 64, bits: 4)
        self.proj = QuantizedLinear(dim, dim, bias: true, groupSize: 64, bits: 4)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))

        let qkvOut = qkv(x)
        let qkvReshaped = qkvOut.reshaped(B, T, 3, numHeads, headDim).transposed(0, 2, 3, 1, 4)
        let q = qkvReshaped[0..., 0, 0..., 0..., 0...]
        let k = qkvReshaped[0..., 1, 0..., 0..., 0...]
        let v = qkvReshaped[0..., 2, 0..., 0..., 0...]

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, T, C)

        return proj(output)
    }
}
