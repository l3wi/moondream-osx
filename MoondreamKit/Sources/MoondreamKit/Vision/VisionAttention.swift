// Copyright 2024 Moondream AI
// Vision Attention layer for standard model

import Foundation
import MLX
import MLXFast
import MLXNN

/// Vision attention layer with fused QKV projection
/// Used in the Vision Transformer encoder for image processing
internal class VisionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear

    init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)

        self.qkv = Linear(dim, dim * 3, bias: true)
        self.proj = Linear(dim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))

        // Python: qkv = qkv.reshape(B, T, 3, n_heads, head_dim).transpose(0, 2, 3, 1, 4)
        let qkvOut = qkv(x)
        let qkvReshaped = qkvOut.reshaped(B, T, 3, numHeads, headDim).transposed(0, 2, 3, 1, 4)
        let q = qkvReshaped[0..., 0, 0..., 0..., 0...]
        let k = qkvReshaped[0..., 1, 0..., 0..., 0...]
        let v = qkvReshaped[0..., 2, 0..., 0..., 0...]

        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, T, C)

        return proj(output)
    }
}
