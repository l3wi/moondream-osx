// Copyright 2024 Moondream AI
// Language model transformer block

import Foundation
import MLX
import MLXNN

/// Text transformer block with attention and MLP
/// Layers 0-3 use DenseMLP, layers 4-23 use MoE MLP
internal class LanguageBlock: Module {
    let layerIdx: Int
    let isMoE: Bool

    @ModuleInfo var ln: LayerNorm
    @ModuleInfo(key: "attn") var attention: LanguageAttention
    var mlp: UnaryLayer

    init(_ config: Moondream3Configuration.TextConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.isMoE = config.moe.startLayer <= layerIdx

        self.ln = LayerNorm(dimensions: config.dim)
        self._attention.wrappedValue = LanguageAttention(
            dim: config.dim,
            numHeads: config.nHeads,
            numKvHeads: config.nKvHeads
        )

        if isMoE {
            self.mlp = MoEMLP(
                dim: config.dim,
                numExperts: config.moe.numExperts,
                expertDim: config.moe.expertInnerDim,
                expertsPerToken: config.moe.expertsPerToken,
                bits: config.bits ?? 4  // Support int4 (default) or int8
            )
        } else {
            self.mlp = DenseMLP(dim: config.dim, hiddenDim: config.ffDim)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        freqsCis: MLXArray,
        positions: MLXArray,
        mask: MLXArray?,
        cache: (MLXArray, MLXArray)?,
        cachePos: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let h = ln(x)
        let (attnOut, newCache) = attention(h, freqsCis: freqsCis, positions: positions, mask: mask, cache: cache, cachePos: cachePos)
        let mlpOut = mlp(h)
        return (x + attnOut + mlpOut, newCache)
    }
}
