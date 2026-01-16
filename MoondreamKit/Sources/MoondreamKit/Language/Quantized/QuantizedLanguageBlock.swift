// Copyright 2024 Moondream AI
// Quantized Language model transformer block

import Foundation
import MLX
import MLXNN

/// Quantized text transformer block
/// Layers 0-3 use QuantizedDenseMLP, layers 4-23 use MoE MLP
internal class QuantizedLanguageBlock: Module {
    let layerIdx: Int
    let isMoE: Bool

    @ModuleInfo var ln: LayerNorm
    @ModuleInfo(key: "attn") var attention: QuantizedLanguageAttention
    var mlp: UnaryLayer

    init(_ config: Moondream3Configuration.TextConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.isMoE = config.moe.startLayer <= layerIdx

        self.ln = LayerNorm(dimensions: config.dim)
        self._attention.wrappedValue = QuantizedLanguageAttention(
            dim: config.dim,
            numHeads: config.nHeads,
            numKvHeads: config.nKvHeads
        )

        if isMoE {
            // MoE layers use the same MoEMLP (already quantized)
            self.mlp = MoEMLP(
                dim: config.dim,
                numExperts: config.moe.numExperts,
                expertDim: config.moe.expertInnerDim,
                expertsPerToken: config.moe.expertsPerToken,
                bits: config.bits ?? 4
            )
        } else {
            self.mlp = QuantizedDenseMLP(dim: config.dim, hiddenDim: config.ffDim)
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
