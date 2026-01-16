// Copyright 2024 Moondream AI
// Quantized Language model (TextModel)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Quantized language model with transformer blocks
/// Uses int4 quantized embedding and output layers
internal class QuantizedTextModel: Module, KVCacheDimensionProvider {
    let config: Moondream3Configuration.TextConfiguration

    @ModuleInfo(key: "wte") var wte: QuantizedEmbedding
    let blocks: [QuantizedLanguageBlock]
    @ModuleInfo(key: "post_ln") var postLn: LayerNorm
    @ModuleInfo(key: "lm_head") var lmHead: QuantizedLinear

    let freqsCis: MLXArray

    var kvHeads: [Int]

    init(_ config: Moondream3Configuration.TextConfiguration) {
        self.config = config

        self._wte.wrappedValue = QuantizedEmbedding(embeddingCount: config.vocabSize, dimensions: config.dim, groupSize: 64, bits: 4)
        self.blocks = (0..<config.nLayers).map { QuantizedLanguageBlock(config, layerIdx: $0) }
        self._postLn.wrappedValue = LayerNorm(dimensions: config.dim)
        self._lmHead.wrappedValue = QuantizedLinear(config.dim, config.vocabSize, bias: true, groupSize: 64, bits: 4)

        let rotDim = config.dim / (2 * config.nHeads)
        self.freqsCis = precomputeFreqsCis(dim: rotDim, maxLen: config.maxContext)

        self.kvHeads = (0..<config.nLayers).map { _ in 16 }
    }

    func embed(_ tokens: MLXArray) -> MLXArray {
        return wte(tokens)
    }

    func callAsFunction(
        _ embeddings: MLXArray,
        positions: MLXArray,
        mask: MLXArray?,
        cache: [(MLXArray, MLXArray)]?,
        cachePos: Int
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var x = embeddings
        var newCaches: [(MLXArray, MLXArray)] = []

        for (i, block) in blocks.enumerated() {
            let blockCache = cache?[i]
            let (out, newCache) = block(x, freqsCis: freqsCis, positions: positions, mask: mask, cache: blockCache, cachePos: cachePos)
            x = out
            newCaches.append(newCache)
        }

        return (x, newCaches)
    }

    func generateLogits(_ hidden: MLXArray) -> MLXArray {
        let lastHidden = hidden[0..., (-1)..., 0...].squeezed(axis: 1)
        let normed = postLn(lastHidden)
        return lmHead(normed)
    }
}
