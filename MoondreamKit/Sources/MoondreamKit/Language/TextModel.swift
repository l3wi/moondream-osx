// Copyright 2024 Moondream AI
// Language models (standard and quantized variants)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Dense MLP (Standard)

/// Dense MLP (for non-MoE layers 0-3)
/// Uses GELU approximate activation
internal class DenseMLP: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(dim: Int, hiddenDim: Int) {
        self.fc1 = Linear(dim, hiddenDim, bias: true)
        self.fc2 = Linear(hiddenDim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

// MARK: - Dense MLP (Quantized)

/// Quantized Dense MLP for non-MoE layers (0-3)
/// Uses int4 quantized linear layers
internal class QuantizedDenseMLP: Module, UnaryLayer {
    @ModuleInfo var fc1: QuantizedLinear
    @ModuleInfo var fc2: QuantizedLinear

    init(dim: Int, hiddenDim: Int) {
        self.fc1 = QuantizedLinear(dim, hiddenDim, bias: true, groupSize: 64, bits: 4)
        self.fc2 = QuantizedLinear(hiddenDim, dim, bias: true, groupSize: 64, bits: 4)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

// MARK: - Text Model (Standard)

/// Language model with transformer blocks
/// 24 layers, 2048-dim, 32 heads
internal class TextModel: Module, KVCacheDimensionProvider {
    let config: Moondream3Configuration.TextConfiguration

    @ModuleInfo(key: "wte") var wte: Embedding
    let blocks: [LanguageBlock]
    @ModuleInfo(key: "post_ln") var postLn: LayerNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let freqsCis: MLXArray

    var kvHeads: [Int]

    init(_ config: Moondream3Configuration.TextConfiguration) {
        self.config = config

        self._wte.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.dim)
        self.blocks = (0..<config.nLayers).map { LanguageBlock(config, layerIdx: $0) }
        self._postLn.wrappedValue = LayerNorm(dimensions: config.dim)
        self._lmHead.wrappedValue = Linear(config.dim, config.vocabSize, bias: true)

        // Precompute RoPE frequencies
        // Python: rot_dim = config.dim // (2 * config.n_heads)
        // With dim=2048, n_heads=32: rot_dim = 2048 // 64 = 32
        let rotDim = config.dim / (2 * config.nHeads)
        self.freqsCis = precomputeFreqsCis(dim: rotDim, maxLen: config.maxContext)

        // Effective KV heads = kvDim / headDim = 1024 / 64 = 16
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
        // Python: hidden = hidden[:, -1, :]  # Shape [B, dim]
        // Swift (-1)... creates a range, keeping the dimension as [B, 1, dim]
        // We need to squeeze to get [B, dim]
        let lastHidden = hidden[0..., (-1)..., 0...].squeezed(axis: 1)
        let normed = postLn(lastHidden)
        return lmHead(normed)
    }
}

// MARK: - Text Model (Quantized)

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
