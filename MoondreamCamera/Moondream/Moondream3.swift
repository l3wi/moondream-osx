// Copyright © 2024 Apple Inc. / Moondream AI
// Moondream3 VLM implementation for MLX Swift

import CoreGraphics
import CoreImage
import Foundation
import Hub
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

// MARK: - Configuration

/// Configuration for Moondream3 model
public struct Moondream3Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let dim: Int
        public let ffDim: Int
        public let nLayers: Int
        public let vocabSize: Int
        public let maxContext: Int
        public let nHeads: Int
        public let nKvHeads: Int
        public let prefixAttn: Int
        public let moe: MoEConfiguration

        enum CodingKeys: String, CodingKey {
            case dim
            case ffDim = "ff_dim"
            case nLayers = "n_layers"
            case vocabSize = "vocab_size"
            case maxContext = "max_context"
            case nHeads = "n_heads"
            case nKvHeads = "n_kv_heads"
            case prefixAttn = "prefix_attn"
            case moe
        }
    }

    public struct MoEConfiguration: Codable, Sendable {
        public let numExperts: Int
        public let startLayer: Int
        public let expertsPerToken: Int
        public let expertInnerDim: Int

        enum CodingKeys: String, CodingKey {
            case numExperts = "num_experts"
            case startLayer = "start_layer"
            case expertsPerToken = "experts_per_token"
            case expertInnerDim = "expert_inner_dim"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let encDim: Int
        public let encPatchSize: Int
        public let encNLayers: Int
        public let encFfDim: Int
        public let encNHeads: Int
        public let projOutDim: Int
        public let cropSize: Int
        public let inChannels: Int
        public let maxCrops: Int
        public let overlapMargin: Int
        public let projInnerDim: Int

        enum CodingKeys: String, CodingKey {
            case encDim = "enc_dim"
            case encPatchSize = "enc_patch_size"
            case encNLayers = "enc_n_layers"
            case encFfDim = "enc_ff_dim"
            case encNHeads = "enc_n_heads"
            case projOutDim = "proj_out_dim"
            case cropSize = "crop_size"
            case inChannels = "in_channels"
            case maxCrops = "max_crops"
            case overlapMargin = "overlap_margin"
            case projInnerDim = "proj_inner_dim"
        }
    }

    public struct RegionConfiguration: Codable, Sendable {
        public let dim: Int
        public let coordFeatDim: Int
        public let coordOutDim: Int
        public let sizeFeatDim: Int
        public let sizeOutDim: Int

        enum CodingKeys: String, CodingKey {
            case dim
            case coordFeatDim = "coord_feat_dim"
            case coordOutDim = "coord_out_dim"
            case sizeFeatDim = "size_feat_dim"
            case sizeOutDim = "size_out_dim"
        }
    }

    public let modelType: String
    public let text: TextConfiguration
    public let vision: VisionConfiguration
    public let region: RegionConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text
        case vision
        case region
    }
}

// MARK: - Vision Encoder

private enum Vision {

    /// Vision attention layer with fused QKV
    /// Keys: qkv, proj
    class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var qkv: Linear  // Fused Q, K, V projection
        @ModuleInfo var proj: Linear  // Output projection

        init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            self.headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            // Fused QKV: output is 3 * dims
            self.qkv = Linear(dims, dims * 3, bias: true)
            self.proj = Linear(dims, dims, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

            // Fused QKV projection
            let qkvOut = qkv(x)
            let qkvSplit = qkvOut.split(parts: 3, axis: -1)
            var queries = qkvSplit[0]
            var keys = qkvSplit[1]
            var values = qkvSplit[2]

            queries = queries.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return proj(output)
        }
    }

    /// Vision MLP
    class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(dims: Int, hiddenDims: Int) {
            self.fc1 = Linear(dims, hiddenDims, bias: true)
            self.fc2 = Linear(hiddenDims, dims, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(gelu(fc1(x)))
        }
    }

    /// Vision transformer layer
    /// Keys: attn, ln1, ln2, mlp
    class TransformerLayer: Module {
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "ln1") var ln1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "ln2") var ln2: LayerNorm

        init(_ config: Moondream3Configuration.VisionConfiguration) {
            self._attention.wrappedValue = Attention(
                dims: config.encDim,
                numHeads: config.encNHeads
            )
            self._ln1.wrappedValue = LayerNorm(dimensions: config.encDim)
            self.mlp = MLP(dims: config.encDim, hiddenDims: config.encFfDim)
            self._ln2.wrappedValue = LayerNorm(dimensions: config.encDim)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x + attention(ln1(x))
            h = h + mlp(ln2(h))
            return h
        }
    }

    /// Vision encoder with integrated projector
    /// Keys: patch_emb, pos_emb, layers, post_ln, proj_fc1, proj_fc2
    class Encoder: Module {
        // Patch embedding (Conv2d)
        @ModuleInfo(key: "patch_emb") var patchEmb: Conv2d
        // Position embedding (not Embedding module, just a parameter)
        @ModuleInfo(key: "pos_emb") var posEmb: MLXArray

        let layers: [TransformerLayer]
        @ModuleInfo(key: "post_ln") var postLn: LayerNorm

        // Projector layers (integrated into vision encoder)
        @ModuleInfo(key: "proj_fc1") var projFc1: Linear
        @ModuleInfo(key: "proj_fc2") var projFc2: Linear

        let numPatches: Int

        init(_ config: Moondream3Configuration.VisionConfiguration) {
            let patchSize = config.encPatchSize
            self._patchEmb.wrappedValue = Conv2d(
                inputChannels: config.inChannels,
                outputChannels: config.encDim,
                kernelSize: .init(patchSize),
                stride: .init(patchSize)
            )

            let gridSize = config.cropSize / patchSize
            self.numPatches = gridSize * gridSize

            // Position embedding as a parameter (will be loaded from weights)
            self._posEmb.wrappedValue = MLXArray.zeros([numPatches, config.encDim])

            self.layers = (0 ..< config.encNLayers).map { _ in
                TransformerLayer(config)
            }
            self._postLn.wrappedValue = LayerNorm(dimensions: config.encDim)

            // Projector: vision dim -> inner dim -> text dim
            self._projFc1.wrappedValue = Linear(config.encDim, config.projInnerDim, bias: true)
            self._projFc2.wrappedValue = Linear(config.projInnerDim, config.projOutDim, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // Patch embedding
            var patches = patchEmb(x)
            patches = patches.flattened(start: 1, end: 2)

            // Add position embedding
            patches = patches + posEmb

            // Transformer layers
            var h = patches
            for layer in layers {
                h = layer(h)
            }
            h = postLn(h)

            // Project to text dimension
            h = projFc2(gelu(projFc1(h)))
            return h
        }
    }
}

// MARK: - Language Model

private enum Language {

    /// RMSNorm for language model
    class RMSNorm: Module, UnaryLayer {
        let weight: MLXArray
        let eps: Float

        init(dims: Int, eps: Float = 1e-6) {
            self.weight = MLXArray.ones([dims])
            self.eps = eps
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let norm = MLX.sqrt(mean(x * x, axis: -1, keepDims: true) + eps)
            return weight * (x / norm)
        }
    }

    /// Language attention with fused QKV and tau (differential) attention
    /// Architecture: qkv -> fused projection, proj -> output, tau_* -> differential attention params
    class Attention: Module {
        let config: Moondream3Configuration.TextConfiguration
        let scale: Float
        let rope: RoPE
        let numHeads: Int
        let headDim: Int

        // Fused QKV projection
        @ModuleInfo var qkv: Linear
        // Output projection
        @ModuleInfo var proj: Linear

        // Tau (differential) attention parameters - per head
        @ModuleInfo(key: "tau_alpha") var tauAlpha: MLXArray
        @ModuleInfo(key: "tau_wq") var tauWq: MLXArray
        @ModuleInfo(key: "tau_wv") var tauWv: MLXArray

        init(_ config: Moondream3Configuration.TextConfiguration) {
            self.config = config
            self.numHeads = config.nHeads
            self.headDim = config.dim / config.nHeads
            self.scale = pow(Float(headDim), -0.5)

            // Fused QKV: output is [Q, K, V] concatenated = 3 * dim
            self.qkv = Linear(config.dim, config.dim * 3, bias: true)
            self.proj = Linear(config.dim, config.dim, bias: true)

            // Tau parameters initialized (will be loaded from weights)
            self._tauAlpha.wrappedValue = MLXArray.ones([numHeads])
            self._tauWq.wrappedValue = MLXArray.zeros([numHeads, config.dim * 3])
            self._tauWv.wrappedValue = MLXArray.zeros([numHeads, config.dim * 3])

            self.rope = RoPE(dimensions: headDim, traditional: false, base: 10000)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            // Fused QKV projection
            let qkvOut = qkv(x)

            // Split into Q, K, V
            let qkvSplit = qkvOut.split(parts: 3, axis: -1)
            var queries = qkvSplit[0]
            var keys = qkvSplit[1]
            var values = qkvSplit[2]

            // Reshape for multi-head attention
            queries = queries.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

            // Apply RoPE
            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }

            // Standard attention (tau modulation can be added later if needed)
            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return proj(output)
        }
    }

    /// Standard MLP (used for first few layers before MoE kicks in)
    /// Uses simple 2-layer structure: fc1 -> GELU -> fc2
    class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(dims: Int, hiddenDims: Int) {
            self.fc1 = Linear(dims, hiddenDims, bias: true)
            self.fc2 = Linear(hiddenDims, dims, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(gelu(fc1(x)))
        }
    }

    /// Quantized Mixture of Experts layer
    /// Uses packed quantized weights for all experts: fc1_q, fc1_scales, fc1_biases, etc.
    /// This is a simplified implementation that loads weights but uses dequantized computation
    class MoE: Module, UnaryLayer {
        let numExperts: Int
        let expertsPerToken: Int
        let dim: Int

        @ModuleInfo var router: Linear

        // Quantized fc1 weights for all experts (packed)
        @ModuleInfo(key: "fc1_q") var fc1Q: MLXArray
        @ModuleInfo(key: "fc1_scales") var fc1Scales: MLXArray
        @ModuleInfo(key: "fc1_biases") var fc1Biases: MLXArray

        // Quantized fc2 weights for all experts (packed)
        @ModuleInfo(key: "fc2_q") var fc2Q: MLXArray
        @ModuleInfo(key: "fc2_scales") var fc2Scales: MLXArray
        @ModuleInfo(key: "fc2_biases") var fc2Biases: MLXArray

        init(_ config: Moondream3Configuration.TextConfiguration) {
            let moeConfig = config.moe
            self.numExperts = moeConfig.numExperts
            self.expertsPerToken = moeConfig.expertsPerToken
            self.dim = config.dim

            // Router has bias in this model
            self.router = Linear(config.dim, moeConfig.numExperts, bias: true)

            // Initialize quantized weight placeholders (will be loaded from weights)
            // fc1: [num_experts, expert_dim, quantized_dim]
            // Actual shapes will be set when weights are loaded
            self._fc1Q.wrappedValue = MLXArray.zeros([numExperts, moeConfig.expertInnerDim, 256])
            self._fc1Scales.wrappedValue = MLXArray.zeros([numExperts, moeConfig.expertInnerDim, 32])
            self._fc1Biases.wrappedValue = MLXArray.zeros([numExperts, moeConfig.expertInnerDim, 32])

            self._fc2Q.wrappedValue = MLXArray.zeros([numExperts, config.dim, 128])
            self._fc2Scales.wrappedValue = MLXArray.zeros([numExperts, config.dim, 16])
            self._fc2Biases.wrappedValue = MLXArray.zeros([numExperts, config.dim, 16])
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
            let xFlat = x.reshaped(-1, D)

            // Get router logits and select top-k experts
            let routerLogits = router(xFlat)

            // Implement top-k manually using argSort
            let sortedIndices = argSort(routerLogits, axis: -1)
            let topkIndices = sortedIndices[0..., (numExperts - expertsPerToken)...]

            // Get top-k logits for softmax normalization
            var topkLogitsList = [MLXArray]()
            for i in 0 ..< expertsPerToken {
                let idx = topkIndices[0..., i]
                let gathered = take(routerLogits, idx.flattened(), axis: 1).diagonal()
                topkLogitsList.append(gathered)
            }
            let topkLogits = stacked(topkLogitsList, axis: 1)
            let topkWeights = softmax(topkLogits, axis: -1)

            // For now, use a simplified computation
            // TODO: Implement proper quantized expert computation
            // This is a placeholder that won't give correct results but allows loading
            var output = MLXArray.zeros(like: xFlat)

            // Simplified: just use router weights as a proxy for output scaling
            // Real implementation would dequantize and apply expert MLPs
            let avgWeight = MLX.mean(topkWeights, axis: -1, keepDims: true)
            output = xFlat * avgWeight

            return output.reshaped(B, L, D)
        }
    }

    /// Unified transformer block that handles both MLP and MoE
    /// For simplicity, we use MoE for all layers and let the MoE handle both cases
    /// (MoE with 1 expert = MLP behavior for non-MoE layers)
    class TransformerBlock: Module {
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var ln: LayerNorm
        @ModuleInfo var mlp: FeedForward  // Unified feed-forward (MLP or MoE)

        init(_ config: Moondream3Configuration.TextConfiguration, layerIdx: Int) {
            self._attention.wrappedValue = Attention(config)
            self.ln = LayerNorm(dimensions: config.dim)
            self._mlp.wrappedValue = FeedForward(config, layerIdx: layerIdx)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?
        ) -> MLXArray {
            let normed = ln(x)
            let h = x + attention(normed, mask: mask, cache: cache)
            return h + mlp(normed)
        }
    }

    /// Unified feed-forward layer that handles both standard MLP and MoE
    class FeedForward: Module, UnaryLayer {
        // For standard MLP (layers 0-3)
        @ModuleInfo var fc1: Linear?
        @ModuleInfo var fc2: Linear?

        // For MoE (layers 4+)
        @ModuleInfo var router: Linear?
        @ModuleInfo(key: "fc1_q") var fc1Q: MLXArray?
        @ModuleInfo(key: "fc1_scales") var fc1Scales: MLXArray?
        @ModuleInfo(key: "fc1_biases") var fc1Biases: MLXArray?
        @ModuleInfo(key: "fc2_q") var fc2Q: MLXArray?
        @ModuleInfo(key: "fc2_scales") var fc2Scales: MLXArray?
        @ModuleInfo(key: "fc2_biases") var fc2Biases: MLXArray?

        let isMoE: Bool
        let numExperts: Int
        let expertsPerToken: Int
        let dim: Int
        let expertInnerDim: Int

        init(_ config: Moondream3Configuration.TextConfiguration, layerIdx: Int) {
            self.isMoE = layerIdx >= config.moe.startLayer
            self.numExperts = config.moe.numExperts
            self.expertsPerToken = config.moe.expertsPerToken
            self.dim = config.dim
            self.expertInnerDim = config.moe.expertInnerDim

            if isMoE {
                // MoE layer - initialize placeholders (weights loaded from file)
                // fc1 outputs 2 * expertInnerDim (gate + up for SwiGLU)
                // fc2 takes expertInnerDim input and outputs dim
                let fc1OutDim = 2 * config.moe.expertInnerDim  // 2048 for gate+up
                self._router.wrappedValue = Linear(config.dim, config.moe.numExperts, bias: true)
                self._fc1Q.wrappedValue = MLXArray.zeros([numExperts, fc1OutDim, 256])  // [64, 2048, 256]
                self._fc1Scales.wrappedValue = MLXArray.zeros([numExperts, fc1OutDim, 32])  // [64, 2048, 32]
                self._fc1Biases.wrappedValue = MLXArray.zeros([numExperts, fc1OutDim, 32])  // [64, 2048, 32]
                self._fc2Q.wrappedValue = MLXArray.zeros([numExperts, config.dim, 128])  // [64, 2048, 128]
                self._fc2Scales.wrappedValue = MLXArray.zeros([numExperts, config.dim, 16])  // [64, 2048, 16]
                self._fc2Biases.wrappedValue = MLXArray.zeros([numExperts, config.dim, 16])  // [64, 2048, 16]
                self._fc1.wrappedValue = nil
                self._fc2.wrappedValue = nil
            } else {
                // Standard MLP
                self._fc1.wrappedValue = Linear(config.dim, config.ffDim, bias: true)
                self._fc2.wrappedValue = Linear(config.ffDim, config.dim, bias: true)
                self._router.wrappedValue = nil
                self._fc1Q.wrappedValue = nil
                self._fc1Scales.wrappedValue = nil
                self._fc1Biases.wrappedValue = nil
                self._fc2Q.wrappedValue = nil
                self._fc2Scales.wrappedValue = nil
                self._fc2Biases.wrappedValue = nil
            }
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            if isMoE {
                return moeForward(x)
            } else {
                // Standard MLP
                guard let fc1 = fc1, let fc2 = fc2 else { return x }
                return fc2(gelu(fc1(x)))
            }
        }

        /// Full MoE forward pass with quantized experts
        /// Uses SwiGLU activation: fc1 outputs gate+up, then hidden = gelu(gate) * up
        /// Uses MLX's quantizedMatmul for efficient int4 computation
        private func moeForward(_ x: MLXArray) -> MLXArray {
            guard let router = router,
                  let fc1Q = fc1Q, let fc1Scales = fc1Scales, let fc1Biases = fc1Biases,
                  let fc2Q = fc2Q, let fc2Scales = fc2Scales, let fc2Biases = fc2Biases else {
                return x
            }

            let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
            let numTokens = B * L
            let xFlat = x.reshaped(numTokens, D)

            // Get router logits and compute expert selection
            let routerLogits = router(xFlat)  // [numTokens, numExperts]

            // Find top-k expert indices using argSort (ascending, take last k for highest)
            let sortedIndices = argSort(routerLogits, axis: -1)
            let topkIndices = sortedIndices[0..., (numExperts - expertsPerToken)...]  // [numTokens, k]

            // Gather top-k logits using takeAlong
            let topkLogits = takeAlong(routerLogits, topkIndices, axis: 1)  // [numTokens, k]
            let topkWeights = softmax(topkLogits, axis: -1)  // [numTokens, k]

            // Find unique selected experts to avoid processing all 64
            eval(topkIndices)  // Force evaluation to get values
            let indicesFlat = topkIndices.flattened().asArray(Int32.self)
            let selectedExperts = Set(indicesFlat.map { Int($0) })

            // Process only selected experts
            var output = MLXArray.zeros([numTokens, D])

            // Quantization parameters: group_size=64, bits=4 for md3p-int4
            let groupSize = 64
            let bits = 4

            for expertIdx in selectedExperts {
                // Create combined mask and weights for this expert across all k positions
                var expertWeights = MLXArray.zeros([numTokens, 1])

                for k in 0 ..< expertsPerToken {
                    let expertIdxForK = topkIndices[0..., k]
                    let weightForK = topkWeights[0..., k...(k + 1)]
                    let mask = (expertIdxForK .== expertIdx).asType(.float32).expandedDimensions(axis: -1)
                    expertWeights = expertWeights + mask * weightForK
                }

                // Skip if total weight is negligible
                eval(expertWeights)
                let totalWeight = MLX.sum(expertWeights).item(Float.self)
                if totalWeight < 1e-6 {
                    continue
                }

                // Get this expert's weights (index into first dimension)
                let fc1W = fc1Q[expertIdx]
                let fc1S = fc1Scales[expertIdx]
                let fc1B = fc1Biases[expertIdx]
                let fc2W = fc2Q[expertIdx]
                let fc2S = fc2Scales[expertIdx]
                let fc2B = fc2Biases[expertIdx]

                // fc1 outputs gate + up combined: [numTokens, 2*expertInnerDim]
                let fc1Out = quantizedMatmul(
                    xFlat, fc1W, scales: fc1S, biases: fc1B,
                    transpose: true, groupSize: groupSize, bits: bits
                )

                // SwiGLU: split into gate and up, apply gelu to gate, multiply
                // fc1Out shape: [numTokens, 2048] -> gate: [numTokens, 1024], up: [numTokens, 1024]
                let gatePlusUp = fc1Out.split(parts: 2, axis: -1)
                let gate = gatePlusUp[0]  // First half
                let up = gatePlusUp[1]    // Second half
                let hidden = gelu(gate) * up  // [numTokens, expertInnerDim=1024]

                // fc2: hidden @ fc2.T -> [numTokens, dim=2048]
                let expertOut = quantizedMatmul(
                    hidden, fc2W, scales: fc2S, biases: fc2B,
                    transpose: true, groupSize: groupSize, bits: bits
                )

                // Add weighted output
                output = output + expertOut * expertWeights
            }

            return output.reshaped(B, L, D)
        }
    }

    /// Language model with mixed MLP/MoE layers
    class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "post_ln") var postLn: LayerNorm

        // Use a standard layers array - MLX will handle indexing
        let layers: [TransformerBlock]

        let config: Moondream3Configuration.TextConfiguration
        var kvHeads: [Int]

        init(_ config: Moondream3Configuration.TextConfiguration) {
            self.config = config

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.dim
            )

            // Create all layers with their indices
            self.layers = (0 ..< config.nLayers).map { idx in
                TransformerBlock(config, layerIdx: idx)
            }

            self._postLn.wrappedValue = LayerNorm(dimensions: config.dim)
            self.kvHeads = (0 ..< config.nLayers).map { _ in config.nKvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbedding ?? embedTokens(inputs)
            let mask = createAttentionMask(h: h, cache: cache)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            h = postLn(h)
            return embedTokens.asLinear(h)
        }
    }
}

// MARK: - Main Model

/// Error types for Moondream3
public enum Moondream3Error: LocalizedError {
    case imageRequired
    case invalidImageDimensions

    public var errorDescription: String? {
        switch self {
        case .imageRequired:
            return "Image input is required for this operation"
        case .invalidImageDimensions:
            return "Invalid image dimensions"
        }
    }
}

/// Moondream3 Vision-Language Model
/// Architecture: vision encoder (with integrated projector) + language model
public class Moondream3: Module, LanguageModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision") private var visionEncoder: Vision.Encoder
    @ModuleInfo(key: "text") private var languageModel: Language.LanguageModel

    public let config: Moondream3Configuration

    /// Cached image features from prepare()
    private var cachedImageFeatures: MLXArray?

    public var vocabularySize: Int { config.text.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Moondream3Configuration) {
        self.config = config

        self._visionEncoder.wrappedValue = Vision.Encoder(config.vision)
        self._languageModel.wrappedValue = Language.LanguageModel(config.text)
    }

    /// Encode image to features (projector is now part of vision encoder)
    public func encodeImage(_ pixels: MLXArray) -> MLXArray {
        // Pixels should be [B, H, W, C] format
        // Vision encoder now includes projection to text dimension
        return visionEncoder(pixels)
    }

    // MARK: - LanguageModel Protocol

    /// Prepare the model with image and text input
    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        let inputIds = input.text.tokens

        // Process image if present
        if let image = input.image {
            // Encode image through vision encoder and projector
            // Image should be [B, H, W, C] format after preprocessing
            let imageFeatures = encodeImage(image.pixels)
            cachedImageFeatures = imageFeatures

            // Get text embeddings
            let textEmbeddings = languageModel.embedTokens(inputIds)

            // Prepend image features to text embeddings
            let combined = concatenated([imageFeatures, textEmbeddings], axis: 1)

            // Run through language model with combined embeddings
            let logits = languageModel(inputIds, cache: cache, inputEmbedding: combined)

            return .logits(LMOutput(logits: logits))
        } else {
            // Text-only mode
            let logits = languageModel(inputIds, cache: cache)
            return .logits(LMOutput(logits: logits))
        }
    }

    /// Generate next token logits
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// Sanitize loaded weights - rename keys to match Swift model structure
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            var newKey = key

            // Skip position IDs and rotary embeddings
            if key.contains("position_id") || key.contains("rotary_emb") {
                continue
            }

            // Skip region module (not implemented)
            if key.hasPrefix("region.") {
                continue
            }

            // Rename weight keys to match Swift model structure
            // text.blocks.N -> text.layers.N (transformer layers)
            newKey = newKey.replacingOccurrences(of: "text.blocks.", with: "text.layers.")
            // vision.blocks.N -> vision.layers.N (vision transformer layers)
            newKey = newKey.replacingOccurrences(of: "vision.blocks.", with: "vision.layers.")
            // text.wte -> text.embed_tokens (word embeddings)
            newKey = newKey.replacingOccurrences(of: "text.wte", with: "text.embed_tokens")

            // Skip lm_head - using tied embeddings via embedTokens.asLinear()
            if newKey.contains("lm_head") {
                continue
            }

            // Handle conv2d weight transposition (PyTorch -> MLX format)
            // patch_emb weight: [out_channels, in_channels, kH, kW] -> [out, kH, kW, in]
            if newKey.contains("patch_emb") && newKey.contains("weight") && value.ndim == 4 {
                sanitized[newKey] = value.transposed(0, 2, 3, 1)
                continue
            }

            sanitized[newKey] = value
        }

        return sanitized
    }
}

// MARK: - Processor Configuration

/// Configuration for Moondream3 input processing
public struct Moondream3ProcessorConfiguration: Codable, Sendable {
    public let cropSize: Int
    public let imageMean: [Float]
    public let imageStd: [Float]

    public init() {
        // Default values for Moondream3
        self.cropSize = 378
        self.imageMean = [0.5, 0.5, 0.5]
        self.imageStd = [0.5, 0.5, 0.5]
    }

    public init(cropSize: Int, imageMean: [Float], imageStd: [Float]) {
        self.cropSize = cropSize
        self.imageMean = imageMean
        self.imageStd = imageStd
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (CGFloat(imageMean[0]), CGFloat(imageMean[1]), CGFloat(imageMean[2]))
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (CGFloat(imageStd[0]), CGFloat(imageStd[1]), CGFloat(imageStd[2]))
    }

    public var size: CGSize {
        CGSize(width: cropSize, height: cropSize)
    }
}

// MARK: - Processor

/// Processor for Moondream3 input preparation
public class Moondream3Processor: UserInputProcessor {
    private let config: Moondream3ProcessorConfiguration
    private let tokenizer: any Tokenizer

    /// Number of image tokens (patches in vision encoder)
    /// For 378x378 image with 14x14 patches: (378/14)^2 = 729 patches
    private let imageSequenceLength: Int

    // Moondream3 token templates (from config.py)
    private struct Templates {
        // Caption templates - complete sequences
        static let captionShort: [Int] = [1, 32708, 2, 12492, 3]
        static let captionNormal: [Int] = [1, 32708, 2, 6382, 3]
        static let captionLong: [Int] = [1, 32708, 2, 4059, 3]

        // Query template - prefix + question tokens + suffix
        static let queryPrefix: [Int] = [1, 15381, 2]
        static let querySuffix: [Int] = [3]

        // Detect template
        static let detectPrefix: [Int] = [1, 7235, 476, 2]
        static let detectSuffix: [Int] = [3]

        // Point template
        static let pointPrefix: [Int] = [1, 2581, 2]
        static let pointSuffix: [Int] = [3]
    }

    public init(config: Moondream3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer

        // Calculate image sequence length from config
        let patchSize = 14  // Moondream3 uses 14x14 patches
        let gridSize = config.cropSize / patchSize
        self.imageSequenceLength = gridSize * gridSize
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Get the prompt text
        let promptText: String
        switch input.prompt {
        case .text(let text):
            promptText = text
        case .chat(let messages):
            promptText = messages.map { message in
                switch message.role {
                case .user:
                    return message.content
                case .assistant:
                    return message.content
                case .system:
                    return message.content
                case .tool:
                    return message.content
                }
            }.joined(separator: "\n")
        case .messages(let messages):
            promptText = messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
        }

        // Process image if present
        var processedImage: LMInput.ProcessedImage?
        if let firstImage = input.images.first {
            let pixels = try await processImage(firstImage, processing: input.processing)
            processedImage = LMInput.ProcessedImage(pixels: pixels)
        }

        // Build token sequence using Moondream3 templates
        let tokens = buildTokenSequence(from: promptText)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)

        return LMInput(
            text: LMInput.Text(tokens: tokenArray),
            image: processedImage
        )
    }

    /// Build token sequence using Moondream3 templates
    private func buildTokenSequence(from prompt: String) -> [Int] {
        // Check for special markers and use appropriate templates
        if prompt.hasPrefix("<caption:") {
            if prompt.contains("short") {
                return Templates.captionShort
            } else if prompt.contains("long") {
                return Templates.captionLong
            } else {
                return Templates.captionNormal
            }
        } else if prompt.hasPrefix("<point>") {
            let object = String(prompt.dropFirst(7))  // Remove "<point>"
            let objectTokens = tokenizer.encode(text: object)
            return Templates.pointPrefix + objectTokens + Templates.pointSuffix
        } else if prompt.hasPrefix("<detect>") {
            let object = String(prompt.dropFirst(8))  // Remove "<detect>"
            let objectTokens = tokenizer.encode(text: object)
            return Templates.detectPrefix + objectTokens + Templates.detectSuffix
        } else {
            // Default: treat as query
            let questionTokens = tokenizer.encode(text: prompt)
            return Templates.queryPrefix + questionTokens + Templates.querySuffix
        }
    }

    private func processImage(_ image: UserInput.Image, processing: UserInput.Processing) async throws -> MLXArray {
        // Get CIImage from the input
        var ciImage = try image.asCIImage()

        // Convert to sRGB color space for consistent processing
        ciImage = MediaProcessing.inSRGBToneCurveSpace(ciImage)

        // Apply any user-specified processing (like resize)
        if let resize = processing.resize {
            ciImage = MediaProcessing.resampleBicubic(ciImage, to: resize)
        }

        // Resample to the model's expected size
        ciImage = MediaProcessing.resampleBicubic(ciImage, to: config.size)

        // Normalize with mean and std
        ciImage = MediaProcessing.normalize(
            ciImage,
            mean: config.imageMeanTuple,
            std: config.imageStdTuple
        )

        // Convert to MLXArray [B, H, W, C]
        var pixels = MediaProcessing.asMLXArray(ciImage)

        // Add batch dimension if needed
        if pixels.ndim == 3 {
            pixels = pixels.expandedDimensions(axis: 0)
        }

        return pixels
    }
}
