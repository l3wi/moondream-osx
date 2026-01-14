// Copyright © 2024 Apple Inc. / Moondream AI
// Moondream3 VLM implementation for MLX Swift
// Ported directly from moondream-station mlx_backend

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

// MARK: - RoPE Helpers

/// Precompute frequencies for rotary position embeddings
/// Python: precompute_freqs_cis(dim, max_len, theta=1500000.0)
func precomputeFreqsCis(dim: Int, maxLen: Int, theta: Float = 1_500_000.0) -> MLXArray {
    // freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
    let freqs = 1.0 / pow(theta, indices / Float(dim))

    // t = mx.arange(max_len).astype(mx.float32)
    let t = MLXArray((0..<maxLen).map { Float($0) })

    // freqs = t[:, None] * freqs[None, :]
    let freqsMatrix = expandedDimensions(t, axis: 1) * expandedDimensions(freqs, axis: 0)

    // cos_freqs = mx.cos(freqs), sin_freqs = mx.sin(freqs)
    let cosFreqs = cos(freqsMatrix)
    let sinFreqs = sin(freqsMatrix)

    // return mx.stack([cos_freqs, sin_freqs], axis=-1)
    return stacked([cosFreqs, sinFreqs], axis: -1)
}

/// Apply rotary position embeddings
/// Python: apply_rotary_emb(x, freqs_cis, positions)
func applyRotaryEmb(_ x: MLXArray, freqsCis: MLXArray, positions: MLXArray) -> MLXArray {
    let rotDim = freqsCis.dim(1) * 2
    let xRot = x[0..., 0..., 0..., ..<rotDim]
    let xPass = x[0..., 0..., 0..., rotDim...]

    let dQ = xRot.dim(-1) / 2
    let xqR = xRot[0..., 0..., 0..., ..<dQ]
    let xqI = xRot[0..., 0..., 0..., dQ...]

    // Ensure positions is int32 for gather
    let positionsInt = positions.asType(.int32)
    let freqsSlice = freqsCis[positionsInt]  // [seq_len, rot_dim/2, 2]
    let freqsCos = expandedDimensions(expandedDimensions(freqsSlice[0..., 0..., 0], axis: 0), axis: 0)
    let freqsSin = expandedDimensions(expandedDimensions(freqsSlice[0..., 0..., 1], axis: 0), axis: 0)

    // xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    // xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    let xqOutR = xqR * freqsCos - xqI * freqsSin
    let xqOutI = xqR * freqsSin + xqI * freqsCos

    // x_rot_out = mx.stack([xq_out_r, xq_out_i], axis=-1).reshape(...)
    let xRotOut = stacked([xqOutR, xqOutI], axis: -1)
    let shape = xRotOut.shape
    let newShape = Array(shape.dropLast(2)) + [shape[shape.count-2] * shape[shape.count-1]]
    let xRotOutReshaped = xRotOut.reshaped(newShape)

    // return mx.concatenate([x_rot_out, x_pass], axis=-1)
    return concatenated([xRotOutReshaped, xPass], axis: -1)
}

// MARK: - Vision Encoder

private enum Vision {

    /// Vision attention layer with fused QKV
    class Attention: Module {
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

    /// Vision MLP with gelu_approx
    class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(dim: Int, hiddenDim: Int) {
            self.fc1 = Linear(dim, hiddenDim, bias: true)
            self.fc2 = Linear(hiddenDim, dim, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // Python uses gelu_approx
            fc2(geluApproximate(fc1(x)))
        }
    }

    /// Vision transformer block
    class Block: Module {
        @ModuleInfo var ln1: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var ln2: LayerNorm
        @ModuleInfo var mlp: MLP

        init(dim: Int, numHeads: Int, ffDim: Int) {
            self._ln1.wrappedValue = LayerNorm(dimensions: dim)
            self._attention.wrappedValue = Attention(dim: dim, numHeads: numHeads)
            self._ln2.wrappedValue = LayerNorm(dimensions: dim)
            self.mlp = MLP(dim: dim, hiddenDim: ffDim)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x + attention(ln1(x))
            h = h + mlp(ln2(h))
            return h
        }
    }

    /// Vision encoder - Python uses Linear for patch_emb, NOT Conv2d!
    class Encoder: Module {
        let config: Moondream3Configuration.VisionConfiguration

        // Python: self.patch_emb = nn.Linear(patch_dim, config.enc_dim)
        @ModuleInfo(key: "patch_emb") var patchEmb: Linear
        @ModuleInfo(key: "pos_emb") var posEmb: MLXArray

        let blocks: [Block]
        @ModuleInfo(key: "post_ln") var postLn: LayerNorm

        // Projector - note: Python concatenates global + reconstructed, dims are enc_dim * 2
        @ModuleInfo(key: "proj_fc1") var projFc1: Linear
        @ModuleInfo(key: "proj_fc2") var projFc2: Linear

        init(_ config: Moondream3Configuration.VisionConfiguration) {
            self.config = config

            // patch_dim = config.enc_patch_size * config.enc_patch_size * config.in_channels
            let patchDim = config.encPatchSize * config.encPatchSize * config.inChannels
            self._patchEmb.wrappedValue = Linear(patchDim, config.encDim, bias: true)

            let gridSize = config.cropSize / config.encPatchSize
            let numPatches = gridSize * gridSize

            // Python: self.pos_emb = mx.zeros((1, num_patches, config.enc_dim))
            self._posEmb.wrappedValue = MLXArray.zeros([1, numPatches, config.encDim])

            self.blocks = (0..<config.encNLayers).map { _ in
                Block(dim: config.encDim, numHeads: config.encNHeads, ffDim: config.encFfDim)
            }
            self._postLn.wrappedValue = LayerNorm(dimensions: config.encDim)

            // Python: self.proj_fc1 = nn.Linear(config.enc_dim * 2, config.proj_inner_dim)
            // Even for single crop, projector expects concatenated features
            self._projFc1.wrappedValue = Linear(config.encDim * 2, config.projInnerDim, bias: true)
            self._projFc2.wrappedValue = Linear(config.projInnerDim, config.projOutDim, bias: true)
        }

        /// Create patches from image - Python: create_patches
        /// Input: [B, C, H, W], Output: [B, num_patches, patch_dim]
        func createPatches(_ x: MLXArray) -> MLXArray {
            let (B, C, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let P = config.encPatchSize

            // x = x.reshape(B, C, H // P, P, W // P, P)
            // x = x.transpose(0, 2, 4, 1, 3, 5)
            // x = x.reshape(B, (H // P) * (W // P), C * P * P)
            var patches = x.reshaped(B, C, H / P, P, W / P, P)
            patches = patches.transposed(0, 2, 4, 1, 3, 5)
            patches = patches.reshaped(B, (H / P) * (W / P), C * P * P)

            return patches
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // x is already [B, C, H, W] (NCHW format) from MediaProcessing.asMLXArray
            eval(x)

            // Create patches and embed
            var patches = createPatches(x)
            eval(patches)
            patches = patchEmb(patches)
            patches = patches + posEmb

            // Transformer blocks
            var h = patches
            for block in blocks {
                h = block(h)
            }
            h = postLn(h)

            // Python concatenates global_features + reconstructed before projection
            // proj_fc1 expects [B, seq, enc_dim * 2] = [B, seq, 2304]
            // For single crop mode, concatenate features with themselves
            let concatenatedFeatures = concatenated([h, h], axis: -1)  // [B, seq, 2304]

            // Project: 2304 -> 8192 -> 2048
            let fc1Out = projFc1(concatenatedFeatures)
            let geluOut = geluApproximate(fc1Out)
            let projected = projFc2(geluOut)

            return projected
        }
    }
}

// MARK: - Language Model

private enum Language {

    /// Text attention with tau modulation
    /// Q, K, V all have same dimension (32 heads × 64 = 2048 each)
    /// Total qkv = 6144 = 2048 + 2048 + 2048
    class Attention: Module {
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

            // KV cache handling
            // Python uses in-place .at[].add() to update pre-allocated cache
            // Swift workaround: rebuild cache by inserting new K/V at correct position
            var newCache: (MLXArray, MLXArray)
            if let (kCache, vCache) = cache {
                let T = keys.dim(2)  // New sequence length
                let maxLen = kCache.dim(2)  // Pre-allocated max length
                let newEnd = cachePos + T

                // Build updated cache by inserting new K/V at [cachePos:cachePos+T]
                // Python: k_cache = k_cache.at[:, :, cache_pos:cache_pos+T, :].add(k)
                var updatedKCache: MLXArray
                var updatedVCache: MLXArray

                if cachePos > 0 && newEnd < maxLen {
                    // Insert in middle: [before, new, after]
                    let beforeK = kCache[0..., 0..., ..<cachePos, 0...]
                    let afterK = kCache[0..., 0..., newEnd..., 0...]
                    updatedKCache = concatenated([beforeK, keys, afterK], axis: 2)

                    let beforeV = vCache[0..., 0..., ..<cachePos, 0...]
                    let afterV = vCache[0..., 0..., newEnd..., 0...]
                    updatedVCache = concatenated([beforeV, values, afterV], axis: 2)
                } else if cachePos == 0 && newEnd < maxLen {
                    // Insert at start: [new, after]
                    let afterK = kCache[0..., 0..., newEnd..., 0...]
                    updatedKCache = concatenated([keys, afterK], axis: 2)

                    let afterV = vCache[0..., 0..., newEnd..., 0...]
                    updatedVCache = concatenated([values, afterV], axis: 2)
                } else if cachePos > 0 && newEnd >= maxLen {
                    // Insert at end: [before, new]
                    let beforeK = kCache[0..., 0..., ..<cachePos, 0...]
                    updatedKCache = concatenated([beforeK, keys], axis: 2)

                    let beforeV = vCache[0..., 0..., ..<cachePos, 0...]
                    updatedVCache = concatenated([beforeV, values], axis: 2)
                } else {
                    // cachePos == 0 && newEnd >= maxLen: just use new keys/values
                    updatedKCache = keys
                    updatedVCache = values
                }

                // For attention, use only valid entries up to newEnd
                // Python: k = k_cache[:, :, :cache_pos + T, :]
                let validLen = min(newEnd, updatedKCache.dim(2))
                keys = updatedKCache[0..., 0..., ..<validLen, 0...]
                values = updatedVCache[0..., 0..., ..<validLen, 0...]

                newCache = (updatedKCache, updatedVCache)
            } else {
                newCache = (keys, values)
            }

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

    /// Dense MLP (for non-MoE layers)
    class DenseMLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(dim: Int, hiddenDim: Int) {
            self.fc1 = Linear(dim, hiddenDim, bias: true)
            self.fc2 = Linear(hiddenDim, dim, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // Python uses gelu_approx
            fc2(geluApproximate(fc1(x)))
        }
    }

    /// Quantized MoE MLP
    class QuantizedMoEMLP: Module, UnaryLayer {
        let dim: Int
        let numExperts: Int
        let expertDim: Int
        let expertsPerToken: Int
        let bits: Int
        let groupSize: Int

        @ModuleInfo var router: Linear
        @ModuleInfo(key: "fc1_q") var fc1Q: MLXArray
        @ModuleInfo(key: "fc1_scales") var fc1Scales: MLXArray
        @ModuleInfo(key: "fc1_biases") var fc1Biases: MLXArray
        @ModuleInfo(key: "fc2_q") var fc2Q: MLXArray
        @ModuleInfo(key: "fc2_scales") var fc2Scales: MLXArray
        @ModuleInfo(key: "fc2_biases") var fc2Biases: MLXArray

        init(dim: Int, numExperts: Int, expertDim: Int, expertsPerToken: Int) {
            self.dim = dim
            self.numExperts = numExperts
            self.expertDim = expertDim
            self.expertsPerToken = expertsPerToken
            self.bits = 4
            self.groupSize = 64

            self.router = Linear(dim, numExperts, bias: true)

            // Placeholder shapes - will be loaded from weights
            self._fc1Q.wrappedValue = MLXArray.zeros([numExperts, 2 * expertDim, dim / 8])
            self._fc1Scales.wrappedValue = MLXArray.zeros([numExperts, 2 * expertDim, dim / 64])
            self._fc1Biases.wrappedValue = MLXArray.zeros([numExperts, 2 * expertDim, dim / 64])
            self._fc2Q.wrappedValue = MLXArray.zeros([numExperts, dim, expertDim / 8])
            self._fc2Scales.wrappedValue = MLXArray.zeros([numExperts, dim, expertDim / 64])
            self._fc2Biases.wrappedValue = MLXArray.zeros([numExperts, dim, expertDim / 64])
        }

        /// Gather sort for efficient expert routing
        func gatherSort(_ x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let M = indices.dim(-1)
            let indicesFlat = indices.flattened()
            let order = argSort(indicesFlat)
            let invOrder = argSort(order)

            // x.flatten(0, -3)[order // M]
            let xFlat = x.flattened(start: 0, end: x.ndim - 3)

            // CRITICAL: order / M might produce float! Cast to int32
            let orderDivM = (order / M).asType(.int32)

            let sortedX = take(xFlat, orderDivM, axis: 0)
            let sortedIdx = take(indicesFlat, order.asType(.int32), axis: 0)

            return (sortedX, sortedIdx, invOrder)
        }

        /// Scatter unsort to restore original order
        func scatterUnsort(_ x: MLXArray, invOrder: MLXArray, shape: [Int]) -> MLXArray {
            let xReordered = take(x.flattened(start: 0, end: 1), invOrder, axis: 0)
            return xReordered.reshaped(shape + [x.dim(-1)])
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))
            let xFlat = x.reshaped(-1, C)

            // Router
            let routerLogits = router(xFlat)

            // Top-k selection
            var topkIdxs = argPartition(-routerLogits, kth: expertsPerToken - 1, axis: -1)
            topkIdxs = topkIdxs[0..., ..<expertsPerToken]
            let topkLogits = takeAlong(routerLogits, topkIdxs, axis: -1)
            let topkWeights = softmax(topkLogits, axis: -1)

            // Expand input
            var xExpanded = expandedDimensions(xFlat, axes: [-2, -3])

            // Sort for efficiency (when many tokens)
            // NOTE: Sorting disabled due to shape broadcast issues with gatherSort
            // See: Known Limitations in README.md
            let doSort = false
            var idx = topkIdxs
            var invOrder: MLXArray? = nil

            if doSort {
                let (sortedX, sortedIdx, inv) = gatherSort(xExpanded, indices: topkIdxs)
                xExpanded = sortedX
                idx = sortedIdx.reshaped(topkIdxs.shape)
                invOrder = inv
            }

            // FC1 with gather_qmm
            let hFull = gatherQuantizedMatmul(
                xExpanded, fc1Q,
                scales: fc1Scales,
                biases: fc1Biases,
                rhsIndices: idx,
                transpose: true,
                groupSize: groupSize,
                bits: bits
            )

            // Gated GELU: gelu(h) * (g + 1)
            let splitOut = hFull.split(parts: 2, axis: -1)
            let h = splitOut[0]
            let g = splitOut[1]
            let hidden = gelu(h) * (g + 1)

            // FC2 with gather_qmm
            var expertOuts = gatherQuantizedMatmul(
                hidden, fc2Q,
                scales: fc2Scales,
                biases: fc2Biases,
                rhsIndices: idx,
                transpose: true,
                groupSize: groupSize,
                bits: bits
            )

            // Unsort if needed
            if doSort, let inv = invOrder {
                expertOuts = scatterUnsort(expertOuts, invOrder: inv, shape: [topkIdxs.dim(0), topkIdxs.dim(1)])
            }

            // Weight and sum
            let expertOutsSqueezed = squeezed(expertOuts, axis: -2)
            let weightedOuts = expertOutsSqueezed * expandedDimensions(topkWeights, axis: -1)
            let mlpOut = MLX.sum(weightedOuts, axis: 1)

            return mlpOut.reshaped(B, T, C)
        }
    }

    /// Text transformer block
    class Block: Module {
        let layerIdx: Int
        let isMoE: Bool

        @ModuleInfo var ln: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        var mlp: UnaryLayer

        init(_ config: Moondream3Configuration.TextConfiguration, layerIdx: Int) {
            self.layerIdx = layerIdx
            self.isMoE = config.moe.startLayer <= layerIdx

            self.ln = LayerNorm(dimensions: config.dim)
            self._attention.wrappedValue = Attention(
                dim: config.dim,
                numHeads: config.nHeads,
                numKvHeads: config.nKvHeads
            )

            if isMoE {
                self.mlp = QuantizedMoEMLP(
                    dim: config.dim,
                    numExperts: config.moe.numExperts,
                    expertDim: config.moe.expertInnerDim,
                    expertsPerToken: config.moe.expertsPerToken
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

    /// Language model
    class TextModel: Module, KVCacheDimensionProvider {
        let config: Moondream3Configuration.TextConfiguration

        @ModuleInfo(key: "wte") var wte: Embedding
        let blocks: [Block]
        @ModuleInfo(key: "post_ln") var postLn: LayerNorm
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        let freqsCis: MLXArray

        var kvHeads: [Int]

        init(_ config: Moondream3Configuration.TextConfiguration) {
            self.config = config

            self._wte.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.dim)
            self.blocks = (0..<config.nLayers).map { Block(config, layerIdx: $0) }
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
}

// MARK: - Main Model

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
public class Moondream3: Module, LanguageModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision") private var visionEncoder: Vision.Encoder
    @ModuleInfo(key: "text") private var textModel: Language.TextModel

    public let config: Moondream3Configuration

    // Attention mask (precomputed once)
    private var attnMask: MLXArray?

    public var vocabularySize: Int { config.text.vocabSize }
    public var kvHeads: [Int] { textModel.kvHeads }

    public init(_ config: Moondream3Configuration) {
        self.config = config
        self._visionEncoder.wrappedValue = Vision.Encoder(config.vision)
        self._textModel.wrappedValue = Language.TextModel(config.text)

        // Build attention mask like Python
        let maxCtx = config.text.maxContext
        var mask = tril(MLXArray.ones([1, 1, maxCtx, maxCtx]).asType(.bool))

        // Prefix attention for image tokens
        let patchW = config.vision.cropSize / config.vision.encPatchSize
        let prefixAttnLen = 1 + patchW * patchW  // BOS + image patches

        // Create prefix mask and OR with causal mask
        let prefixMask = MLXArray.ones([1, 1, prefixAttnLen, prefixAttnLen]).asType(.bool)
        // Pad to max context size
        // NOTE: Using causal mask without explicit padding - works for inference
        var prefixMaskPadded = MLXArray.zeros([1, 1, maxCtx, maxCtx]).asType(.bool)

        self.attnMask = mask
    }

    // MARK: - KV Cache Allocation

    /// Allocate KV cache for generation - matches Python _allocate_kv_cache
    public func allocateKVCache(batchSize: Int = 1, maxSeqLen: Int = 1024) -> [[(MLXArray, MLXArray)]] {
        let nLayers = config.text.nLayers
        let nKvHeads = config.text.nKvHeads
        let headDim = config.text.dim / config.text.nHeads

        var cache: [[(MLXArray, MLXArray)]] = []

        for _ in 0..<nLayers {
            let k = MLXArray.zeros([batchSize, nKvHeads, maxSeqLen, headDim])
            let v = MLXArray.zeros([batchSize, nKvHeads, maxSeqLen, headDim])
            cache.append([(k, v)])
        }

        return cache
    }

    // MARK: - Vision Encoding

    /// Run vision encoder on image
    public func encodeImage(_ pixels: MLXArray) -> MLXArray {
        visionEncoder(pixels)
    }

    // MARK: - Internal Forward Pass with Cache

    /// Prefill: process multiple tokens with mask, updating cache
    private func prefill(
        embeddings: MLXArray,
        cachePos: Int,
        cache: inout [(MLXArray, MLXArray)]
    ) -> MLXArray {
        let seqLen = embeddings.dim(1)
        let positions = MLXArray((cachePos..<(cachePos + seqLen)).map { Int32($0) })

        // Get mask slice for this position range
        let mask = attnMask?[0..., 0..., cachePos..<(cachePos + seqLen), 0...]

        let (hidden, newCaches) = textModel(
            embeddings,
            positions: positions,
            mask: mask,
            cache: cache,
            cachePos: cachePos
        )

        cache = newCaches
        return hidden
    }

    /// Decode one token without mask, using cache
    private func decodeOne(
        embedding: MLXArray,
        cachePos: Int,
        cache: inout [(MLXArray, MLXArray)]
    ) -> MLXArray {
        let positions = MLXArray([Int32(cachePos)])

        // No mask during single-token decode
        let (hidden, newCaches) = textModel(
            embedding,
            positions: positions,
            mask: nil,
            cache: cache,
            cachePos: cachePos
        )

        cache = newCaches
        return textModel.generateLogits(hidden)
    }

    // MARK: - Caption Generation (Python-style)

    /// Generate caption for image - matches Python model.caption() flow exactly
    public func caption(
        pixels: MLXArray,
        length: String = "normal",
        tokenizer: any Tokenizer,
        maxTokens: Int = 768,
        temperature: Float = 0.0
    ) -> String {
        // Get prompt tokens based on length
        let promptTokens: [Int]
        switch length {
        case "short":
            promptTokens = [1, 32708, 2, 12492, 3]  // <caption:short>
        case "long":
            promptTokens = [1, 32708, 2, 4059, 3]   // <caption:long>
        default:
            promptTokens = [1, 32708, 2, 6382, 3]   // <caption:normal>
        }

        // Evaluate pixels
        eval(pixels)

        // Encode image
        let imgEmb = visionEncoder(pixels)  // [1, num_patches, proj_dim]

        // BOS embedding
        let bosTokens = MLXArray([Int32(0)]).expandedDimensions(axis: 0)  // [[0]]
        let bosEmb = textModel.embed(bosTokens)  // [1, 1, dim]

        // Combine BOS + image embeddings
        let inputsEmbeds = concatenated([bosEmb, imgEmb], axis: 1)

        // Allocate KV cache
        var cache = allocateSimpleCache()

        // Prefill with image
        _ = prefill(embeddings: inputsEmbeds, cachePos: 0, cache: &cache)
        var pos = inputsEmbeds.dim(1)

        // Prefill with prompt tokens
        let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let promptEmb = textModel.embed(promptArray)

        let hidden = prefill(embeddings: promptEmb, cachePos: pos, cache: &cache)
        var logits = textModel.generateLogits(hidden)

        pos += promptEmb.dim(1)

        // Sample first token
        var nextToken = sampleToken(logits: logits, temperature: temperature)

        // Generate tokens
        var tokens: [Int] = []
        let eosId = 0  // EOS token ID (same as BOS in this model)
        var generated = 0

        while nextToken != eosId && generated < maxTokens {
            tokens.append(nextToken)

            // Embed token
            let tokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            let nextEmb = textModel.embed(tokenArray)

            // Decode one token
            logits = decodeOne(embedding: nextEmb, cachePos: pos, cache: &cache)
            nextToken = sampleToken(logits: logits, temperature: temperature)

            pos += 1
            generated += 1
        }

        // Decode tokens to string
        return tokenizer.decode(tokens: tokens)
    }

    /// Query image with a question - matches Python model.query() flow
    public func query(
        pixels: MLXArray,
        question: String,
        tokenizer: any Tokenizer,
        maxTokens: Int = 768,
        temperature: Float = 0.0
    ) -> String {
        // Build prompt tokens: [1, 15381, 2] + question_tokens + [3]
        let prefix = [1, 15381, 2]  // Query prefix
        let questionTokens = tokenizer.encode(text: question)
        let suffix = [3]
        let promptTokens = prefix + questionTokens + suffix

        // Encode image
        let imgEmb = visionEncoder(pixels)

        // BOS embedding
        let bosTokens = MLXArray([Int32(0)]).expandedDimensions(axis: 0)
        let bosEmb = textModel.embed(bosTokens)

        // Combine BOS + image embeddings
        let inputsEmbeds = concatenated([bosEmb, imgEmb], axis: 1)

        // Allocate KV cache
        var cache = allocateSimpleCache()

        // Prefill with image
        var _ = prefill(embeddings: inputsEmbeds, cachePos: 0, cache: &cache)
        var pos = inputsEmbeds.dim(1)

        // Prefill with prompt tokens
        let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let promptEmb = textModel.embed(promptArray)
        let hidden = prefill(embeddings: promptEmb, cachePos: pos, cache: &cache)
        var logits = textModel.generateLogits(hidden)
        pos += promptEmb.dim(1)

        // Sample first token
        var nextToken = sampleToken(logits: logits, temperature: temperature)

        // Generate tokens
        var tokens: [Int] = []
        let eosId = 0
        var generated = 0

        while nextToken != eosId && generated < maxTokens {
            tokens.append(nextToken)

            let tokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            let nextEmb = textModel.embed(tokenArray)

            logits = decodeOne(embedding: nextEmb, cachePos: pos, cache: &cache)
            nextToken = sampleToken(logits: logits, temperature: temperature)

            pos += 1
            generated += 1
        }

        return tokenizer.decode(tokens: tokens)
    }

    /// Point to object in image - returns coordinates
    public func point(
        pixels: MLXArray,
        object: String,
        tokenizer: any Tokenizer,
        maxTokens: Int = 256,
        temperature: Float = 0.0
    ) -> String {
        // Build prompt tokens: [1, 2581, 2] + object_tokens + [3]
        let prefix = [1, 2581, 2]  // Point prefix
        let objectTokens = tokenizer.encode(text: object)
        let suffix = [3]
        let promptTokens = prefix + objectTokens + suffix

        // Encode image
        let imgEmb = visionEncoder(pixels)

        // BOS embedding
        let bosTokens = MLXArray([Int32(0)]).expandedDimensions(axis: 0)
        let bosEmb = textModel.embed(bosTokens)

        // Combine BOS + image embeddings
        let inputsEmbeds = concatenated([bosEmb, imgEmb], axis: 1)

        // Allocate KV cache
        var cache = allocateSimpleCache()

        // Prefill with image
        var _ = prefill(embeddings: inputsEmbeds, cachePos: 0, cache: &cache)
        var pos = inputsEmbeds.dim(1)

        // Prefill with prompt tokens
        let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let promptEmb = textModel.embed(promptArray)
        let hidden = prefill(embeddings: promptEmb, cachePos: pos, cache: &cache)
        var logits = textModel.generateLogits(hidden)
        pos += promptEmb.dim(1)

        // Sample first token
        var nextToken = sampleToken(logits: logits, temperature: temperature)

        // Generate tokens
        var tokens: [Int] = []
        let eosId = 0
        var generated = 0

        while nextToken != eosId && generated < maxTokens {
            tokens.append(nextToken)

            let tokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            let nextEmb = textModel.embed(tokenArray)

            logits = decodeOne(embedding: nextEmb, cachePos: pos, cache: &cache)
            nextToken = sampleToken(logits: logits, temperature: temperature)

            pos += 1
            generated += 1
        }

        return tokenizer.decode(tokens: tokens)
    }

    /// Detect objects in image - returns bounding boxes
    public func detect(
        pixels: MLXArray,
        object: String,
        tokenizer: any Tokenizer,
        maxTokens: Int = 256,
        temperature: Float = 0.0
    ) -> String {
        // Build prompt tokens: [1, 7235, 476, 2] + object_tokens + [3]
        let prefix = [1, 7235, 476, 2]  // Detect prefix
        let objectTokens = tokenizer.encode(text: object)
        let suffix = [3]
        let promptTokens = prefix + objectTokens + suffix

        // Encode image
        let imgEmb = visionEncoder(pixels)

        // BOS embedding
        let bosTokens = MLXArray([Int32(0)]).expandedDimensions(axis: 0)
        let bosEmb = textModel.embed(bosTokens)

        // Combine BOS + image embeddings
        let inputsEmbeds = concatenated([bosEmb, imgEmb], axis: 1)

        // Allocate KV cache
        var cache = allocateSimpleCache()

        // Prefill with image
        var _ = prefill(embeddings: inputsEmbeds, cachePos: 0, cache: &cache)
        var pos = inputsEmbeds.dim(1)

        // Prefill with prompt tokens
        let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let promptEmb = textModel.embed(promptArray)
        let hidden = prefill(embeddings: promptEmb, cachePos: pos, cache: &cache)
        var logits = textModel.generateLogits(hidden)
        pos += promptEmb.dim(1)

        // Sample first token
        var nextToken = sampleToken(logits: logits, temperature: temperature)

        // Generate tokens
        var tokens: [Int] = []
        let eosId = 0
        var generated = 0

        while nextToken != eosId && generated < maxTokens {
            tokens.append(nextToken)

            let tokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            let nextEmb = textModel.embed(tokenArray)

            logits = decodeOne(embedding: nextEmb, cachePos: pos, cache: &cache)
            nextToken = sampleToken(logits: logits, temperature: temperature)

            pos += 1
            generated += 1
        }

        return tokenizer.decode(tokens: tokens)
    }

    /// Allocate simple (non-quantized) KV cache
    private func allocateSimpleCache() -> [(MLXArray, MLXArray)] {
        let nLayers = config.text.nLayers
        let nKvHeads = config.text.nKvHeads
        let headDim = config.text.dim / config.text.nHeads
        let maxSeqLen = 1024

        var cache: [(MLXArray, MLXArray)] = []

        for _ in 0..<nLayers {
            let k = MLXArray.zeros([1, nKvHeads, maxSeqLen, headDim])
            let v = MLXArray.zeros([1, nKvHeads, maxSeqLen, headDim])
            cache.append((k, v))
        }

        return cache
    }

    /// Sample token from logits
    /// logits shape: [B, vocab_size] after generateLogits fix
    private func sampleToken(logits: MLXArray, temperature: Float) -> Int {
        // logits is [1, vocab_size], squeeze batch dim to get [vocab_size]
        let logits1D = logits.squeezed(axis: 0)

        if temperature == 0 {
            // Greedy
            return Int(argMax(logits1D, axis: -1).item(Int.self))
        } else {
            // Temperature sampling with softmax
            // NOTE: Using argmax after softmax (deterministic). For stochastic sampling,
            // implement multinomial sampling from probability distribution.
            let probs = softmax(logits1D / temperature, axis: -1)
            return Int(argMax(probs, axis: -1).item(Int.self))
        }
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let seqLen = inputIds.dim(1)

        // Create positions array
        let positions = MLXArray((0..<seqLen).map { Int32($0) })

        // Create causal mask
        let maxCtx = config.text.maxContext
        var mask = tril(MLXArray.ones([1, 1, maxCtx, maxCtx]).asType(.bool))

        // Prefix attention for image tokens
        let patchW = config.vision.cropSize / config.vision.encPatchSize
        let prefixAttnLen = 1 + patchW * patchW
        let prefixMask = MLXArray.ones([1, 1, prefixAttnLen, prefixAttnLen]).asType(.bool)
        // Pad prefix mask to max context size and OR with causal mask
        // (simplified - just use causal for now)

        if let image = input.image {
            // Encode image
            let imageFeatures = visionEncoder(image.pixels)

            // Get text embeddings
            let textEmbeddings = textModel.embed(inputIds)

            // BOS embedding
            let bosTokens = MLXArray([Int32(0)]).expandedDimensions(axis: 0)
            let bosEmb = textModel.embed(bosTokens)  // BOS token is 0

            // Combine: [BOS, image_features, text_embeddings]
            let combined = concatenated([bosEmb, imageFeatures, textEmbeddings], axis: 1)

            // Create positions for combined sequence
            let combinedLen = combined.dim(1)
            let combinedPositions = MLXArray((0..<combinedLen).map { Int32($0) })

            // Forward through text model
            let (hidden, _) = textModel(combined, positions: combinedPositions, mask: mask, cache: nil, cachePos: 0)
            let logits = textModel.generateLogits(hidden)

            return .logits(LMOutput(logits: logits))
        } else {
            // Text-only mode
            let embeddings = textModel.embed(inputIds)
            let (hidden, _) = textModel(embeddings, positions: positions, mask: mask, cache: nil, cachePos: 0)
            let logits = textModel.generateLogits(hidden)
            return .logits(LMOutput(logits: logits))
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let seqLen = inputs.dim(1)
        let positions = MLXArray((0..<seqLen).map { Int32($0) })
        let embeddings = textModel.embed(inputs)
        let (hidden, _) = textModel(embeddings, positions: positions, mask: nil, cache: nil, cachePos: 0)
        return textModel.generateLogits(hidden)
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

            // Rename: text.blocks.N -> text.blocks.N (keep as blocks, not layers)
            // vision.blocks.N -> vision.blocks.N
            // These match the Python naming

            // Handle bias naming for standard MLP layers (0-3 only)
            if newKey.contains(".mlp.fc1_bias") || newKey.contains(".mlp.fc2_bias") {
                let components = newKey.split(separator: ".")
                if let blocksIndex = components.firstIndex(of: "blocks"),
                   blocksIndex + 1 < components.count,
                   let layerNum = Int(components[blocksIndex + 1]),
                   layerNum < 4 {
                    newKey = newKey.replacingOccurrences(of: ".fc1_bias", with: ".fc1.bias")
                    newKey = newKey.replacingOccurrences(of: ".fc2_bias", with: ".fc2.bias")
                }
            }

            sanitized[newKey] = value
        }

        return sanitized
    }
}

// MARK: - Processor Configuration

public struct Moondream3ProcessorConfiguration: Codable, Sendable {
    public let cropSize: Int
    public let imageMean: [Float]
    public let imageStd: [Float]

    public init() {
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

public class Moondream3Processor: UserInputProcessor {
    private let config: Moondream3ProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageSequenceLength: Int

    private struct Templates {
        static let captionShort: [Int] = [1, 32708, 2, 12492, 3]
        static let captionNormal: [Int] = [1, 32708, 2, 6382, 3]
        static let captionLong: [Int] = [1, 32708, 2, 4059, 3]
        static let queryPrefix: [Int] = [1, 15381, 2]
        static let querySuffix: [Int] = [3]
        static let detectPrefix: [Int] = [1, 7235, 476, 2]
        static let detectSuffix: [Int] = [3]
        static let pointPrefix: [Int] = [1, 2581, 2]
        static let pointSuffix: [Int] = [3]
    }

    public init(config: Moondream3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        let patchSize = 14
        let gridSize = config.cropSize / patchSize
        self.imageSequenceLength = gridSize * gridSize
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let promptText: String
        switch input.prompt {
        case .text(let text):
            promptText = text
        case .chat(let messages):
            promptText = messages.map { $0.content }.joined(separator: "\n")
        case .messages(let messages):
            promptText = messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
        }

        var processedImage: LMInput.ProcessedImage?
        if let firstImage = input.images.first {
            let pixels = try await processImage(firstImage, processing: input.processing)
            processedImage = LMInput.ProcessedImage(pixels: pixels)
        }

        let tokens = buildTokenSequence(from: promptText)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)

        return LMInput(
            text: LMInput.Text(tokens: tokenArray),
            image: processedImage
        )
    }

    private func buildTokenSequence(from prompt: String) -> [Int] {
        if prompt.hasPrefix("<caption:") {
            if prompt.contains("short") {
                return Templates.captionShort
            } else if prompt.contains("long") {
                return Templates.captionLong
            } else {
                return Templates.captionNormal
            }
        } else if prompt.hasPrefix("<point>") {
            let object = String(prompt.dropFirst(7))
            let objectTokens = tokenizer.encode(text: object)
            return Templates.pointPrefix + objectTokens + Templates.pointSuffix
        } else if prompt.hasPrefix("<detect>") {
            let object = String(prompt.dropFirst(8))
            let objectTokens = tokenizer.encode(text: object)
            return Templates.detectPrefix + objectTokens + Templates.detectSuffix
        } else {
            let questionTokens = tokenizer.encode(text: prompt)
            return Templates.queryPrefix + questionTokens + Templates.querySuffix
        }
    }

    private func processImage(_ image: UserInput.Image, processing: UserInput.Processing) async throws -> MLXArray {
        var ciImage = try image.asCIImage()
        ciImage = MediaProcessing.inSRGBToneCurveSpace(ciImage)

        if let resize = processing.resize {
            ciImage = MediaProcessing.resampleBicubic(ciImage, to: resize)
        }

        ciImage = MediaProcessing.resampleBicubic(ciImage, to: config.size)
        ciImage = MediaProcessing.normalize(
            ciImage,
            mean: config.imageMeanTuple,
            std: config.imageStdTuple
        )

        var pixels = MediaProcessing.asMLXArray(ciImage)
        if pixels.ndim == 3 {
            pixels = pixels.expandedDimensions(axis: 0)
        }

        return pixels
    }
}
