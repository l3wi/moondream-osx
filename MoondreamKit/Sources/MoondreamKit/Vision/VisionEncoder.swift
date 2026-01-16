// Copyright 2024 Moondream AI
// Vision Encoder for Moondream3 (standard and quantized variants)

import Foundation
import MLX
import MLXNN

// MARK: - Vision MLP (Standard)

/// Vision MLP with GELU approximate activation
/// Used in Vision Transformer blocks
internal class VisionMLP: Module, UnaryLayer {
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

// MARK: - Vision MLP (Quantized)

/// Quantized vision MLP with mixed precision
/// fc1 is quantized, fc2 kept as BF16 due to shape incompatibility (4304 % 64 != 0)
internal class QuantizedVisionMLP: Module, UnaryLayer {
    @ModuleInfo var fc1: QuantizedLinear
    @ModuleInfo var fc2: Linear  // fc2 kept as BF16 due to shape incompatibility

    init(dim: Int, hiddenDim: Int) {
        self.fc1 = QuantizedLinear(dim, hiddenDim, bias: true, groupSize: 64, bits: 4)
        self.fc2 = Linear(hiddenDim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

// MARK: - Vision Encoder (Standard)

/// Vision encoder - processes images into embeddings for the language model
/// Python uses Linear for patch_emb, NOT Conv2d!
internal class VisionEncoder: Module {
    let config: Moondream3Configuration.VisionConfiguration

    // Python: self.patch_emb = nn.Linear(patch_dim, config.enc_dim)
    @ModuleInfo(key: "patch_emb") var patchEmb: Linear
    @ModuleInfo(key: "pos_emb") var posEmb: MLXArray

    let blocks: [VisionBlock]
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
            VisionBlock(dim: config.encDim, numHeads: config.encNHeads, ffDim: config.encFfDim)
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

// MARK: - Vision Encoder (Quantized)

/// Quantized vision encoder for the compact model variant
/// patchEmb kept as BF16 due to shape incompatibility (588 % 64 != 0, where 588 = 14*14*3)
internal class QuantizedVisionEncoder: Module {
    let config: Moondream3Configuration.VisionConfiguration

    // patchEmb kept as BF16 - patchDim (588 = 14*14*3) is not divisible by 64
    @ModuleInfo(key: "patch_emb") var patchEmb: Linear
    @ModuleInfo(key: "pos_emb") var posEmb: MLXArray

    let blocks: [QuantizedVisionBlock]
    @ModuleInfo(key: "post_ln") var postLn: LayerNorm

    @ModuleInfo(key: "proj_fc1") var projFc1: QuantizedLinear
    @ModuleInfo(key: "proj_fc2") var projFc2: QuantizedLinear

    init(_ config: Moondream3Configuration.VisionConfiguration) {
        self.config = config

        let patchDim = config.encPatchSize * config.encPatchSize * config.inChannels
        // patchEmb kept as BF16 - patchDim (588 = 14*14*3) is not divisible by 64
        self._patchEmb.wrappedValue = Linear(patchDim, config.encDim, bias: true)

        let gridSize = config.cropSize / config.encPatchSize
        let numPatches = gridSize * gridSize

        self._posEmb.wrappedValue = MLXArray.zeros([1, numPatches, config.encDim])

        self.blocks = (0..<config.encNLayers).map { _ in
            QuantizedVisionBlock(dim: config.encDim, numHeads: config.encNHeads, ffDim: config.encFfDim)
        }
        self._postLn.wrappedValue = LayerNorm(dimensions: config.encDim)

        self._projFc1.wrappedValue = QuantizedLinear(config.encDim * 2, config.projInnerDim, bias: true, groupSize: 64, bits: 4)
        self._projFc2.wrappedValue = QuantizedLinear(config.projInnerDim, config.projOutDim, bias: true, groupSize: 64, bits: 4)
    }

    func createPatches(_ x: MLXArray) -> MLXArray {
        let (B, C, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let P = config.encPatchSize

        var patches = x.reshaped(B, C, H / P, P, W / P, P)
        patches = patches.transposed(0, 2, 4, 1, 3, 5)
        patches = patches.reshaped(B, (H / P) * (W / P), C * P * P)

        return patches
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        eval(x)

        var patches = createPatches(x)
        eval(patches)
        patches = patchEmb(patches)
        patches = patches + posEmb

        var h = patches
        for block in blocks {
            h = block(h)
        }
        h = postLn(h)

        let concatenatedFeatures = concatenated([h, h], axis: -1)

        let fc1Out = projFc1(concatenatedFeatures)
        let geluOut = geluApproximate(fc1Out)
        let projected = projFc2(geluOut)

        return projected
    }
}
