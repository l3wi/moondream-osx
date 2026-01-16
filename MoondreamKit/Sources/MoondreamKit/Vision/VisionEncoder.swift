// Copyright 2024 Moondream AI
// Vision Encoder for standard model

import Foundation
import MLX
import MLXNN

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
