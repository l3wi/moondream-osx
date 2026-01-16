// Copyright 2024 Moondream AI
// Quantized Vision Encoder

import Foundation
import MLX
import MLXNN

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
