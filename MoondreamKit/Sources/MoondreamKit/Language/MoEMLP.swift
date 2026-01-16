// Copyright 2024 Moondream AI
// Mixture of Experts MLP

import Foundation
import MLX
import MLXNN

/// Quantized Mixture of Experts MLP
/// Used in layers 4-23 of the language model (MoE layers)
/// 64 experts, 8 active per token
internal class MoEMLP: Module, UnaryLayer {
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

    init(dim: Int, numExperts: Int, expertDim: Int, expertsPerToken: Int, bits: Int = 4) {
        self.dim = dim
        self.numExperts = numExperts
        self.expertDim = expertDim
        self.expertsPerToken = expertsPerToken
        self.bits = bits
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
