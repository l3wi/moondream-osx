// Copyright 2024 Moondream AI
// Vision MLP layer for standard model

import Foundation
import MLX
import MLXNN

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
        // Python uses gelu_approx
        fc2(geluApproximate(fc1(x)))
    }
}
