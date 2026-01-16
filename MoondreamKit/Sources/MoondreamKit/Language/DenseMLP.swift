// Copyright 2024 Moondream AI
// Dense MLP for non-MoE layers

import Foundation
import MLX
import MLXNN

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
        // Python uses gelu_approx
        fc2(geluApproximate(fc1(x)))
    }
}
