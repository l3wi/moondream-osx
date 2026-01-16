// Copyright 2024 Moondream AI
// Quantized Vision MLP layer

import Foundation
import MLX
import MLXNN

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
