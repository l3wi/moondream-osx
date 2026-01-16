// Copyright 2024 Moondream AI
// Quantized Dense MLP for non-MoE layers

import Foundation
import MLX
import MLXNN

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
