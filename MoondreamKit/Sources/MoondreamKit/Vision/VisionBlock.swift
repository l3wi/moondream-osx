import Foundation
import MLX
import MLXNN

// MARK: - Vision Block (Standard)

/// Vision Transformer block with attention and MLP
/// Pre-norm architecture with residual connections
internal class VisionBlock: Module {
    @ModuleInfo var ln1: LayerNorm
    @ModuleInfo(key: "attn") var attention: VisionAttention
    @ModuleInfo var ln2: LayerNorm
    @ModuleInfo var mlp: VisionMLP

    init(dim: Int, numHeads: Int, ffDim: Int) {
        self._ln1.wrappedValue = LayerNorm(dimensions: dim)
        self._attention.wrappedValue = VisionAttention(dim: dim, numHeads: numHeads)
        self._ln2.wrappedValue = LayerNorm(dimensions: dim)
        self.mlp = VisionMLP(dim: dim, hiddenDim: ffDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(ln1(x))
        h = h + mlp(ln2(h))
        return h
    }
}

// MARK: - Vision Block (Quantized)

/// Quantized Vision Transformer block
/// Uses quantized attention and MLP layers
internal class QuantizedVisionBlock: Module {
    @ModuleInfo var ln1: LayerNorm
    @ModuleInfo(key: "attn") var attention: QuantizedVisionAttention
    @ModuleInfo var ln2: LayerNorm
    @ModuleInfo var mlp: QuantizedVisionMLP

    init(dim: Int, numHeads: Int, ffDim: Int) {
        self._ln1.wrappedValue = LayerNorm(dimensions: dim)
        self._attention.wrappedValue = QuantizedVisionAttention(dim: dim, numHeads: numHeads)
        self._ln2.wrappedValue = LayerNorm(dimensions: dim)
        self.mlp = QuantizedVisionMLP(dim: dim, hiddenDim: ffDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(ln1(x))
        h = h + mlp(ln2(h))
        return h
    }
}
