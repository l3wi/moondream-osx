// Copyright 2024 Moondream AI
// Rotary Position Embedding (RoPE) implementation

import Foundation
import MLX

// MARK: - RoPE Helpers

/// Precompute frequencies for rotary position embeddings
/// Python reference: precompute_freqs_cis(dim, max_len, theta=1500000.0)
/// - Parameters:
///   - dim: Dimension of the rotary embedding (typically head_dim / 2)
///   - maxLen: Maximum sequence length
///   - theta: Base frequency (default 1,500,000.0 for long context)
/// - Returns: Precomputed frequencies tensor of shape [maxLen, dim/2, 2]
internal func precomputeFreqsCis(dim: Int, maxLen: Int, theta: Float = 1_500_000.0) -> MLXArray {
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

/// Apply rotary position embeddings to queries/keys
/// Python reference: apply_rotary_emb(x, freqs_cis, positions)
/// - Parameters:
///   - x: Input tensor of shape [B, num_heads, seq_len, head_dim]
///   - freqsCis: Precomputed frequencies from `precomputeFreqsCis`
///   - positions: Position indices for this sequence
/// - Returns: Tensor with rotary embeddings applied
internal func applyRotaryEmb(_ x: MLXArray, freqsCis: MLXArray, positions: MLXArray) -> MLXArray {
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
    let newShape = Array(shape.dropLast(2)) + [shape[shape.count - 2] * shape[shape.count - 1]]
    let xRotOutReshaped = xRotOut.reshaped(newShape)

    // return mx.concatenate([x_rot_out, x_pass], axis=-1)
    return concatenated([xRotOutReshaped, xPass], axis: -1)
}
