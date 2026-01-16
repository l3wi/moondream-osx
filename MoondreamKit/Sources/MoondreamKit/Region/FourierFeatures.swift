// Copyright 2024 Moondream AI
// Fourier feature encoding for Region model

import Foundation
import MLX

/// Fourier feature encoding for coordinates and sizes
/// Used by RegionModel for point() and detect() skills
internal enum FourierFeatures {

    /// Fourier feature encoding for coordinates/sizes
    /// Python: def fourier_features(x, w): f = 2 * math.pi * (x @ w); return mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)
    /// - Parameters:
    ///   - x: Input coordinates or sizes
    ///   - w: Learnable frequency weights
    /// - Returns: Fourier feature embedding
    static func encode(_ x: MLXArray, w: MLXArray) -> MLXArray {
        let f = 2 * Float.pi * matmul(x, w)
        return concatenated([cos(f), sin(f)], axis: -1)
    }

    /// Convert bin index to size value
    /// Python: def bin_to_size(bin_idx): return mx.power(2.0, (bin_idx.astype(mx.float32) / 1023.0) * 10.0 - 10.0)
    /// - Parameter binIdx: Bin index from size decoder
    /// - Returns: Decoded size value
    static func binToSize(_ binIdx: MLXArray) -> MLXArray {
        return pow(2.0, (binIdx.asType(.float32) / 1023.0) * 10.0 - 10.0)
    }
}
