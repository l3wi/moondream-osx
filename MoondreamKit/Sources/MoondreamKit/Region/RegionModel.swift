import Foundation
import MLX
import MLXNN

// MARK: - Fourier Features

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

// MARK: - Region Model (Standard)

/// Region model for coordinate/size encoding/decoding
/// Used for point() and detect() skills to decode coordinates from hidden states
internal class RegionModel: Module {
    let config: Moondream3Configuration.RegionConfiguration

    @ModuleInfo(key: "coord_features") var coordFeatures: MLXArray
    @ModuleInfo(key: "coord_encoder") var coordEncoder: Linear
    @ModuleInfo(key: "coord_decoder") var coordDecoder: Linear

    @ModuleInfo(key: "size_features") var sizeFeatures: MLXArray
    @ModuleInfo(key: "size_encoder") var sizeEncoder: Linear
    @ModuleInfo(key: "size_decoder") var sizeDecoder: Linear

    init(_ config: Moondream3Configuration.RegionConfiguration) {
        self.config = config

        // Coordinate encoding/decoding
        // Python: self.coord_features = mx.zeros((1, config.coord_feat_dim // 2))
        self._coordFeatures.wrappedValue = MLXArray.zeros([1, config.coordFeatDim / 2])
        // Python: self.coord_encoder = nn.Linear(config.coord_feat_dim, config.dim)
        self._coordEncoder.wrappedValue = Linear(config.coordFeatDim, config.dim, bias: true)
        // Python: self.coord_decoder = nn.Linear(config.dim, config.coord_out_dim)
        self._coordDecoder.wrappedValue = Linear(config.dim, config.coordOutDim, bias: true)

        // Size encoding/decoding
        // Python: self.size_features = mx.zeros((2, config.size_feat_dim // 2))
        self._sizeFeatures.wrappedValue = MLXArray.zeros([2, config.sizeFeatDim / 2])
        // Python: self.size_encoder = nn.Linear(config.size_feat_dim, config.dim)
        self._sizeEncoder.wrappedValue = Linear(config.sizeFeatDim, config.dim, bias: true)
        // Python: self.size_decoder = nn.Linear(config.dim, config.size_out_dim)
        self._sizeDecoder.wrappedValue = Linear(config.dim, config.sizeOutDim, bias: true)
    }

    /// Encode a coordinate value to embedding
    /// Python: def encode_coordinate(self, coord): features = fourier_features(coord, self.coord_features); return self.coord_encoder(features)
    func encodeCoordinate(_ coord: MLXArray) -> MLXArray {
        let features = FourierFeatures.encode(coord, w: coordFeatures)
        return coordEncoder(features)
    }

    /// Decode hidden state to coordinate logits
    /// Python: def decode_coordinate(self, hidden_state): return self.coord_decoder(hidden_state)
    func decodeCoordinate(_ hiddenState: MLXArray) -> MLXArray {
        return coordDecoder(hiddenState)
    }

    /// Encode size values to embedding
    /// Python: def encode_size(self, size): features = fourier_features(size, self.size_features); return self.size_encoder(features)
    func encodeSize(_ size: MLXArray) -> MLXArray {
        let features = FourierFeatures.encode(size, w: sizeFeatures)
        return sizeEncoder(features)
    }

    /// Decode hidden state to size logits
    /// Python: def decode_size(self, hidden_state): logits = self.size_decoder(hidden_state); return logits.reshape(2, -1)
    func decodeSize(_ hiddenState: MLXArray) -> MLXArray {
        let logits = sizeDecoder(hiddenState)
        return logits.reshaped(2, -1)
    }
}

// MARK: - Region Model (Quantized)

/// Quantized region model with int4 linear layers
/// Used in the fully quantized model variant
internal class QuantizedRegionModel: Module {
    let config: Moondream3Configuration.RegionConfiguration

    @ModuleInfo(key: "coord_features") var coordFeatures: MLXArray
    @ModuleInfo(key: "coord_encoder") var coordEncoder: QuantizedLinear
    @ModuleInfo(key: "coord_decoder") var coordDecoder: QuantizedLinear

    @ModuleInfo(key: "size_features") var sizeFeatures: MLXArray
    @ModuleInfo(key: "size_encoder") var sizeEncoder: QuantizedLinear
    @ModuleInfo(key: "size_decoder") var sizeDecoder: QuantizedLinear

    init(_ config: Moondream3Configuration.RegionConfiguration) {
        self.config = config

        self._coordFeatures.wrappedValue = MLXArray.zeros([1, config.coordFeatDim / 2])
        self._coordEncoder.wrappedValue = QuantizedLinear(config.coordFeatDim, config.dim, bias: true, groupSize: 64, bits: 4)
        self._coordDecoder.wrappedValue = QuantizedLinear(config.dim, config.coordOutDim, bias: true, groupSize: 64, bits: 4)

        self._sizeFeatures.wrappedValue = MLXArray.zeros([2, config.sizeFeatDim / 2])
        self._sizeEncoder.wrappedValue = QuantizedLinear(config.sizeFeatDim, config.dim, bias: true, groupSize: 64, bits: 4)
        self._sizeDecoder.wrappedValue = QuantizedLinear(config.dim, config.sizeOutDim, bias: true, groupSize: 64, bits: 4)
    }

    func encodeCoordinate(_ coord: MLXArray) -> MLXArray {
        let features = FourierFeatures.encode(coord, w: coordFeatures)
        return coordEncoder(features)
    }

    func decodeCoordinate(_ hiddenState: MLXArray) -> MLXArray {
        return coordDecoder(hiddenState)
    }

    func encodeSize(_ size: MLXArray) -> MLXArray {
        let features = FourierFeatures.encode(size, w: sizeFeatures)
        return sizeEncoder(features)
    }

    func decodeSize(_ hiddenState: MLXArray) -> MLXArray {
        let logits = sizeDecoder(hiddenState)
        return logits.reshaped(2, -1)
    }
}
