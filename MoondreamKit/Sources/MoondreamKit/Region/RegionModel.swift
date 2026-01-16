// Copyright 2024 Moondream AI
// Region model for coordinate/size encoding/decoding

import Foundation
import MLX
import MLXNN

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
