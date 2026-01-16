// Copyright 2024 Moondream AI
// Quantized Region model for coordinate/size encoding/decoding

import Foundation
import MLX
import MLXNN

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
