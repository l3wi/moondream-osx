// Copyright 2024 Moondream AI
// Model configuration for Moondream3

import Foundation

/// Configuration for Moondream3 model
/// Loaded from config.json in the model repository
public struct Moondream3Configuration: Codable, Sendable {

    // MARK: - Text Configuration

    public struct TextConfiguration: Codable, Sendable {
        public let dim: Int
        public let ffDim: Int
        public let nLayers: Int
        public let vocabSize: Int
        public let maxContext: Int
        public let nHeads: Int
        public let nKvHeads: Int
        public let prefixAttn: Int
        public let moe: MoEConfiguration
        public let bits: Int?
        public let groupSize: Int?

        enum CodingKeys: String, CodingKey {
            case dim
            case ffDim = "ff_dim"
            case nLayers = "n_layers"
            case vocabSize = "vocab_size"
            case maxContext = "max_context"
            case nHeads = "n_heads"
            case nKvHeads = "n_kv_heads"
            case prefixAttn = "prefix_attn"
            case moe
            case bits
            case groupSize = "group_size"
        }
    }

    // MARK: - MoE Configuration

    public struct MoEConfiguration: Codable, Sendable {
        public let numExperts: Int
        public let startLayer: Int
        public let expertsPerToken: Int
        public let expertInnerDim: Int

        enum CodingKeys: String, CodingKey {
            case numExperts = "num_experts"
            case startLayer = "start_layer"
            case expertsPerToken = "experts_per_token"
            case expertInnerDim = "expert_inner_dim"
        }
    }

    // MARK: - Vision Configuration

    public struct VisionConfiguration: Codable, Sendable {
        public let encDim: Int
        public let encPatchSize: Int
        public let encNLayers: Int
        public let encFfDim: Int
        public let encNHeads: Int
        public let projOutDim: Int
        public let cropSize: Int
        public let inChannels: Int
        public let maxCrops: Int
        public let overlapMargin: Int
        public let projInnerDim: Int

        enum CodingKeys: String, CodingKey {
            case encDim = "enc_dim"
            case encPatchSize = "enc_patch_size"
            case encNLayers = "enc_n_layers"
            case encFfDim = "enc_ff_dim"
            case encNHeads = "enc_n_heads"
            case projOutDim = "proj_out_dim"
            case cropSize = "crop_size"
            case inChannels = "in_channels"
            case maxCrops = "max_crops"
            case overlapMargin = "overlap_margin"
            case projInnerDim = "proj_inner_dim"
        }
    }

    // MARK: - Region Configuration

    public struct RegionConfiguration: Codable, Sendable {
        public let dim: Int
        public let coordFeatDim: Int
        public let coordOutDim: Int
        public let sizeFeatDim: Int
        public let sizeOutDim: Int

        enum CodingKeys: String, CodingKey {
            case dim
            case coordFeatDim = "coord_feat_dim"
            case coordOutDim = "coord_out_dim"
            case sizeFeatDim = "size_feat_dim"
            case sizeOutDim = "size_out_dim"
        }
    }

    // MARK: - Properties

    public let modelType: String
    public let text: TextConfiguration
    public let vision: VisionConfiguration
    public let region: RegionConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text
        case vision
        case region
    }
}
