import Foundation

// MARK: - Model Configuration

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

// MARK: - Platform Configuration

/// Platform-specific configuration constants
/// Centralizes all `#if os()` conditional values
public enum PlatformConfiguration {

    // MARK: - KV Cache

    /// Maximum sequence length for KV cache
    /// iOS uses smaller cache to reduce memory pressure
    #if os(iOS)
    public static let maxCacheSeqLen: Int = 256
    #else
    public static let maxCacheSeqLen: Int = 1024
    #endif

    // MARK: - Generation

    /// Default maximum tokens for text generation
    /// iOS uses conservative limits for memory
    #if os(iOS)
    public static let defaultMaxTokens: Int = 128
    #else
    public static let defaultMaxTokens: Int = 768
    #endif

    // MARK: - Logging

    /// Whether file-based logging is enabled
    /// Only enabled on iOS for debugging inference issues
    #if os(iOS)
    public static let fileLoggingEnabled: Bool = true
    #else
    public static let fileLoggingEnabled: Bool = false
    #endif

    // MARK: - Model Selection

    /// Recommended model ID for this platform
    /// iOS prefers the compact model to fit in memory
    #if os(iOS)
    public static let recommendedModelId: String = "lewi/md3p-int4-smol"
    #else
    public static let recommendedModelId: String = "moondream/md3p-int4"
    #endif
}

// MARK: - Token Constants

/// Centralized token constants for Moondream3 model
/// These match the tokenizer configuration from moondream/starmie-v1
public enum TokenConstants {
    // MARK: - Special Tokens

    /// Beginning of sequence token (also used as EOS in this model)
    public static let bos: Int = 0

    /// End of sequence token (same as BOS in Moondream3)
    public static let eos: Int = 0

    /// Answer token - marks end of reasoning section
    public static let answer: Int = 3

    /// Thinking token - marks start of reasoning
    public static let thinking: Int = 4

    /// Coordinate token - used for point() and detect() skills
    public static let coord: Int = 5

    // MARK: - Caption Templates

    /// Token templates for caption generation
    public enum Caption {
        /// Short caption: concise 1-2 sentence description
        public static let short: [Int] = [1, 32708, 2, 12492, 3]

        /// Normal caption: standard description
        public static let normal: [Int] = [1, 32708, 2, 6382, 3]

        /// Long caption: detailed description
        public static let long: [Int] = [1, 32708, 2, 4059, 3]

        /// Get template for caption length
        public static func template(for length: String) -> [Int] {
            switch length {
            case "short": return short
            case "long": return long
            default: return normal
            }
        }
    }

    // MARK: - Query Templates

    /// Token templates for query (visual Q&A)
    public enum Query {
        /// Prefix tokens before question
        public static let prefix: [Int] = [1, 15381, 2]

        /// Suffix tokens after question
        public static let suffix: [Int] = [3]
    }

    // MARK: - Point Templates

    /// Token templates for point (locate object)
    public enum Point {
        /// Prefix tokens before object name
        public static let prefix: [Int] = [1, 2581, 2]

        /// Suffix tokens after object name
        public static let suffix: [Int] = [3]
    }

    // MARK: - Detect Templates

    /// Token templates for detect (bounding boxes)
    public enum Detect {
        /// Prefix tokens before object name
        public static let prefix: [Int] = [1, 7235, 476, 2]

        /// Suffix tokens after object name
        public static let suffix: [Int] = [3]
    }
}
