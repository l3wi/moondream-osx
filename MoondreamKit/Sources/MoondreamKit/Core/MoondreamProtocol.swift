import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

/// Protocol defining the common interface for Moondream models
/// Both standard and quantized variants conform to this protocol
public protocol MoondreamModel: Module, LanguageModel, KVCacheDimensionProvider {

    /// Model configuration
    var config: Moondream3Configuration { get }

    /// Encode image pixels into embeddings for the language model
    /// - Parameter pixels: Image tensor in NCHW format [B, C, H, W]
    /// - Returns: Image embeddings [B, num_patches, dim]
    func encodeImage(_ pixels: MLXArray) -> MLXArray

    /// Generate a caption for an image
    /// - Parameters:
    ///   - pixels: Image tensor in NCHW format
    ///   - length: Caption length ("short", "normal", "long")
    ///   - tokenizer: Tokenizer for encoding/decoding
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature (0 = greedy)
    /// - Returns: Generated caption string
    func caption(
        pixels: MLXArray,
        length: String,
        tokenizer: any Tokenizer,
        maxTokens: Int,
        temperature: Float
    ) -> String

    /// Answer a question about an image
    /// - Parameters:
    ///   - pixels: Image tensor in NCHW format
    ///   - question: Question to answer
    ///   - tokenizer: Tokenizer for encoding/decoding
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature
    /// - Returns: Generated answer string
    func query(
        pixels: MLXArray,
        question: String,
        tokenizer: any Tokenizer,
        maxTokens: Int,
        temperature: Float
    ) -> String

    /// Locate an object in the image
    /// - Parameters:
    ///   - pixels: Image tensor in NCHW format
    ///   - object: Object description to find
    ///   - tokenizer: Tokenizer for encoding
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature
    /// - Returns: Formatted coordinate string "(x, y), ..."
    func point(
        pixels: MLXArray,
        object: String,
        tokenizer: any Tokenizer,
        maxTokens: Int,
        temperature: Float
    ) -> String

    /// Detect all instances of an object in the image
    /// - Parameters:
    ///   - pixels: Image tensor in NCHW format
    ///   - object: Object description to detect
    ///   - tokenizer: Tokenizer for encoding
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature
    /// - Returns: Formatted bounding box string "[xmin, ymin, xmax, ymax], ..."
    func detect(
        pixels: MLXArray,
        object: String,
        tokenizer: any Tokenizer,
        maxTokens: Int,
        temperature: Float
    ) -> String
}
