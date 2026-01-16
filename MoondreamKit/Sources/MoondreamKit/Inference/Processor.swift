// Copyright 2024 Moondream AI
// Image processor for Moondream3

import CoreGraphics
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Tokenizers

// MARK: - Processor Configuration

/// Configuration for image preprocessing
public struct Moondream3ProcessorConfiguration: Codable, Sendable {
    public let cropSize: Int
    public let imageMean: [Float]
    public let imageStd: [Float]

    public init() {
        self.cropSize = 378
        self.imageMean = [0.5, 0.5, 0.5]
        self.imageStd = [0.5, 0.5, 0.5]
    }

    public init(cropSize: Int, imageMean: [Float], imageStd: [Float]) {
        self.cropSize = cropSize
        self.imageMean = imageMean
        self.imageStd = imageStd
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (CGFloat(imageMean[0]), CGFloat(imageMean[1]), CGFloat(imageMean[2]))
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (CGFloat(imageStd[0]), CGFloat(imageStd[1]), CGFloat(imageStd[2]))
    }

    public var size: CGSize {
        CGSize(width: cropSize, height: cropSize)
    }
}

// MARK: - Processor

/// Moondream3 image and text processor
/// Handles image preprocessing and prompt tokenization
public class Moondream3Processor: UserInputProcessor {
    private let config: Moondream3ProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageSequenceLength: Int

    public init(config: Moondream3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        let patchSize = 14
        let gridSize = config.cropSize / patchSize
        self.imageSequenceLength = gridSize * gridSize
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let promptText: String
        switch input.prompt {
        case .text(let text):
            promptText = text
        case .chat(let messages):
            promptText = messages.map { $0.content }.joined(separator: "\n")
        case .messages(let messages):
            promptText = messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
        }

        var processedImage: LMInput.ProcessedImage?
        if let firstImage = input.images.first {
            let pixels = try await processImage(firstImage, processing: input.processing)
            processedImage = LMInput.ProcessedImage(pixels: pixels)
        }

        let tokens = buildTokenSequence(from: promptText)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)

        return LMInput(
            text: LMInput.Text(tokens: tokenArray),
            image: processedImage
        )
    }

    private func buildTokenSequence(from prompt: String) -> [Int] {
        if prompt.hasPrefix("<caption:") {
            if prompt.contains("short") {
                return TokenConstants.Caption.short
            } else if prompt.contains("long") {
                return TokenConstants.Caption.long
            } else {
                return TokenConstants.Caption.normal
            }
        } else if prompt.hasPrefix("<point>") {
            let object = String(prompt.dropFirst(7))
            let objectTokens = tokenizer.encode(text: object)
            return TokenConstants.Point.prefix + objectTokens + TokenConstants.Point.suffix
        } else if prompt.hasPrefix("<detect>") {
            let object = String(prompt.dropFirst(8))
            let objectTokens = tokenizer.encode(text: object)
            return TokenConstants.Detect.prefix + objectTokens + TokenConstants.Detect.suffix
        } else {
            let questionTokens = tokenizer.encode(text: prompt)
            return TokenConstants.Query.prefix + questionTokens + TokenConstants.Query.suffix
        }
    }

    private func processImage(_ image: UserInput.Image, processing: UserInput.Processing) async throws -> MLXArray {
        var ciImage = try image.asCIImage()
        ciImage = MediaProcessing.inSRGBToneCurveSpace(ciImage)

        if let resize = processing.resize {
            ciImage = MediaProcessing.resampleBicubic(ciImage, to: resize)
        }

        ciImage = MediaProcessing.resampleBicubic(ciImage, to: config.size)
        ciImage = MediaProcessing.normalize(
            ciImage,
            mean: config.imageMeanTuple,
            std: config.imageStdTuple
        )

        var pixels = MediaProcessing.asMLXArray(ciImage)
        if pixels.ndim == 3 {
            pixels = pixels.expandedDimensions(axis: 0)
        }

        return pixels
    }
}
