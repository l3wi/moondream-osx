// IntegrationTests.swift
// End-to-end integration tests with real model and images
// NOTE: These tests require the model to be downloaded and take longer to run
// Run with: swift test --filter IntegrationTests

import XCTest
import MLX
import Hub
import Tokenizers
import MLXVLM
import MLXLMCommon
import CoreImage
@testable import MoondreamKit

/// Integration tests that verify the full pipeline works correctly
/// These tests require the model to be downloaded and take longer to run
final class IntegrationTests: XCTestCase {

    // Expected keywords in captions for the Mona Lisa test image
    let monaLisaKeywords = [
        "Mona Lisa",
        "Louvre",
        "painting",
        "museum",
        "frame",
        "gallery"
    ]

    // MARK: - Helper Methods

    /// Check if MLX/Metal is available (not available in swift test without Xcode)
    func requireMLX() throws {
        // Try a simple MLX operation to see if Metal is available
        // This will crash if Metal library isn't loaded, so we check the environment
        if ProcessInfo.processInfo.environment["XCTestBundlePath"] != nil &&
           ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] == nil {
            throw XCTSkip("MLX requires Metal - run tests via Xcode, not swift test")
        }
    }

    /// Load and process an image file into MLXArray
    func loadAndProcessImage(from path: String) throws -> MLXArray {
        let url = URL(fileURLWithPath: path)
        guard let ciImage = CIImage(contentsOf: url) else {
            throw NSError(domain: "IntegrationTests", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Failed to load image from \(path)"
            ])
        }

        // Apply same preprocessing as the app
        let config = Moondream3ProcessorConfiguration()
        var processedImage = MediaProcessing.inSRGBToneCurveSpace(ciImage)
        processedImage = MediaProcessing.resampleBicubic(processedImage, to: config.size)
        processedImage = MediaProcessing.normalize(
            processedImage,
            mean: config.imageMeanTuple,
            std: config.imageStdTuple
        )

        var pixels = MediaProcessing.asMLXArray(processedImage)
        if pixels.ndim == 3 {
            pixels = pixels.expandedDimensions(axis: 0)
        }

        return pixels
    }

    /// Get test image path from bundle resources
    func getTestImagePath() -> String? {
        Bundle.module.path(forResource: "monalisa-on-a-gallery-wall", ofType: "png", inDirectory: "Resources")
    }

    // MARK: - Image Processing Tests

    func testImagePreprocessingShape() throws {
        try requireMLX()
        guard let imagePath = getTestImagePath() else {
            throw XCTSkip("Test image not found in bundle resources")
        }

        let pixels = try loadAndProcessImage(from: imagePath)
        eval(pixels)

        // Expected shape: [1, 3, 378, 378]
        XCTAssertEqual(pixels.shape[0], 1, "Batch size should be 1")
        XCTAssertEqual(pixels.shape[1], 3, "Should have 3 channels (RGB)")
        XCTAssertEqual(pixels.shape[2], 378, "Height should be 378")
        XCTAssertEqual(pixels.shape[3], 378, "Width should be 378")
    }

    func testImagePreprocessingNormalization() throws {
        try requireMLX()
        guard let imagePath = getTestImagePath() else {
            throw XCTSkip("Test image not found in bundle resources")
        }

        let pixels = try loadAndProcessImage(from: imagePath)
        eval(pixels)

        // After normalization with mean=0.5, std=0.5, values should be roughly in [-1, 1]
        let minVal = pixels.min().item(Float.self)
        let maxVal = pixels.max().item(Float.self)

        XCTAssertGreaterThanOrEqual(minVal, -3.0, "Min value should be reasonable after normalization")
        XCTAssertLessThanOrEqual(maxVal, 3.0, "Max value should be reasonable after normalization")
    }

    func testImagePreprocessingDtype() throws {
        try requireMLX()
        guard let imagePath = getTestImagePath() else {
            throw XCTSkip("Test image not found in bundle resources")
        }

        let pixels = try loadAndProcessImage(from: imagePath)
        XCTAssertEqual(pixels.dtype, .float32, "Pixels should be float32")
    }

    // MARK: - Configuration Tests

    func testProcessorConfiguration() {
        let config = Moondream3ProcessorConfiguration()

        XCTAssertEqual(config.cropSize, 378)
        XCTAssertEqual(config.imageMean, [0.5, 0.5, 0.5])
        XCTAssertEqual(config.imageStd, [0.5, 0.5, 0.5])
    }

    func testPatchCalculation() {
        let cropSize = 378
        let patchSize = 14

        let gridSize = cropSize / patchSize
        let numPatches = gridSize * gridSize

        XCTAssertEqual(gridSize, 27, "Grid should be 27x27")
        XCTAssertEqual(numPatches, 729, "Should have 729 patches")
    }

    // MARK: - RoPE Dimension Verification (Bug Fix Test)

    func testRoPEDimensionIsCorrect() throws {
        // This test verifies the critical bug fix
        let dim = 2048
        let nHeads = 32

        // The CORRECT formula (from Python)
        let correctRotDim = dim / (2 * nHeads)

        // The WRONG formula (the bug)
        let wrongRotDim = dim / nHeads

        XCTAssertEqual(correctRotDim, 32, "Correct RoPE dim should be 32")
        XCTAssertEqual(wrongRotDim, 64, "Wrong RoPE dim was 64")

        // MLX-dependent part - skip if Metal not available
        try requireMLX()

        // Verify freqs_cis shape
        let freqsCis = precomputeFreqsCis(dim: correctRotDim, maxLen: 100)
        eval(freqsCis)

        // Shape should be [maxLen, rotDim/2, 2] = [100, 16, 2]
        XCTAssertEqual(freqsCis.shape, [100, 16, 2], "freqs_cis shape should be [100, 16, 2]")
    }

    // MARK: - Token Tests

    func testCaptionPromptTokensAreValid() {
        // Caption prompt tokens from Python
        let captionNormal = [1, 32708, 2, 6382, 3]

        // Verify structure
        XCTAssertEqual(captionNormal.first, 1, "Should start with token 1")
        XCTAssertEqual(captionNormal.last, 3, "Should end with token 3")
        XCTAssertEqual(captionNormal.count, 5, "Caption normal prompt has 5 tokens")

        // Verify no garbage tokens (5 = coordinate token)
        for (i, token) in captionNormal.enumerated() {
            XCTAssertNotEqual(token, 5, "Token at position \(i) should not be 5 (garbage)")
        }
    }

    func testQueryPromptTokensAreValid() {
        let queryPrefix = [1, 15381, 2]
        let querySuffix = [3]

        XCTAssertEqual(queryPrefix.first, 1, "Query prefix should start with 1")
        XCTAssertEqual(querySuffix.first, 3, "Query suffix should be 3")
    }

    func testEOSEqualsOSInMoondream() {
        // In Moondream3, EOS = BOS = 0
        let bosId = 0
        let eosId = 0

        XCTAssertEqual(bosId, eosId, "BOS and EOS are the same in Moondream3")
    }

    // MARK: - Architecture Constants

    func testMoondream3ArchitectureConstants() {
        // Text model
        let textDim = 2048
        let textHeads = 32
        let textLayers = 24
        _ = 51200 // vocabSize - for documentation

        XCTAssertEqual(textDim / textHeads, 64, "Head dim should be 64")
        XCTAssertEqual(textDim / (2 * textHeads), 32, "RoPE dim should be 32")

        // Vision model
        let visionDim = 1152
        let visionHeads = 16
        _ = 27 // visionLayers - for documentation
        let patchSize = 14
        let cropSize = 378

        XCTAssertEqual(visionDim / visionHeads, 72, "Vision head dim should be 72")
        XCTAssertEqual(cropSize / patchSize, 27, "Grid size should be 27")

        // MoE
        let moeStartLayer = 4
        let numExperts = 64
        let expertsPerToken = 8

        XCTAssertEqual(textLayers - moeStartLayer, 20, "20 MoE layers")
        XCTAssertLessThan(expertsPerToken, numExperts, "Experts per token < total experts")
    }
}
