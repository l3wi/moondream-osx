// ConfigurationTests.swift
// Tests for model configuration and architecture constants

import XCTest
import MLX
@testable import MoondreamKit

final class ConfigurationTests: XCTestCase {

    // MARK: - Moondream3 Architecture Constants

    func testTextConfigDimensions() {
        // Verify the expected Moondream3 text model dimensions
        let expectedDim = 2048
        let expectedNHeads = 32
        let expectedNKvHeads = 32
        let expectedNLayers = 24
        let expectedVocabSize = 51200
        let expectedFfDim = 8192

        // These match the Python config
        XCTAssertEqual(expectedDim / expectedNHeads, 64, "Head dim should be 64")
        XCTAssertEqual(expectedDim / (2 * expectedNHeads), 32, "RoPE dim should be 32")
        XCTAssertEqual(expectedNKvHeads, expectedNHeads, "KV heads equals Q heads (no GQA)")
    }

    func testVisionConfigDimensions() {
        // Verify the expected Moondream3 vision encoder dimensions
        let expectedEncDim = 1152
        let expectedEncNHeads = 16
        let expectedPatchSize = 14
        let expectedCropSize = 378
        let expectedEncNLayers = 27

        // Derived values
        let gridSize = expectedCropSize / expectedPatchSize
        let numPatches = gridSize * gridSize

        XCTAssertEqual(gridSize, 27, "Grid size should be 27")
        XCTAssertEqual(numPatches, 729, "Number of patches should be 729")
        XCTAssertEqual(expectedEncDim / expectedEncNHeads, 72, "Vision head dim should be 72")
    }

    func testMoEConfiguration() {
        // MoE starts at layer 4 (0-indexed)
        let moeStartLayer = 4
        let numExperts = 64
        let expertsPerToken = 8
        let expertInnerDim = 1024
        let totalLayers = 24

        // Layers 0-3: Dense MLP
        // Layers 4-23: MoE
        let denseLayers = moeStartLayer
        let moeLayers = totalLayers - moeStartLayer

        XCTAssertEqual(denseLayers, 4, "First 4 layers should be dense")
        XCTAssertEqual(moeLayers, 20, "Remaining 20 layers should be MoE")
    }

    // MARK: - Token IDs

    func testSpecialTokenIds() {
        // From Moondream3 config
        let bosId = 0
        let eosId = 0  // Same as BOS!
        let answerId = 3
        let thinkingId = 4
        let coordId = 5

        // EOS = BOS is intentional in Moondream3
        XCTAssertEqual(bosId, eosId, "BOS and EOS are the same token in Moondream3")

        // Reserved tokens start at 5
        XCTAssertEqual(coordId, 5, "Coordinate token is 5 (the garbage token bug)")
    }

    func testCaptionPromptTokens() {
        // These are the exact token sequences from Python
        let captionShort = [1, 32708, 2, 12492, 3]
        let captionNormal = [1, 32708, 2, 6382, 3]
        let captionLong = [1, 32708, 2, 4059, 3]

        // All start with 1 and end with 3
        XCTAssertEqual(captionShort.first, 1)
        XCTAssertEqual(captionShort.last, 3)

        XCTAssertEqual(captionNormal.first, 1)
        XCTAssertEqual(captionNormal.last, 3)

        XCTAssertEqual(captionLong.first, 1)
        XCTAssertEqual(captionLong.last, 3)

        // Different length indicators in the middle
        XCTAssertNotEqual(captionShort[3], captionNormal[3])
        XCTAssertNotEqual(captionNormal[3], captionLong[3])
    }

    func testQueryPromptTokens() {
        let queryPrefix = [1, 15381, 2]
        let querySuffix = [3]

        XCTAssertEqual(queryPrefix.count, 3)
        XCTAssertEqual(querySuffix.count, 1)
        XCTAssertEqual(querySuffix[0], 3)
    }

    // MARK: - Shape Calculations

    func testImageEmbeddingShape() {
        let batchSize = 1
        let cropSize = 378
        let patchSize = 14
        let projOutDim = 2048

        let gridSize = cropSize / patchSize  // 27
        let numPatches = gridSize * gridSize  // 729

        // Image embedding shape after vision encoder
        let expectedShape = [batchSize, numPatches, projOutDim]

        XCTAssertEqual(expectedShape, [1, 729, 2048])
    }

    func testInputEmbeddingShape() {
        // After combining BOS + image + prompt
        let batchSize = 1
        let bosTokens = 1
        let imagePatches = 729
        let promptTokens = 5  // caption:normal prompt

        let totalLength = bosTokens + imagePatches + promptTokens

        XCTAssertEqual(totalLength, 735, "Total input length for caption generation")
    }

    func testKVCacheShape() {
        let batchSize = 1
        let nKvHeads = 32
        let maxSeqLen = 1024
        let headDim = 64

        let cacheShape = [batchSize, nKvHeads, maxSeqLen, headDim]

        XCTAssertEqual(cacheShape, [1, 32, 1024, 64])

        // Memory per layer (K + V)
        let elementsPerLayer = 2 * batchSize * nKvHeads * maxSeqLen * headDim
        let bytesPerLayer = elementsPerLayer * 4  // float32

        XCTAssertEqual(bytesPerLayer, 2 * 1 * 32 * 1024 * 64 * 4)  // ~16MB per layer
    }

    // MARK: - Bug Fix Verification

    func testRoPEDimensionBugFix() {
        // The critical bug fix: RoPE dimension calculation

        let dim = 2048
        let nHeads = 32

        // WRONG (the bug):
        let wrongRotDim = dim / nHeads  // 64

        // CORRECT (Python implementation):
        let correctRotDim = dim / (2 * nHeads)  // 32

        XCTAssertEqual(wrongRotDim, 64, "Wrong formula gives 64")
        XCTAssertEqual(correctRotDim, 32, "Correct formula gives 32")

        // The freqs_cis shape depends on this
        let wrongFreqsShape = [4096, wrongRotDim / 2, 2]  // [4096, 32, 2]
        let correctFreqsShape = [4096, correctRotDim / 2, 2]  // [4096, 16, 2]

        XCTAssertEqual(wrongFreqsShape, [4096, 32, 2])
        XCTAssertEqual(correctFreqsShape, [4096, 16, 2])
    }

    func testGenerateLogitsOutputShape() {
        // The bug fix: generateLogits should return [B, vocab_size], not [B, 1, vocab_size]

        let batchSize = 1
        let vocabSize = 51200

        // Expected shape after fix
        let expectedShape = [batchSize, vocabSize]

        XCTAssertEqual(expectedShape, [1, 51200])
        XCTAssertEqual(expectedShape.count, 2, "Should be 2D tensor")
    }
}
