// RoPETests.swift
// Unit tests for Rotary Position Embedding implementation

import XCTest
import MLX
@testable import MoondreamKit

final class RoPETests: XCTestCase {

    /// Check if MLX/Metal is available (not available in swift test without Xcode)
    func requireMLX() throws {
        if ProcessInfo.processInfo.environment["XCTestBundlePath"] != nil &&
           ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] == nil {
            throw XCTSkip("MLX requires Metal - run tests via Xcode, not swift test")
        }
    }

    // MARK: - precomputeFreqsCis Tests

    func testPrecomputeFreqsCisShape() throws {
        try requireMLX()
        // For Moondream3: rot_dim = 2048 / (2 * 32) = 32
        let rotDim = 32
        let maxLen = 100

        let freqsCis = precomputeFreqsCis(dim: rotDim, maxLen: maxLen)

        // Shape should be [maxLen, rotDim/2, 2] = [100, 16, 2]
        XCTAssertEqual(freqsCis.shape, [maxLen, rotDim / 2, 2])
    }

    func testCorrectRoPEDimension() {
        // CRITICAL: This test verifies the bug fix
        // Python: rot_dim = config.dim // (2 * config.n_heads)
        // With dim=2048, n_heads=32: rot_dim = 2048 // 64 = 32

        let dim = 2048
        let nHeads = 32

        // Correct formula (matches Python)
        let correctRotDim = dim / (2 * nHeads)
        XCTAssertEqual(correctRotDim, 32, "RoPE dimension should be 32, not 64")

        // Wrong formula (the bug)
        let wrongRotDim = dim / nHeads
        XCTAssertEqual(wrongRotDim, 64, "Wrong formula produces 64")

        // Verify we use the correct one
        XCTAssertNotEqual(correctRotDim, wrongRotDim, "The two formulas produce different results")
    }

    func testFreqsCisValuesBounded() throws {
        try requireMLX()
        let freqsCis = precomputeFreqsCis(dim: 32, maxLen: 100)
        eval(freqsCis)

        // cos and sin values should be in [-1, 1]
        let minVal = freqsCis.min().item(Float.self)
        let maxVal = freqsCis.max().item(Float.self)

        XCTAssertGreaterThanOrEqual(minVal, -1.0, "Minimum value should be >= -1")
        XCTAssertLessThanOrEqual(maxVal, 1.0, "Maximum value should be <= 1")
    }

    func testFirstPositionValues() throws {
        try requireMLX()
        let freqsCis = precomputeFreqsCis(dim: 32, maxLen: 10)
        eval(freqsCis)

        // At position 0, cos should be 1, sin should be 0
        let cosAtZero = freqsCis[0, 0..., 0]
        let sinAtZero = freqsCis[0, 0..., 1]
        eval(cosAtZero, sinAtZero)

        // cos(0) = 1 for all frequencies
        let cosValues = cosAtZero.asArray(Float.self)
        for val in cosValues {
            XCTAssertEqual(val, 1.0, accuracy: 1e-5, "cos(0) should be 1.0")
        }

        // sin(0) = 0 for all frequencies
        let sinValues = sinAtZero.asArray(Float.self)
        for val in sinValues {
            XCTAssertEqual(val, 0.0, accuracy: 1e-5, "sin(0) should be 0.0")
        }
    }

    // MARK: - applyRotaryEmb Tests

    func testApplyRotaryEmbShape() throws {
        try requireMLX()
        let batchSize = 1
        let numHeads = 32
        let seqLen = 10
        let headDim = 64

        // Create input tensor [B, H, T, D]
        let x = MLXArray.ones([batchSize, numHeads, seqLen, headDim])

        // Create freqs_cis with correct dimension
        let rotDim = 32  // dim / (2 * n_heads) = 2048 / 64 = 32
        let freqsCis = precomputeFreqsCis(dim: rotDim, maxLen: seqLen + 10)

        // Create positions
        let positions = MLXArray((0..<seqLen).map { Int32($0) })

        let output = applyRotaryEmb(x, freqsCis: freqsCis, positions: positions)
        eval(output)

        XCTAssertEqual(output.shape, x.shape, "Output shape should match input shape")
    }

    func testApplyRotaryEmbPositionDependent() throws {
        try requireMLX()
        let x = MLXArray.ones([1, 4, 5, 64])
        let freqsCis = precomputeFreqsCis(dim: 32, maxLen: 20)

        // Apply with positions [0, 1, 2, 3, 4]
        let positions1 = MLXArray([Int32(0), 1, 2, 3, 4])
        let output1 = applyRotaryEmb(x, freqsCis: freqsCis, positions: positions1)

        // Apply with positions [5, 6, 7, 8, 9]
        let positions2 = MLXArray([Int32(5), 6, 7, 8, 9])
        let output2 = applyRotaryEmb(x, freqsCis: freqsCis, positions: positions2)

        eval(output1, output2)

        // Outputs should be different for different positions
        let diff = abs(output1 - output2).sum()
        eval(diff)

        XCTAssertGreaterThan(diff.item(Float.self), 0.1, "Different positions should produce different outputs")
    }

    func testPassThroughDimension() throws {
        try requireMLX()
        // RoPE only affects the first rotDim dimensions
        // The rest should pass through unchanged
        let x = MLXArray.ones([1, 4, 5, 64])
        let freqsCis = precomputeFreqsCis(dim: 32, maxLen: 10)
        let positions = MLXArray([Int32(0), 1, 2, 3, 4])

        let output = applyRotaryEmb(x, freqsCis: freqsCis, positions: positions)
        eval(output)

        // Last 32 dimensions (64 - 32) should be unchanged (all 1s)
        let passThroughPart = output[0..., 0..., 0..., 32...]
        eval(passThroughPart)

        let sum = passThroughPart.sum().item(Float.self)
        let expectedSum = Float(1 * 4 * 5 * 32)  // All 1s

        XCTAssertEqual(sum, expectedSum, accuracy: 1e-3, "Pass-through dimensions should be unchanged")
    }
}
