// KVCacheTests.swift
// Unit tests for KV Cache implementation

import XCTest
import MLX
@testable import MoondreamKit

final class KVCacheTests: XCTestCase {

    /// Check if MLX/Metal is available (not available in swift test without Xcode)
    func requireMLX() throws {
        if ProcessInfo.processInfo.environment["XCTestBundlePath"] != nil &&
           ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] == nil {
            throw XCTSkip("MLX requires Metal - run tests via Xcode, not swift test")
        }
    }

    // MARK: - Cache Allocation Tests

    func testCacheAllocationShape() throws {
        try requireMLX()
        // Moondream3 config: 24 layers, 32 kv_heads, head_dim=64
        let nLayers = 24
        let nKvHeads = 32
        let headDim = 64
        let batchSize = 1
        let maxSeqLen = 1024

        // Simulate cache allocation (matching allocateSimpleCache)
        var cache: [(MLXArray, MLXArray)] = []
        for _ in 0..<nLayers {
            let k = MLXArray.zeros([batchSize, nKvHeads, maxSeqLen, headDim])
            let v = MLXArray.zeros([batchSize, nKvHeads, maxSeqLen, headDim])
            cache.append((k, v))
        }

        XCTAssertEqual(cache.count, nLayers)
        XCTAssertEqual(cache[0].0.shape, [1, 32, 1024, 64])
        XCTAssertEqual(cache[0].1.shape, [1, 32, 1024, 64])
    }

    // MARK: - Cache Update Tests

    func testCacheUpdateAtZero() throws {
        try requireMLX()
        let batchSize = 1
        let nKvHeads = 4
        let maxSeqLen = 100
        let headDim = 16
        let seqLen = 10

        // Pre-allocated cache
        let kCache = MLXArray.zeros([batchSize, nKvHeads, maxSeqLen, headDim])

        // New keys to insert
        let newK = MLXArray.ones([batchSize, nKvHeads, seqLen, headDim])

        // Update at position 0 (matching Swift implementation)
        let cachePos = 0
        let T = seqLen
        let newEnd = cachePos + T

        // Insert at start: [new, after]
        let afterK = kCache[0..., 0..., newEnd..., 0...]
        let updatedKCache = concatenated([newK, afterK], axis: 2)

        eval(updatedKCache)

        // Verify shape is preserved
        XCTAssertEqual(updatedKCache.shape, kCache.shape, "Cache shape should be preserved")

        // Verify values at positions 0-9 are 1s
        let insertedK = updatedKCache[0..., 0..., ..<seqLen, 0...]
        eval(insertedK)
        let sumK = insertedK.sum().item(Float.self)
        let expectedSumK = Float(batchSize * nKvHeads * seqLen * headDim)
        XCTAssertEqual(sumK, expectedSumK, accuracy: 1e-3, "Inserted K values should be 1s")

        // Verify positions 10+ are still zeros
        let remainingK = updatedKCache[0..., 0..., seqLen..., 0...]
        eval(remainingK)
        let sumRemaining = remainingK.sum().item(Float.self)
        XCTAssertEqual(sumRemaining, 0.0, accuracy: 1e-3, "Remaining positions should be zeros")
    }

    func testCacheUpdateAtMiddle() throws {
        try requireMLX()
        let batchSize = 1
        let nKvHeads = 4
        let maxSeqLen = 100
        let headDim = 16

        // Pre-allocated cache with some existing values at positions 0-9
        var kCache = MLXArray.zeros([batchSize, nKvHeads, maxSeqLen, headDim])

        // Simulate existing cache content at positions 0-9
        let existingK = MLXArray.ones([batchSize, nKvHeads, 10, headDim])
        let existingAfter = kCache[0..., 0..., 10..., 0...]
        kCache = concatenated([existingK, existingAfter], axis: 2)

        // New keys to insert at position 10
        let newK = MLXArray.ones([batchSize, nKvHeads, 5, headDim]) * 3

        let cachePos = 10
        let T = 5
        let newEnd = cachePos + T

        // Insert in middle: [before, new, after]
        let beforeK = kCache[0..., 0..., ..<cachePos, 0...]
        let afterK = kCache[0..., 0..., newEnd..., 0...]
        let updatedKCache = concatenated([beforeK, newK, afterK], axis: 2)

        eval(updatedKCache)

        // Verify shape
        XCTAssertEqual(updatedKCache.shape, [batchSize, nKvHeads, maxSeqLen, headDim])

        // Verify positions 0-9 still have value 1
        let firstPart = updatedKCache[0..., 0..., ..<10, 0...]
        eval(firstPart)
        let sumFirst = firstPart.sum().item(Float.self)
        let expectedFirst = Float(batchSize * nKvHeads * 10 * headDim)
        XCTAssertEqual(sumFirst, expectedFirst, accuracy: 1e-3, "First part should have 1s")

        // Verify positions 10-14 have value 3
        let middlePart = updatedKCache[0..., 0..., 10..<15, 0...]
        eval(middlePart)
        let sumMiddle = middlePart.sum().item(Float.self)
        let expectedMiddle = Float(batchSize * nKvHeads * 5 * headDim) * 3
        XCTAssertEqual(sumMiddle, expectedMiddle, accuracy: 1e-3, "Middle part should have 3s")
    }

    func testValidCacheSlice() throws {
        try requireMLX()
        let batchSize = 1
        let nKvHeads = 4
        let maxSeqLen = 100
        let headDim = 16

        // Cache with values at positions 0-29
        let filledK = MLXArray.ones([batchSize, nKvHeads, 30, headDim])
        let zerosK = MLXArray.zeros([batchSize, nKvHeads, 70, headDim])
        let kCache = concatenated([filledK, zerosK], axis: 2)

        let cachePos = 30

        // Get valid slice for attention (Python: k = k_cache[:, :, :cache_pos + T, :])
        let T = 1  // Single token decode
        let validLen = min(cachePos + T, maxSeqLen)
        let validK = kCache[0..., 0..., ..<validLen, 0...]

        eval(validK)

        XCTAssertEqual(validK.shape, [batchSize, nKvHeads, 31, headDim])
    }

    // MARK: - Edge Cases

    func testCacheUpdateAtMax() throws {
        try requireMLX()
        let batchSize = 1
        let nKvHeads = 4
        let maxSeqLen = 100
        let headDim = 16

        // Cache almost full (positions 0-98 filled)
        let kCache = MLXArray.ones([batchSize, nKvHeads, maxSeqLen, headDim])

        let cachePos = 99
        // T = 1 for single token decode

        // New key to insert at position 99
        let newK = MLXArray.ones([batchSize, nKvHeads, 1, headDim]) * 5

        // Insert at end: [before, new]
        let beforeK = kCache[0..., 0..., ..<cachePos, 0...]
        let updatedKCache = concatenated([beforeK, newK], axis: 2)

        eval(updatedKCache)

        XCTAssertEqual(updatedKCache.shape, [batchSize, nKvHeads, maxSeqLen, headDim])

        // Verify last position has value 5
        let lastPos = updatedKCache[0..., 0..., 99..<100, 0...]
        eval(lastPos)
        let sumLast = lastPos.sum().item(Float.self)
        let expectedLast = Float(batchSize * nKvHeads * 1 * headDim) * 5
        XCTAssertEqual(sumLast, expectedLast, accuracy: 1e-3, "Last position should have 5s")
    }

    func testCachePreservesDtype() throws {
        try requireMLX()
        let kCache = MLXArray.zeros([1, 4, 100, 16])
        let newK = MLXArray.ones([1, 4, 10, 16])

        XCTAssertEqual(kCache.dtype, .float32)
        XCTAssertEqual(newK.dtype, .float32)

        let afterK = kCache[0..., 0..., 10..., 0...]
        let updated = concatenated([newK, afterK], axis: 2)

        XCTAssertEqual(updated.dtype, .float32, "Cache dtype should be preserved")
    }
}
