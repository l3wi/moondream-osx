import Foundation
import MLX

/// KV Cache manager for efficient attention computation
/// Handles cache allocation and updates for both standard and quantized models
internal enum KVCacheManager {

    // MARK: - Cache Allocation

    /// Allocate a simple KV cache for generation
    /// - Parameters:
    ///   - layers: Number of transformer layers
    ///   - kvHeads: Number of key-value heads
    ///   - headDim: Dimension of each attention head
    ///   - maxSeqLen: Maximum sequence length (defaults to platform-specific value)
    ///   - batchSize: Batch size (default 1)
    /// - Returns: Array of (K, V) cache tuples, one per layer
    static func allocateCache(
        layers: Int,
        kvHeads: Int,
        headDim: Int,
        maxSeqLen: Int? = nil,
        batchSize: Int = 1
    ) -> [(MLXArray, MLXArray)] {
        let seqLen = maxSeqLen ?? PlatformConfiguration.maxCacheSeqLen

        md3Log("Allocating KV cache: \(layers) layers, \(kvHeads) heads, \(seqLen) seq, \(headDim) dim")

        var cache: [(MLXArray, MLXArray)] = []

        for _ in 0..<layers {
            let k = MLXArray.zeros([batchSize, kvHeads, seqLen, headDim])
            let v = MLXArray.zeros([batchSize, kvHeads, seqLen, headDim])
            cache.append((k, v))
        }

        return cache
    }

    /// Allocate a nested KV cache structure (for MLXLMCommon compatibility)
    /// - Parameters:
    ///   - layers: Number of transformer layers
    ///   - kvHeads: Number of key-value heads
    ///   - headDim: Dimension of each attention head
    ///   - maxSeqLen: Maximum sequence length
    ///   - batchSize: Batch size (default 1)
    /// - Returns: Nested array of (K, V) cache tuples
    static func allocateNestedCache(
        layers: Int,
        kvHeads: Int,
        headDim: Int,
        maxSeqLen: Int = 1024,
        batchSize: Int = 1
    ) -> [[(MLXArray, MLXArray)]] {
        var cache: [[(MLXArray, MLXArray)]] = []

        for _ in 0..<layers {
            let k = MLXArray.zeros([batchSize, kvHeads, maxSeqLen, headDim])
            let v = MLXArray.zeros([batchSize, kvHeads, maxSeqLen, headDim])
            cache.append([(k, v)])
        }

        return cache
    }

    // MARK: - Cache Updates

    /// Update KV cache with new keys and values
    /// This is the core caching logic that both model variants use
    /// - Parameters:
    ///   - keys: New key tensor to insert
    ///   - values: New value tensor to insert
    ///   - cache: Existing cache tuple (K, V)
    ///   - cachePos: Position in cache to insert at
    /// - Returns: Tuple of (updatedCache, keysForAttention, valuesForAttention)
    static func updateCache(
        keys: MLXArray,
        values: MLXArray,
        cache: (MLXArray, MLXArray)?,
        cachePos: Int
    ) -> (cache: (MLXArray, MLXArray), keys: MLXArray, values: MLXArray) {
        guard let (kCache, vCache) = cache else {
            // No cache - return keys/values as-is
            return ((keys, values), keys, values)
        }

        let seqLen = keys.dim(2)  // New sequence length
        let maxLen = kCache.dim(2)  // Pre-allocated max length
        let newEnd = cachePos + seqLen

        var updatedKCache: MLXArray
        var updatedVCache: MLXArray

        // Build updated cache by inserting new K/V at correct position
        // Python: k_cache = k_cache.at[:, :, cache_pos:cache_pos+T, :].add(k)
        if cachePos > 0 && newEnd < maxLen {
            // Insert in middle: [before, new, after]
            let beforeK = kCache[0..., 0..., ..<cachePos, 0...]
            let afterK = kCache[0..., 0..., newEnd..., 0...]
            updatedKCache = concatenated([beforeK, keys, afterK], axis: 2)

            let beforeV = vCache[0..., 0..., ..<cachePos, 0...]
            let afterV = vCache[0..., 0..., newEnd..., 0...]
            updatedVCache = concatenated([beforeV, values, afterV], axis: 2)
        } else if cachePos == 0 && newEnd < maxLen {
            // Insert at start: [new, after]
            let afterK = kCache[0..., 0..., newEnd..., 0...]
            updatedKCache = concatenated([keys, afterK], axis: 2)

            let afterV = vCache[0..., 0..., newEnd..., 0...]
            updatedVCache = concatenated([values, afterV], axis: 2)
        } else if cachePos > 0 && newEnd >= maxLen {
            // Insert at end: [before, new]
            let beforeK = kCache[0..., 0..., ..<cachePos, 0...]
            updatedKCache = concatenated([beforeK, keys], axis: 2)

            let beforeV = vCache[0..., 0..., ..<cachePos, 0...]
            updatedVCache = concatenated([beforeV, values], axis: 2)
        } else {
            // cachePos == 0 && newEnd >= maxLen: just use new keys/values
            updatedKCache = keys
            updatedVCache = values
        }

        // For attention, use only valid entries up to newEnd
        // Python: k = k_cache[:, :, :cache_pos + T, :]
        let validLen = min(newEnd, updatedKCache.dim(2))
        let keysForAttn = updatedKCache[0..., 0..., ..<validLen, 0...]
        let valuesForAttn = updatedVCache[0..., 0..., ..<validLen, 0...]

        return ((updatedKCache, updatedVCache), keysForAttn, valuesForAttn)
    }
}
