// Copyright 2024 Moondream AI
// Platform-specific configuration for MoondreamKit

import Foundation

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
