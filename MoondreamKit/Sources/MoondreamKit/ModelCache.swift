// Copyright 2025 Anthropic. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Hub

/// Utilities for managing the model cache
public enum ModelCache {

    /// Check if a model is downloaded and ready to use
    /// - Parameter modelId: The HuggingFace model ID (e.g., "moondream/md3p-int4")
    /// - Returns: true if the model is fully downloaded
    public static func isDownloaded(_ modelId: String) -> Bool {
        guard let directory = cacheDirectory(for: modelId) else {
            return false
        }

        let fileManager = FileManager.default

        // Check for config.json
        let configPath = directory.appendingPathComponent("config.json")
        guard fileManager.fileExists(atPath: configPath.path) else {
            return false
        }

        // Check for at least one safetensors file
        guard let contents = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ) else {
            return false
        }

        return contents.contains { $0.pathExtension == "safetensors" }
    }

    /// Get all downloaded model IDs from the available models list
    /// - Returns: Set of model IDs that are fully downloaded
    public static func getDownloadedModelIds() -> Set<String> {
        var downloaded = Set<String>()
        for model in AvailableModels.all {
            if isDownloaded(model.id) {
                downloaded.insert(model.id)
            }
        }
        return downloaded
    }

    /// Delete a model from the cache
    /// - Parameter modelId: The HuggingFace model ID
    /// - Throws: File system errors if deletion fails
    public static func deleteModel(_ modelId: String) throws {
        guard let directory = cacheDirectory(for: modelId) else {
            return
        }

        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: directory.path) {
            try fileManager.removeItem(at: directory)
        }
    }

    /// Get the cache directory URL for a model
    /// - Parameter modelId: The HuggingFace model ID
    /// - Returns: The local cache directory URL, or nil if not determinable
    public static func cacheDirectory(for modelId: String) -> URL? {
        let hub = HubApi()
        let repo = Hub.Repo(id: modelId)
        return hub.localRepoLocation(repo)
    }

    /// Get the size of a downloaded model in bytes
    /// - Parameter modelId: The HuggingFace model ID
    /// - Returns: Size in bytes, or nil if not downloaded
    public static func downloadedSize(for modelId: String) -> Int64? {
        guard let directory = cacheDirectory(for: modelId),
              isDownloaded(modelId) else {
            return nil
        }

        let fileManager = FileManager.default
        guard let contents = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey]
        ) else {
            return nil
        }

        var totalSize: Int64 = 0
        for url in contents {
            if let resourceValues = try? url.resourceValues(forKeys: [.fileSizeKey]),
               let size = resourceValues.fileSize {
                totalSize += Int64(size)
            }
        }

        return totalSize
    }
}
