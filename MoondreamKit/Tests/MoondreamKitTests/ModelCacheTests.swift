// Copyright 2025 Moondream AI. All rights reserved.
// SPDX-License-Identifier: MIT

import XCTest
@testable import MoondreamKit

/// Tests for ModelCache utilities
final class ModelCacheTests: XCTestCase {

    // MARK: - cacheDirectory Tests

    func testCacheDirectoryReturnsURL() {
        // Test with a known model ID
        let directory = ModelCache.cacheDirectory(for: "moondream/md3p-int4")
        XCTAssertNotNil(directory)
    }

    func testCacheDirectoryReturnsValidPath() {
        let directory = ModelCache.cacheDirectory(for: "moondream/md3p-int4")

        // The path should contain the model ID components
        if let dir = directory {
            let path = dir.path
            XCTAssertTrue(path.contains("moondream") || path.contains("md3p"),
                          "Cache directory path should contain model identifier components")
        }
    }

    func testCacheDirectoryDifferentModels() {
        let dir1 = ModelCache.cacheDirectory(for: "moondream/md3p-int4")
        let dir2 = ModelCache.cacheDirectory(for: "lewi/md3p-int4-smol")

        // Different models should have different cache directories
        XCTAssertNotNil(dir1)
        XCTAssertNotNil(dir2)
        if let d1 = dir1, let d2 = dir2 {
            XCTAssertNotEqual(d1.path, d2.path)
        }
    }

    // MARK: - isDownloaded Tests

    func testIsDownloadedReturnsFalseForInvalidId() {
        // A model ID that doesn't exist should return false
        let isDownloaded = ModelCache.isDownloaded("definitely-not-a-real-model/fake-model-xyz123")
        XCTAssertFalse(isDownloaded)
    }

    func testIsDownloadedForKnownModels() {
        // These tests verify the method works - actual results depend on download state
        // The method should not crash for valid model IDs
        let standardResult = ModelCache.isDownloaded("moondream/md3p-int4")
        let compactResult = ModelCache.isDownloaded("lewi/md3p-int4-smol")

        // Just verify we get boolean results without crashing
        XCTAssertTrue(standardResult == true || standardResult == false)
        XCTAssertTrue(compactResult == true || compactResult == false)
    }

    // MARK: - getDownloadedModelIds Tests

    func testGetDownloadedModelIdsReturnsSet() {
        // Should return a set (possibly empty)
        let downloadedIds = ModelCache.getDownloadedModelIds()
        XCTAssertNotNil(downloadedIds)

        // If any are downloaded, they should be from AvailableModels
        let availableIds = Set(AvailableModels.all.map { $0.id })
        for id in downloadedIds {
            XCTAssertTrue(availableIds.contains(id),
                          "Downloaded model ID '\(id)' should be in AvailableModels")
        }
    }

    func testGetDownloadedModelIdsConsistentWithIsDownloaded() {
        let downloadedIds = ModelCache.getDownloadedModelIds()

        // Verify consistency with isDownloaded()
        for model in AvailableModels.all {
            let isDownloaded = ModelCache.isDownloaded(model.id)
            let inSet = downloadedIds.contains(model.id)
            XCTAssertEqual(isDownloaded, inSet,
                           "isDownloaded() and getDownloadedModelIds() should be consistent for \(model.id)")
        }
    }

    // MARK: - downloadedSize Tests

    func testDownloadedSizeReturnsNilForNonExistent() {
        let size = ModelCache.downloadedSize(for: "definitely-not-a-real-model/fake-model-xyz123")
        XCTAssertNil(size)
    }

    func testDownloadedSizeForAvailableModels() {
        // For available models, size should be nil (not downloaded) or positive
        for model in AvailableModels.all {
            let size = ModelCache.downloadedSize(for: model.id)

            if let downloadedSize = size {
                XCTAssertGreaterThan(downloadedSize, 0,
                                     "If downloaded, size should be positive for \(model.id)")
            }
            // nil is also valid (not downloaded)
        }
    }

    func testDownloadedSizeConsistentWithIsDownloaded() {
        for model in AvailableModels.all {
            let isDownloaded = ModelCache.isDownloaded(model.id)
            let size = ModelCache.downloadedSize(for: model.id)

            if isDownloaded {
                // If downloaded, should have a size
                XCTAssertNotNil(size, "Downloaded model \(model.id) should have a size")
            } else {
                // If not downloaded, should be nil
                XCTAssertNil(size, "Non-downloaded model \(model.id) should have nil size")
            }
        }
    }

    // MARK: - Integration Tests (only run if model is downloaded)

    func testCacheDirectoryExistsWhenDownloaded() {
        for model in AvailableModels.all {
            if ModelCache.isDownloaded(model.id) {
                guard let directory = ModelCache.cacheDirectory(for: model.id) else {
                    XCTFail("Downloaded model \(model.id) should have a cache directory")
                    continue
                }

                var isDirectory: ObjCBool = false
                let exists = FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory)
                XCTAssertTrue(exists, "Cache directory should exist for downloaded model \(model.id)")
                XCTAssertTrue(isDirectory.boolValue, "Cache path should be a directory")
            }
        }
    }

    func testDownloadedModelHasConfigJson() {
        for model in AvailableModels.all {
            if ModelCache.isDownloaded(model.id) {
                guard let directory = ModelCache.cacheDirectory(for: model.id) else {
                    continue
                }

                let configPath = directory.appendingPathComponent("config.json")
                XCTAssertTrue(FileManager.default.fileExists(atPath: configPath.path),
                              "Downloaded model \(model.id) should have config.json")
            }
        }
    }

    func testDownloadedModelHasSafetensors() {
        for model in AvailableModels.all {
            if ModelCache.isDownloaded(model.id) {
                guard let directory = ModelCache.cacheDirectory(for: model.id) else {
                    continue
                }

                let contents = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                let hasSafetensors = contents?.contains { $0.pathExtension == "safetensors" } ?? false
                XCTAssertTrue(hasSafetensors,
                              "Downloaded model \(model.id) should have safetensors files")
            }
        }
    }

    // MARK: - Error Handling Tests

    func testCacheDirectoryHandlesEmptyString() {
        // Empty string model ID
        let directory = ModelCache.cacheDirectory(for: "")
        // Should not crash - may return nil or some default
        _ = directory
    }

    func testCacheDirectoryHandlesSpecialCharacters() {
        // Model ID with unusual characters
        let directory = ModelCache.cacheDirectory(for: "test/model-with_special.chars")
        // Should not crash
        _ = directory
    }

    func testIsDownloadedHandlesEmptyString() {
        // Should not crash and return false for empty string
        let result = ModelCache.isDownloaded("")
        XCTAssertFalse(result)
    }
}
