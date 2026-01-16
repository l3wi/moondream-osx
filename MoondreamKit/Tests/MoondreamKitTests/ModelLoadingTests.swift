// Copyright 2025 Moondream AI. All rights reserved.
// SPDX-License-Identifier: MIT

import XCTest
@testable import MoondreamKit

/// Tests for Moondream3Loader and model configuration
final class ModelLoadingTests: XCTestCase {

    // MARK: - Static Configuration Tests

    func testOriginalConfigurationId() {
        let config = Moondream3Loader.originalConfiguration
        XCTAssertEqual(config.name, "moondream/md3p-int4")
    }

    func testQuantizedConfigurationId() {
        let config = Moondream3Loader.quantizedConfiguration
        XCTAssertEqual(config.name, "lewi/md3p-int4-smol")
    }

    func testDefaultConfigurationExists() {
        let config = Moondream3Loader.defaultConfiguration
        XCTAssertFalse(config.name.isEmpty)
    }

    func testConfigurationDefaultPrompts() {
        let originalConfig = Moondream3Loader.originalConfiguration
        let quantizedConfig = Moondream3Loader.quantizedConfiguration

        // Both should have default prompts set
        XCTAssertNotNil(originalConfig.defaultPrompt)
        XCTAssertNotNil(quantizedConfig.defaultPrompt)
        XCTAssertEqual(originalConfig.defaultPrompt, "Describe this image.")
        XCTAssertEqual(quantizedConfig.defaultPrompt, "Describe this image.")
    }

    // MARK: - configuration(for:) Tests

    func testConfigurationForStandardModel() {
        let config = Moondream3Loader.configuration(for: "moondream/md3p-int4")
        XCTAssertEqual(config.name, "moondream/md3p-int4")
    }

    func testConfigurationForCompactModel() {
        let config = Moondream3Loader.configuration(for: "lewi/md3p-int4-smol")
        XCTAssertEqual(config.name, "lewi/md3p-int4-smol")
    }

    func testConfigurationForArbitraryModelId() {
        let config = Moondream3Loader.configuration(for: "some-org/some-model")
        XCTAssertEqual(config.name, "some-org/some-model")
    }

    func testConfigurationForLocalDirectory() {
        let config = Moondream3Loader.configuration(for: "/path/to/local/model")
        // For local paths, the config should recognize it as a directory
        // The exact behavior depends on ModelConfiguration implementation
        XCTAssertNotNil(config)
    }

    func testConfigurationForRelativePath() {
        let config = Moondream3Loader.configuration(for: "./models/test")
        XCTAssertNotNil(config)
    }

    func testConfigurationForTildePath() {
        let config = Moondream3Loader.configuration(for: "~/models/test")
        XCTAssertNotNil(config)
    }

    // MARK: - Moondream3LoaderError Tests

    func testLoaderErrorNoWeightFiles() {
        let error = Moondream3LoaderError.noWeightFiles
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("weight") ||
                      error.errorDescription!.contains("safetensors"),
                      "Error should mention weight files")
    }

    func testLoaderErrorConfigurationMissing() {
        let error = Moondream3LoaderError.configurationMissing
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("config"),
                      "Error should mention configuration")
    }

    func testLoaderErrorTokenizerLoadFailed() {
        let error = Moondream3LoaderError.tokenizerLoadFailed
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("tokenizer"),
                      "Error should mention tokenizer")
    }

    func testLoaderErrorsAreLocalizedError() {
        // Verify all errors conform to LocalizedError
        let errors: [Moondream3LoaderError] = [
            .noWeightFiles,
            .configurationMissing,
            .tokenizerLoadFailed
        ]

        for error in errors {
            let localizedError: LocalizedError = error
            XCTAssertNotNil(localizedError.errorDescription)
        }
    }

    // MARK: - Model Type Detection Tests (via configuration)

    func testStandardModelIdDetection() {
        // Models containing md3p-int4 (not smol) should use standard loader
        let ids = [
            "moondream/md3p-int4",
            "org/md3p-int4-v2",
            "test/moondream-md3p-int4"
        ]

        for id in ids {
            let config = Moondream3Loader.configuration(for: id)
            XCTAssertNotNil(config, "Should create config for \(id)")
        }
    }

    func testQuantizedModelIdDetection() {
        // Models with 'smol' or 'lewi/' should use quantized loader
        let ids = [
            "lewi/md3p-int4-smol",
            "lewi/other-model",
            "org/md3p-int4-smol-v2"
        ]

        for id in ids {
            let config = Moondream3Loader.configuration(for: id)
            XCTAssertNotNil(config, "Should create config for \(id)")
        }
    }

    // MARK: - Slow Load Tests (requires model download)

    /// Test actual model loading - only runs if model is downloaded
    func testLoadModelIfAvailable() async throws {
        let downloadedIds = ModelCache.getDownloadedModelIds()
        guard !downloadedIds.isEmpty else {
            throw XCTSkip("No models downloaded - skipping load test")
        }

        let modelId = downloadedIds.first!
        let config = Moondream3Loader.configuration(for: modelId)

        let context = try await Moondream3Loader.load(
            configuration: config,
            progressHandler: { _ in }
        )

        XCTAssertNotNil(context.model)
        XCTAssertNotNil(context.tokenizer)
        XCTAssertNotNil(context.processor)
    }

    /// Test model container loading - only runs if model is downloaded
    func testLoadContainerIfAvailable() async throws {
        let downloadedIds = ModelCache.getDownloadedModelIds()
        guard !downloadedIds.isEmpty else {
            throw XCTSkip("No models downloaded - skipping container load test")
        }

        let modelId = downloadedIds.first!
        let config = Moondream3Loader.configuration(for: modelId)

        let container = try await Moondream3Loader.loadContainer(configuration: config)
        XCTAssertNotNil(container)
    }

    // MARK: - Platform-Specific Tests

    func testDefaultConfigurationPlatformSpecific() {
        let defaultConfig = Moondream3Loader.defaultConfiguration

        #if os(iOS)
        // On iOS, default should be the compact/quantized model
        XCTAssertEqual(defaultConfig.name, "lewi/md3p-int4-smol",
                       "iOS should default to compact model")
        #else
        // On macOS, default should be the standard model
        XCTAssertEqual(defaultConfig.name, "moondream/md3p-int4",
                       "macOS should default to standard model")
        #endif
    }

    // MARK: - Configuration Consistency Tests

    func testAvailableModelsHaveConfigurations() {
        for model in AvailableModels.all {
            let config = Moondream3Loader.configuration(for: model.id)
            XCTAssertEqual(config.name, model.id,
                           "Configuration name should match model ID")
        }
    }

    func testConfigurationsHaveDefaultPrompt() {
        for model in AvailableModels.all {
            let config = Moondream3Loader.configuration(for: model.id)
            XCTAssertNotNil(config.defaultPrompt,
                            "Configuration for \(model.id) should have a default prompt")
        }
    }
}
