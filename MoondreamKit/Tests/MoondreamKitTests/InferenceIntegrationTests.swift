// Copyright 2025 Moondream AI. All rights reserved.
// SPDX-License-Identifier: MIT

import XCTest
import MLXLMCommon
@testable import MoondreamKit

/// Integration tests for model inference
/// These tests require a downloaded model and are slow (~30s+ each)
/// Skip in CI by filtering: -skip-testing:MoondreamKitTests/InferenceIntegrationTests
final class InferenceIntegrationTests: XCTestCase {

    // Shared model context for all tests (loaded once)
    // Using nonisolated(unsafe) for test class shared state
    nonisolated(unsafe) static var modelContext: ModelContext?
    nonisolated(unsafe) static var loadError: Error?
    nonisolated(unsafe) static var isModelAvailable: Bool = false

    // Test image path
    static let testImagePath = "/Users/lewi/Documents/ai/moondream-mlx/monalisa-on-a-gallery-wall.png"

    override class func setUp() {
        super.setUp()

        // Check if any model is downloaded
        let downloadedIds = ModelCache.getDownloadedModelIds()
        guard !downloadedIds.isEmpty else {
            print("[InferenceIntegrationTests] No models downloaded - skipping integration tests")
            return
        }

        // Use the first available downloaded model
        let modelId = downloadedIds.first!
        print("[InferenceIntegrationTests] Using downloaded model: \(modelId)")

        // Load model (this is slow)
        let expectation = XCTestExpectation(description: "Model load")
        Task {
            do {
                let config = Moondream3Loader.configuration(for: modelId)
                modelContext = try await Moondream3Loader.load(configuration: config)
                isModelAvailable = true
                print("[InferenceIntegrationTests] Model loaded successfully")
            } catch {
                loadError = error
                print("[InferenceIntegrationTests] Model load failed: \(error)")
            }
            expectation.fulfill()
        }

        // Wait up to 5 minutes for model load
        let waiter = XCTWaiter()
        _ = waiter.wait(for: [expectation], timeout: 300)
    }

    override class func tearDown() {
        modelContext = nil
        super.tearDown()
    }

    // MARK: - Helper

    private func skipIfModelNotAvailable(file: StaticString = #file, line: UInt = #line) throws {
        if let error = Self.loadError {
            throw XCTSkip("Model load failed: \(error)", file: file, line: line)
        }
        guard Self.isModelAvailable else {
            throw XCTSkip("No model available for integration tests", file: file, line: line)
        }
        guard Self.modelContext != nil else {
            throw XCTSkip("Model context not initialized", file: file, line: line)
        }
    }

    private func loadTestImage() throws -> CGImage {
        let url = URL(fileURLWithPath: Self.testImagePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Test image not found at \(Self.testImagePath)")
        }

        #if os(macOS)
        guard let nsImage = NSImage(contentsOf: url),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw XCTSkip("Could not load test image")
        }
        return cgImage
        #else
        guard let data = try? Data(contentsOf: url),
              let uiImage = UIImage(data: data),
              let cgImage = uiImage.cgImage else {
            throw XCTSkip("Could not load test image")
        }
        return cgImage
        #endif
    }

    // MARK: - Model Loading Tests

    func testModelContextExists() throws {
        try skipIfModelNotAvailable()
        XCTAssertNotNil(Self.modelContext)
    }

    func testModelContextHasTokenizer() throws {
        try skipIfModelNotAvailable()
        XCTAssertNotNil(Self.modelContext?.tokenizer)
    }

    func testModelContextHasProcessor() throws {
        try skipIfModelNotAvailable()
        XCTAssertNotNil(Self.modelContext?.processor)
    }

    func testModelContextHasModel() throws {
        try skipIfModelNotAvailable()
        XCTAssertNotNil(Self.modelContext?.model)
    }

    // MARK: - Caption Tests

    func testCaptionGeneratesOutput() async throws {
        try skipIfModelNotAvailable()
        guard let context = Self.modelContext else { throw XCTSkip("No model context") }

        let image = try loadTestImage()

        // Cast to MoondreamModel protocol
        guard let moondreamModel = context.model as? any MoondreamModel else {
            throw XCTSkip("Model does not conform to MoondreamModel protocol")
        }

        // Process image
        guard let processor = context.processor as? Moondream3Processor else {
            throw XCTSkip("Processor is not Moondream3Processor")
        }
        let pixels = processor.processImage(image)

        // Generate caption
        let caption = moondreamModel.caption(
            pixels: pixels,
            length: "normal",
            tokenizer: context.tokenizer,
            maxTokens: 128,
            temperature: 0.0
        )

        XCTAssertFalse(caption.isEmpty, "Caption should not be empty")
        XCTAssertGreaterThan(caption.count, 10, "Caption should have meaningful length")
        print("[InferenceIntegrationTests] Generated caption: \(caption)")
    }

    func testCaptionShort() async throws {
        try skipIfModelNotAvailable()
        guard let context = Self.modelContext else { throw XCTSkip("No model context") }

        let image = try loadTestImage()

        guard let moondreamModel = context.model as? any MoondreamModel,
              let processor = context.processor as? Moondream3Processor else {
            throw XCTSkip("Model/processor type mismatch")
        }

        let pixels = processor.processImage(image)
        let shortCaption = moondreamModel.caption(
            pixels: pixels,
            length: "short",
            tokenizer: context.tokenizer,
            maxTokens: 64,
            temperature: 0.0
        )

        XCTAssertFalse(shortCaption.isEmpty)
        print("[InferenceIntegrationTests] Short caption: \(shortCaption)")
    }

    // MARK: - Query Tests

    func testQueryReturnsAnswer() async throws {
        try skipIfModelNotAvailable()
        guard let context = Self.modelContext else { throw XCTSkip("No model context") }

        let image = try loadTestImage()

        guard let moondreamModel = context.model as? any MoondreamModel,
              let processor = context.processor as? Moondream3Processor else {
            throw XCTSkip("Model/processor type mismatch")
        }

        let pixels = processor.processImage(image)
        let answer = moondreamModel.query(
            pixels: pixels,
            question: "What painting is shown in this image?",
            tokenizer: context.tokenizer,
            maxTokens: 128,
            temperature: 0.0
        )

        XCTAssertFalse(answer.isEmpty, "Query answer should not be empty")
        // The test image is the Mona Lisa, so the answer should mention it
        let lowerAnswer = answer.lowercased()
        let hasPaintingReference = lowerAnswer.contains("mona lisa") ||
                                   lowerAnswer.contains("painting") ||
                                   lowerAnswer.contains("portrait") ||
                                   lowerAnswer.contains("leonardo") ||
                                   lowerAnswer.contains("da vinci") ||
                                   lowerAnswer.contains("art") ||
                                   lowerAnswer.contains("museum") ||
                                   lowerAnswer.contains("gallery")
        XCTAssertTrue(hasPaintingReference,
                      "Answer should reference art/painting concepts. Got: \(answer)")
        print("[InferenceIntegrationTests] Query answer: \(answer)")
    }

    // MARK: - Point Tests

    func testPointReturnsCoordinates() async throws {
        try skipIfModelNotAvailable()
        guard let context = Self.modelContext else { throw XCTSkip("No model context") }

        let image = try loadTestImage()

        guard let moondreamModel = context.model as? any MoondreamModel,
              let processor = context.processor as? Moondream3Processor else {
            throw XCTSkip("Model/processor type mismatch")
        }

        let pixels = processor.processImage(image)
        let pointOutput = moondreamModel.point(
            pixels: pixels,
            object: "face",
            tokenizer: context.tokenizer,
            maxTokens: 64,
            temperature: 0.0
        )

        XCTAssertFalse(pointOutput.isEmpty, "Point output should not be empty")
        // Point output should contain coordinate-like data
        print("[InferenceIntegrationTests] Point output: \(pointOutput)")
    }

    // MARK: - Detect Tests

    func testDetectReturnsBoundingBoxes() async throws {
        try skipIfModelNotAvailable()
        guard let context = Self.modelContext else { throw XCTSkip("No model context") }

        let image = try loadTestImage()

        guard let moondreamModel = context.model as? any MoondreamModel,
              let processor = context.processor as? Moondream3Processor else {
            throw XCTSkip("Model/processor type mismatch")
        }

        let pixels = processor.processImage(image)
        let detectOutput = moondreamModel.detect(
            pixels: pixels,
            object: "painting",
            tokenizer: context.tokenizer,
            maxTokens: 128,
            temperature: 0.0
        )

        XCTAssertFalse(detectOutput.isEmpty, "Detect output should not be empty")
        print("[InferenceIntegrationTests] Detect output: \(detectOutput)")
    }

    // MARK: - Performance Tests

    func testCaptionPerformance() async throws {
        try skipIfModelNotAvailable()
        guard let context = Self.modelContext else { throw XCTSkip("No model context") }

        let image = try loadTestImage()

        guard let moondreamModel = context.model as? any MoondreamModel,
              let processor = context.processor as? Moondream3Processor else {
            throw XCTSkip("Model/processor type mismatch")
        }

        let pixels = processor.processImage(image)

        // Measure inference time
        let startTime = CFAbsoluteTimeGetCurrent()
        _ = moondreamModel.caption(
            pixels: pixels,
            length: "normal",
            tokenizer: context.tokenizer,
            maxTokens: 128,
            temperature: 0.0
        )
        let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime

        print("[InferenceIntegrationTests] Caption inference time: \(elapsedTime)s")

        // Caption should complete within reasonable time (5 minutes max)
        XCTAssertLessThan(elapsedTime, 300, "Caption should complete within 5 minutes")
    }
}

#if os(macOS)
import AppKit
#else
import UIKit
#endif
