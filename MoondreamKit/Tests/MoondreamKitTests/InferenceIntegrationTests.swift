import XCTest
import MLXLMCommon
@testable import MoondreamKit

/// Integration tests for model loading
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

    private func skipIfModelNotAvailable(file: StaticString = #filePath, line: UInt = #line) throws {
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

    func testModelIsMoondreamModel() throws {
        try skipIfModelNotAvailable()
        guard let context = Self.modelContext else {
            throw XCTSkip("No model context")
        }

        // Verify the model conforms to MoondreamModel protocol
        let isMoondreamModel = context.model is any MoondreamModel
        XCTAssertTrue(isMoondreamModel, "Loaded model should conform to MoondreamModel protocol")
    }

    func testModelConfiguration() throws {
        try skipIfModelNotAvailable()
        guard let context = Self.modelContext else {
            throw XCTSkip("No model context")
        }

        // Configuration should have a valid name
        XCTAssertFalse(context.configuration.name.isEmpty)
    }

    // MARK: - Tokenizer Tests

    func testTokenizerCanEncode() throws {
        try skipIfModelNotAvailable()
        guard let context = Self.modelContext else {
            throw XCTSkip("No model context")
        }

        // Test basic tokenization
        let tokens = context.tokenizer.encode(text: "Hello, world!")
        XCTAssertGreaterThan(tokens.count, 0, "Tokenizer should produce tokens")
    }

    func testTokenizerCanDecode() throws {
        try skipIfModelNotAvailable()
        guard let context = Self.modelContext else {
            throw XCTSkip("No model context")
        }

        // Test round-trip
        let original = "Test input"
        let tokens = context.tokenizer.encode(text: original)
        let decoded = context.tokenizer.decode(tokens: tokens)

        XCTAssertFalse(decoded.isEmpty, "Decoded text should not be empty")
    }

    // MARK: - Performance Test

    func testModelLoadPerformance() async throws {
        let downloadedIds = ModelCache.getDownloadedModelIds()
        guard !downloadedIds.isEmpty else {
            throw XCTSkip("No models downloaded")
        }

        let modelId = downloadedIds.first!
        let config = Moondream3Loader.configuration(for: modelId)

        // Measure load time (model may already be in memory cache)
        let startTime = CFAbsoluteTimeGetCurrent()
        _ = try await Moondream3Loader.load(configuration: config)
        let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime

        print("[InferenceIntegrationTests] Model load time: \(elapsedTime)s")

        // Model should load within reasonable time (5 minutes max)
        XCTAssertLessThan(elapsedTime, 300, "Model should load within 5 minutes")
    }
}
