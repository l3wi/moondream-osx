import Foundation
import CoreImage
import MLX
import MLXNN
import MLXVLM
import MLXLMCommon
import MoondreamKit
import os.log

private let logger = Logger(subsystem: "com.moondream.mac", category: "MoondreamService")

// Minimal token limits for iOS to conserve memory
#if os(iOS)
private let maxGenerationTokens = 128  // Very short responses only
private let maxDetectionTokens = 32    // Minimal detection
#else
private let maxGenerationTokens = 768
private let maxDetectionTokens = 256
#endif

// Configure MLX memory management for iOS
#if os(iOS)
private func configureMLXMemory() {
    // Set extremely aggressive cache limit - 20MB as recommended by mlx-swift-examples
    // This forces MLX to release buffers immediately after use
    GPU.set(cacheLimit: 20 * 1024 * 1024)  // 20MB cache limit

    // Set memory limit just under iOS threshold to trigger earlier cleanup
    // 5.5GB to stay safely under the 6GB hard limit
    GPU.set(memoryLimit: Int(5.5 * 1024 * 1024 * 1024), relaxed: true)

    fileLog("MLX GPU: cache limit=20MB, memory limit=5.5GB (relaxed)")
}
#endif

// File-based logging for iOS debugging (since console logs are hard to capture on device)
private func fileLog(_ message: String) {
    logger.info("\(message)")
    #if os(iOS)
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "HH:mm:ss.SSS"
    let timestamp = dateFormatter.string(from: Date())
    let logMessage = "[\(timestamp)] [MoondreamService] \(message)\n"

    let fileManager = FileManager.default
    if let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
        let logFile = documentsDir.appendingPathComponent("moondream_inference.log")
        if let data = logMessage.data(using: .utf8) {
            if fileManager.fileExists(atPath: logFile.path) {
                if let handle = try? FileHandle(forWritingTo: logFile) {
                    handle.seekToEndOfFile()
                    handle.write(data)
                    handle.closeFile()
                }
            } else {
                try? data.write(to: logFile)
            }
        }
    }
    #endif
}

/// Service for loading and running inference with the Moondream model
@MainActor
final class MoondreamService: ObservableObject {
    static let shared = MoondreamService()

    @Published var loadingProgress: Double = 0
    @Published var isLoading: Bool = false
    @Published var loadError: String?

    private var modelContainer: ModelContainer?

    var isLoaded: Bool { modelContainer != nil }

    /// Currently loaded model ID
    private var loadedModelId: String?

    // MARK: - Idle Timer (Memory Management)

    /// Timer for unloading model after inactivity
    private var unloadTimer: Timer?

    /// Timeout duration before unloading model (30 seconds)
    private let unloadTimeout: TimeInterval = 30

    /// Resets the inactivity timer - call after each inference
    private func resetUnloadTimer() {
        unloadTimer?.invalidate()
        unloadTimer = Timer.scheduledTimer(withTimeInterval: unloadTimeout, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.performUnload()
            }
        }
        fileLog("Unload timer reset - model will unload in \(Int(unloadTimeout))s if idle")
    }

    /// Called when inactivity timeout expires
    private func performUnload() {
        guard modelContainer != nil else { return }
        fileLog("Inactivity timeout reached - unloading model to free memory")
        unloadModel()
        GPU.clearCache()
        fileLog("Model unloaded and GPU cache cleared")
    }

    // MARK: - Model Loading

    /// Load the Moondream3 model with specific model ID, downloading if necessary
    /// - Parameter modelId: HuggingFace model ID (e.g., "moondream/md3p-int4")
    func loadModel(modelId: String? = nil) async throws {
        let targetModelId = modelId ?? AvailableModels.defaultId

        // If already loaded with same model, skip
        if let loaded = loadedModelId, loaded == targetModelId, modelContainer != nil {
            fileLog("Model \(targetModelId) already loaded, skipping")
            return
        }

        isLoading = true
        loadError = nil
        fileLog("=== Starting model load: \(targetModelId) ===")

        #if os(iOS)
        // Configure MLX memory limits before loading model
        configureMLXMemory()
        #endif

        do {
            let configuration = Moondream3Loader.configuration(for: targetModelId)
            modelContainer = try await Moondream3Loader.loadContainer(
                configuration: configuration
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.loadingProgress = progress.fractionCompleted
                }
            }

            loadedModelId = targetModelId
            fileLog("=== Model loaded successfully: \(targetModelId) ===")

            #if os(iOS)
            // Clear GPU cache after model load to minimize baseline memory
            GPU.clearCache()
            fileLog("GPU cache cleared after model load")
            #endif

            loadingProgress = 1.0
            isLoading = false
        } catch {
            fileLog("ERROR: Failed to load model: \(error.localizedDescription)")
            loadError = error.localizedDescription
            isLoading = false
            throw error
        }
    }

    /// Unload the current model to free memory
    func unloadModel() {
        modelContainer = nil
        loadedModelId = nil
        loadingProgress = 0
        fileLog("Model unloaded")
    }

    // MARK: - Inference

    /// Query skill - ask questions about an image using direct model inference
    func query(image: CIImage, question: String) async throws -> QueryResult {
        // Auto-reload model if it was unloaded due to inactivity
        if modelContainer == nil {
            fileLog("Model not loaded - auto-reloading for query")
            try await loadModel()
        }
        guard let container = modelContainer else {
            throw MoondreamError.modelNotLoaded
        }

        fileLog("Starting query: \(question)")
        fileLog("Image size: \(image.extent.size)")

        let output = try await container.perform { context in
            fileLog("Container.perform started")

            // Process image - use .text() prompt type to ensure images array is set
            fileLog("Creating UserInput...")
            let userInput = UserInput(
                prompt: .text(question),
                images: [.ciImage(image)]
            )

            fileLog("Preparing input (image processing)...")
            let input = try await context.processor.prepare(input: userInput)
            fileLog("Input prepared successfully")

            guard let pixels = input.image?.pixels else {
                fileLog("ERROR: Failed to get pixels from processed image")
                throw MoondreamError.imageConversionFailed
            }
            fileLog("Pixels shape: \(pixels.shape)")

            // Support standard and quantized model types
            fileLog("Starting model.query()...")
            let result: String
            if let model = context.model as? Moondream3 {
                result = model.query(
                    pixels: pixels,
                    question: question,
                    tokenizer: context.tokenizer,
                    maxTokens: maxGenerationTokens,
                    temperature: 0.0
                )
            } else if let model = context.model as? Moondream3Quantized {
                result = model.query(
                    pixels: pixels,
                    question: question,
                    tokenizer: context.tokenizer,
                    maxTokens: maxGenerationTokens,
                    temperature: 0.0
                )
            } else {
                fileLog("ERROR: Unknown model type")
                throw MoondreamError.inferenceError("Unknown model type")
            }
            fileLog("model.query() completed successfully")
            return result
        }

        fileLog("Query complete: \(output)")

        #if os(iOS)
        // Clear GPU cache after inference to free memory
        GPU.clearCache()
        fileLog("GPU cache cleared")
        #endif

        // Reset idle timer after successful inference
        resetUnloadTimer()

        return QueryResult(answer: output.trimmingCharacters(in: .whitespacesAndNewlines), rawOutput: output)
    }

    /// Caption skill - generate image description using direct model inference
    func caption(image: CIImage, length: CaptionLength) async throws -> CaptionResult {
        // Auto-reload model if it was unloaded due to inactivity
        if modelContainer == nil {
            fileLog("Model not loaded - auto-reloading for caption")
            try await loadModel()
        }
        guard let container = modelContainer else {
            throw MoondreamError.modelNotLoaded
        }

        fileLog("Starting caption with length: \(length.rawValue)")
        fileLog("Image size: \(image.extent.size)")

        let output = try await container.perform { context in
            fileLog("Container.perform started")

            // Process image - use .text() prompt type to ensure images array is set
            fileLog("Creating UserInput...")
            let userInput = UserInput(
                prompt: .text("<caption:\(length.rawValue)>"),
                images: [.ciImage(image)]
            )

            fileLog("Preparing input (image processing)...")
            let input = try await context.processor.prepare(input: userInput)
            fileLog("Input prepared successfully")

            guard let pixels = input.image?.pixels else {
                fileLog("ERROR: Failed to get pixels from processed image")
                throw MoondreamError.imageConversionFailed
            }
            fileLog("Pixels shape: \(pixels.shape)")

            // Determine length string
            let lengthStr: String
            switch length {
            case .short: lengthStr = "short"
            case .normal: lengthStr = "normal"
            case .long: lengthStr = "long"
            }

            // Support standard and quantized model types
            fileLog("Starting model.caption() with length: \(lengthStr)")
            let result: String
            if let model = context.model as? Moondream3 {
                result = model.caption(
                    pixels: pixels,
                    length: lengthStr,
                    tokenizer: context.tokenizer,
                    maxTokens: maxGenerationTokens,
                    temperature: 0.0
                )
            } else if let model = context.model as? Moondream3Quantized {
                result = model.caption(
                    pixels: pixels,
                    length: lengthStr,
                    tokenizer: context.tokenizer,
                    maxTokens: maxGenerationTokens,
                    temperature: 0.0
                )
            } else {
                fileLog("ERROR: Unknown model type")
                throw MoondreamError.inferenceError("Unknown model type")
            }
            fileLog("model.caption() completed successfully")
            return result
        }

        fileLog("Caption complete: \(output)")

        #if os(iOS)
        GPU.clearCache()
        fileLog("GPU cache cleared")
        #endif

        // Reset idle timer after successful inference
        resetUnloadTimer()

        return CaptionResult(caption: output.trimmingCharacters(in: .whitespacesAndNewlines), rawOutput: output)
    }

    /// Point skill - locate objects by coordinates using direct model inference
    func point(image: CIImage, object: String) async throws -> PointResult {
        // Auto-reload model if it was unloaded due to inactivity
        if modelContainer == nil {
            fileLog("Model not loaded - auto-reloading for point")
            try await loadModel()
        }
        guard let container = modelContainer else {
            throw MoondreamError.modelNotLoaded
        }

        fileLog("Starting point: \(object)")
        fileLog("Image size: \(image.extent.size)")

        let output = try await container.perform { context in
            fileLog("Container.perform started")

            // Process image
            fileLog("Creating UserInput...")
            let userInput = UserInput(
                prompt: .text(object),
                images: [.ciImage(image)]
            )

            fileLog("Preparing input (image processing)...")
            let input = try await context.processor.prepare(input: userInput)
            fileLog("Input prepared successfully")

            guard let pixels = input.image?.pixels else {
                fileLog("ERROR: Failed to get pixels from processed image")
                throw MoondreamError.imageConversionFailed
            }
            fileLog("Pixels shape: \(pixels.shape)")

            // Support standard and quantized model types
            fileLog("Starting model.point()...")
            let result: String
            if let model = context.model as? Moondream3 {
                result = model.point(
                    pixels: pixels,
                    object: object,
                    tokenizer: context.tokenizer,
                    maxTokens: maxDetectionTokens,
                    temperature: 0.0
                )
            } else if let model = context.model as? Moondream3Quantized {
                result = model.point(
                    pixels: pixels,
                    object: object,
                    tokenizer: context.tokenizer,
                    maxTokens: maxDetectionTokens,
                    temperature: 0.0
                )
            } else {
                fileLog("ERROR: Unknown model type")
                throw MoondreamError.inferenceError("Unknown model type")
            }
            fileLog("model.point() completed successfully")
            return result
        }

        fileLog("Point complete: \(output)")

        #if os(iOS)
        GPU.clearCache()
        fileLog("GPU cache cleared")
        #endif

        // Reset idle timer after successful inference
        resetUnloadTimer()

        let points = parsePointResponse(output)
        return PointResult(points: points, rawOutput: output)
    }

    /// Detect skill - detect objects with bounding boxes using direct model inference
    func detect(image: CIImage, object: String) async throws -> DetectResult {
        // Auto-reload model if it was unloaded due to inactivity
        if modelContainer == nil {
            fileLog("Model not loaded - auto-reloading for detect")
            try await loadModel()
        }
        guard let container = modelContainer else {
            throw MoondreamError.modelNotLoaded
        }

        fileLog("Starting detect: \(object)")
        fileLog("Image size: \(image.extent.size)")

        let output = try await container.perform { context in
            fileLog("Container.perform started")

            // Process image
            fileLog("Creating UserInput...")
            let userInput = UserInput(
                prompt: .text(object),
                images: [.ciImage(image)]
            )

            fileLog("Preparing input (image processing)...")
            let input = try await context.processor.prepare(input: userInput)
            fileLog("Input prepared successfully")

            guard let pixels = input.image?.pixels else {
                fileLog("ERROR: Failed to get pixels from processed image")
                throw MoondreamError.imageConversionFailed
            }
            fileLog("Pixels shape: \(pixels.shape)")

            // Support standard and quantized model types
            fileLog("Starting model.detect()...")
            let result: String
            if let model = context.model as? Moondream3 {
                result = model.detect(
                    pixels: pixels,
                    object: object,
                    tokenizer: context.tokenizer,
                    maxTokens: maxDetectionTokens,
                    temperature: 0.0
                )
            } else if let model = context.model as? Moondream3Quantized {
                result = model.detect(
                    pixels: pixels,
                    object: object,
                    tokenizer: context.tokenizer,
                    maxTokens: maxDetectionTokens,
                    temperature: 0.0
                )
            } else {
                fileLog("ERROR: Unknown model type")
                throw MoondreamError.inferenceError("Unknown model type")
            }
            fileLog("model.detect() completed successfully")
            return result
        }

        fileLog("Detect complete: \(output)")

        #if os(iOS)
        GPU.clearCache()
        fileLog("GPU cache cleared")
        #endif

        // Reset idle timer after successful inference
        resetUnloadTimer()

        let boxes = parseDetectResponse(output)
        return DetectResult(boxes: boxes, rawOutput: output)
    }

    // MARK: - Private Methods

    private func runInference(image: CIImage, prompt: String) async throws -> String {
        guard let container = modelContainer else {
            throw MoondreamError.modelNotLoaded
        }

        logger.info("Starting inference with prompt: \(prompt)")

        let userInput = UserInput(
            prompt: prompt,
            images: [.ciImage(image)]
        )

        let result = try await container.perform { context in
            let input = try await context.processor.prepare(input: userInput)

            var outputTokens: [Int] = []

            _ = try MLXLMCommon.generate(
                input: input,
                parameters: GenerateParameters(temperature: 0.0),
                context: context
            ) { tokens in
                outputTokens = tokens
                return tokens.count >= 768 ? .stop : .more
            }

            return context.tokenizer.decode(tokens: outputTokens)
        }

        logger.info("Inference complete: \(result)")
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Prompt Building

    private func buildQueryPrompt(question: String) -> String {
        question
    }

    private func buildCaptionPrompt(length: CaptionLength) -> String {
        switch length {
        case .short: "<caption:short>"
        case .normal: "<caption:normal>"
        case .long: "<caption:long>"
        }
    }

    private func buildPointPrompt(object: String) -> String {
        "<point>\(object)"
    }

    private func buildDetectPrompt(object: String) -> String {
        "<detect>\(object)"
    }

    // MARK: - Response Parsing

    private func parsePointResponse(_ output: String) -> [NormalizedPoint] {
        if let data = output.data(using: .utf8),
           let response = try? JSONDecoder().decode(PointResponse.self, from: data) {
            return response.toNormalizedPoints()
        }

        let pattern = #"\((\d+\.?\d*),\s*(\d+\.?\d*)\)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }

        let range = NSRange(output.startIndex..., in: output)
        let matches = regex.matches(in: output, range: range)

        return matches.compactMap { match -> NormalizedPoint? in
            guard match.numberOfRanges == 3,
                  let xRange = Range(match.range(at: 1), in: output),
                  let yRange = Range(match.range(at: 2), in: output),
                  let x = Double(output[xRange]),
                  let y = Double(output[yRange]) else {
                return nil
            }
            return NormalizedPoint(x: CGFloat(x), y: CGFloat(y))
        }
    }

    private func parseDetectResponse(_ output: String) -> [NormalizedBox] {
        if let data = output.data(using: .utf8),
           let response = try? JSONDecoder().decode(DetectResponse.self, from: data) {
            return response.toNormalizedBoxes()
        }

        let pattern = #"\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }

        let range = NSRange(output.startIndex..., in: output)
        let matches = regex.matches(in: output, range: range)

        return matches.compactMap { match -> NormalizedBox? in
            guard match.numberOfRanges == 5,
                  let xMinRange = Range(match.range(at: 1), in: output),
                  let yMinRange = Range(match.range(at: 2), in: output),
                  let xMaxRange = Range(match.range(at: 3), in: output),
                  let yMaxRange = Range(match.range(at: 4), in: output),
                  let xMin = Double(output[xMinRange]),
                  let yMin = Double(output[yMinRange]),
                  let xMax = Double(output[xMaxRange]),
                  let yMax = Double(output[yMaxRange]) else {
                return nil
            }
            return NormalizedBox(
                xMin: CGFloat(xMin),
                yMin: CGFloat(yMin),
                xMax: CGFloat(xMax),
                yMax: CGFloat(yMax)
            )
        }
    }
}

/// Errors from MoondreamService
enum MoondreamError: LocalizedError {
    case modelNotLoaded
    case imageConversionFailed
    case inferenceError(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            "Model not loaded. Please wait for download to complete."
        case .imageConversionFailed:
            "Failed to convert image for processing."
        case .inferenceError(let message):
            "Inference error: \(message)"
        }
    }
}
