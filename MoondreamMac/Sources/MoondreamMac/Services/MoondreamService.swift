import Foundation
import CoreImage
import MLX
import MLXNN
import MLXVLM
import MLXLMCommon
import os.log

private let logger = Logger(subsystem: "com.moondream.mac", category: "MoondreamService")

/// Service for loading and running inference with the Moondream model
@MainActor
final class MoondreamService: ObservableObject {
    static let shared = MoondreamService()

    @Published var loadingProgress: Double = 0
    @Published var isLoading: Bool = false
    @Published var loadError: String?

    private var modelContainer: ModelContainer?

    var isLoaded: Bool { modelContainer != nil }

    // MARK: - Model Loading

    /// Load the Moondream3 model, downloading if necessary
    func loadModel() async throws {
        isLoading = true
        loadError = nil
        logger.info("Starting model load...")

        do {
            modelContainer = try await Moondream3Loader.loadContainer(
                configuration: Moondream3Loader.defaultConfiguration
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.loadingProgress = progress.fractionCompleted
                }
            }

            logger.info("Model loaded successfully!")
            loadingProgress = 1.0
            isLoading = false
        } catch {
            logger.error("Failed to load model: \(error.localizedDescription)")
            loadError = error.localizedDescription
            isLoading = false
            throw error
        }
    }

    // MARK: - Inference

    /// Query skill - ask questions about an image using direct model inference
    func query(image: CIImage, question: String) async throws -> QueryResult {
        guard let container = modelContainer else {
            throw MoondreamError.modelNotLoaded
        }

        logger.info("Starting query: \(question)")

        let output = try await container.perform { context in
            // Process image - use .text() prompt type to ensure images array is set
            let userInput = UserInput(
                prompt: .text(question),
                images: [.ciImage(image)]
            )
            let input = try await context.processor.prepare(input: userInput)

            guard let pixels = input.image?.pixels else {
                throw MoondreamError.imageConversionFailed
            }

            guard let model = context.model as? Moondream3 else {
                throw MoondreamError.inferenceError("Model is not Moondream3")
            }

            return model.query(
                pixels: pixels,
                question: question,
                tokenizer: context.tokenizer,
                maxTokens: 768,
                temperature: 0.0
            )
        }

        logger.info("Query complete: \(output)")
        return QueryResult(answer: output.trimmingCharacters(in: .whitespacesAndNewlines), rawOutput: output)
    }

    /// Caption skill - generate image description using direct model inference
    func caption(image: CIImage, length: CaptionLength) async throws -> CaptionResult {
        guard let container = modelContainer else {
            throw MoondreamError.modelNotLoaded
        }

        logger.info("Starting caption with length: \(length.rawValue)")

        let output = try await container.perform { context in
            // Process image - use .text() prompt type to ensure images array is set
            let userInput = UserInput(
                prompt: .text("<caption:\(length.rawValue)>"),
                images: [.ciImage(image)]
            )
            let input = try await context.processor.prepare(input: userInput)

            guard let pixels = input.image?.pixels else {
                throw MoondreamError.imageConversionFailed
            }

            // Get model and tokenizer
            guard let model = context.model as? Moondream3 else {
                throw MoondreamError.inferenceError("Model is not Moondream3")
            }

            // Use new caption method that properly handles KV cache
            let lengthStr: String
            switch length {
            case .short: lengthStr = "short"
            case .normal: lengthStr = "normal"
            case .long: lengthStr = "long"
            }

            return model.caption(
                pixels: pixels,
                length: lengthStr,
                tokenizer: context.tokenizer,
                maxTokens: 768,
                temperature: 0.0
            )
        }

        logger.info("Caption complete: \(output)")
        return CaptionResult(caption: output.trimmingCharacters(in: .whitespacesAndNewlines), rawOutput: output)
    }

    /// Point skill - locate objects by coordinates
    func point(image: CIImage, object: String) async throws -> PointResult {
        let prompt = buildPointPrompt(object: object)
        let output = try await runInference(image: image, prompt: prompt)
        let points = parsePointResponse(output)
        return PointResult(points: points, rawOutput: output)
    }

    /// Detect skill - detect objects with bounding boxes
    func detect(image: CIImage, object: String) async throws -> DetectResult {
        let prompt = buildDetectPrompt(object: object)
        let output = try await runInference(image: image, prompt: prompt)
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
