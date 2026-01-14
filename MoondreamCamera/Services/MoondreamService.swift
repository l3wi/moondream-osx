import Foundation
import CoreImage
import MLX
import MLXNN
import MLXVLM
import MLXLMCommon
import os.log

private let logger = Logger(subsystem: "com.moondream.camera", category: "MoondreamService")

// File-based logging for debugging (since os.log doesn't show in idevicesyslog)
private func serviceLog(_ message: String) {
    logger.error("\(message)")
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "HH:mm:ss.SSS"
    let timestamp = dateFormatter.string(from: Date())
    let logLine = "[\(timestamp)] \(message)\n"

    if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
        let logFile = docs.appendingPathComponent("moondream_debug.log")
        if let data = logLine.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: logFile.path) {
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
}

/// Service for loading and running inference with the Moondream model
@MainActor
final class MoondreamService: ObservableObject {
    static let shared = MoondreamService()

    private var modelContainer: ModelContainer?

    var isLoaded: Bool { modelContainer != nil }

    // MARK: - Model Loading

    /// Load the Moondream3 model, downloading if necessary
    func loadModel(progressHandler: @escaping @Sendable (Double) -> Void) async throws {
        logger.error("[MoondreamService] Starting model load...")

        do {
            // Use Moondream3 - our custom VLM implementation
            modelContainer = try await Moondream3Loader.loadContainer(
                configuration: Moondream3Loader.defaultConfiguration
            ) { progress in
                Task { @MainActor in
                    progressHandler(progress.fractionCompleted)
                }
            }

            logger.error("[MoondreamService] Model loaded successfully!")
            progressHandler(1.0)
        } catch {
            logger.error("[MoondreamService] ERROR: Failed to load model")
            logger.error("[MoondreamService] Error: \(error)")
            logger.error("[MoondreamService] Error type: \(type(of: error))")
            logger.error("[MoondreamService] Localized: \(error.localizedDescription)")
            throw error
        }
    }

    // MARK: - Inference

    /// Query skill - ask questions about an image
    func query(image: CIImage, question: String) async throws -> QueryResult {
        let prompt = buildQueryPrompt(question: question)
        let output = try await runInference(image: image, prompt: prompt)

        return QueryResult(answer: output, rawOutput: output)
    }

    /// Caption skill - generate image description
    func caption(image: CIImage, length: CaptionLength) async throws -> CaptionResult {
        let prompt = buildCaptionPrompt(length: length)
        let output = try await runInference(image: image, prompt: prompt)

        return CaptionResult(caption: output, rawOutput: output)
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

        serviceLog("[MoondreamService] Starting inference with prompt: \(prompt)")

        let userInput = UserInput(
            prompt: prompt,
            images: [.ciImage(image)]
        )

        let result = try await container.perform { context in
            serviceLog("[MoondreamService] Preparing input...")
            let input = try await context.processor.prepare(input: userInput)

            // Log input tokens
            let inputTokens = input.text.tokens.asArray(Int.self)
            serviceLog("[MoondreamService] Input tokens (\(inputTokens.count)): \(inputTokens)")

            var outputTokens: [Int] = []

            serviceLog("[MoondreamService] Starting generation...")
            // Moondream3 defaults: temperature=0.5, top_p not used with MLXLMCommon
            // Using temperature=0 for greedy/deterministic output
            _ = try MLXLMCommon.generate(
                input: input,
                parameters: GenerateParameters(temperature: 0.0),
                context: context
            ) { tokens in
                outputTokens = tokens
                // Max tokens 768 as per moondream defaults
                return tokens.count >= 768 ? .stop : .more
            }

            serviceLog("[MoondreamService] Generation complete. Total tokens: \(outputTokens.count)")
            serviceLog("[MoondreamService] Output tokens: \(outputTokens)")

            let decoded = context.tokenizer.decode(tokens: outputTokens)
            serviceLog("[MoondreamService] Decoded output: \(decoded)")

            return decoded
        }

        serviceLog("[MoondreamService] Final result: \(result)")
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Prompt Building
    // Moondream3 uses specific token templates, not natural language
    // These prompts will be processed by Moondream3Processor which handles the token formatting

    private func buildQueryPrompt(question: String) -> String {
        // The processor will wrap this with query tokens [1, 15381, 2] + question + [3]
        question
    }

    private func buildCaptionPrompt(length: CaptionLength) -> String {
        // Return marker that processor will convert to proper token template
        switch length {
        case .short:
            "<caption:short>"
        case .normal:
            "<caption:normal>"
        case .long:
            "<caption:long>"
        }
    }

    private func buildPointPrompt(object: String) -> String {
        // The processor will wrap with point tokens [1, 2581, 2] + object + [3]
        "<point>\(object)"
    }

    private func buildDetectPrompt(object: String) -> String {
        // The processor will wrap with detect tokens [1, 7235, 476, 2] + object + [3]
        "<detect>\(object)"
    }

    // MARK: - Response Parsing

    private func parsePointResponse(_ output: String) -> [NormalizedPoint] {
        // Try to parse JSON response
        if let data = output.data(using: .utf8),
           let response = try? JSONDecoder().decode(PointResponse.self, from: data) {
            return response.toNormalizedPoints()
        }

        // Try to extract coordinates from text like "(0.5, 0.3)"
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
        // Try to parse JSON response
        if let data = output.data(using: .utf8),
           let response = try? JSONDecoder().decode(DetectResponse.self, from: data) {
            return response.toNormalizedBoxes()
        }

        // Try to extract bounding boxes from text like "[0.1, 0.2, 0.5, 0.8]"
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
