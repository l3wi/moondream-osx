// Copyright © 2024 Moondream AI
// Loader for Moondream3 model

import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import os.log

private let logger = Logger(subsystem: "com.moondream.camera", category: "Moondream3Loader")

// File-based logging for debugging (since os.log doesn't show in idevicesyslog)
private func fileLog(_ message: String) {
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

/// Loader for Moondream3 model
public enum Moondream3Loader {

    /// Model configuration for Moondream3
    public static let defaultConfiguration = ModelConfiguration(
        id: "moondream/md3p-int4",
        defaultPrompt: "Describe this image."
    )

    /// Load the Moondream3 model and create a ModelContext
    public static func load(
        configuration: ModelConfiguration = defaultConfiguration,
        hub: HubApi = HubApi(),
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContext {
        fileLog("=== Moondream3Loader.load() started ===")

        // Download model files from HuggingFace
        let modelDirectory = try await downloadModel(
            configuration: configuration,
            hub: hub,
            progressHandler: progressHandler
        )

        // Load configuration
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let modelConfig = try JSONDecoder().decode(Moondream3Configuration.self, from: configData)

        // Create model
        let model = Moondream3(modelConfig)

        // Load weights
        try loadWeights(into: model, from: modelDirectory)

        // Load tokenizer from the already-downloaded model directory
        fileLog("[Moondream3Loader] Loading tokenizer from \(modelDirectory.path)")
        let tokenizer = try await loadTokenizer(from: modelDirectory, configuration: configuration, hub: hub)

        // Create processor
        fileLog("[Moondream3Loader] Creating processor...")
        let processorConfig = Moondream3ProcessorConfiguration()
        let processor = Moondream3Processor(config: processorConfig, tokenizer: tokenizer)

        fileLog("[Moondream3Loader] === Model load complete! ===")
        return ModelContext(
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )
    }

    /// Load the model into a ModelContainer for thread-safe access
    public static func loadContainer(
        configuration: ModelConfiguration = defaultConfiguration,
        hub: HubApi = HubApi(),
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        let context = try await load(
            configuration: configuration,
            hub: hub,
            progressHandler: progressHandler
        )
        return ModelContainer(context: context)
    }

    // MARK: - Private Methods

    private static func downloadModel(
        configuration: ModelConfiguration,
        hub: HubApi,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        // Handle local directory case
        switch configuration.id {
        case .directory(let url):
            return url
        case .id(let id, _):
            let repo = Hub.Repo(id: id)
            let modelDirectory = hub.localRepoLocation(repo)

            // Check if model already exists in cache
            let configPath = modelDirectory.appendingPathComponent("config.json")
            let fileManager = FileManager.default

            if fileManager.fileExists(atPath: configPath.path) {
                // Check for safetensors files
                if let contents = try? fileManager.contentsOfDirectory(
                    at: modelDirectory,
                    includingPropertiesForKeys: nil
                ),
                    contents.contains(where: { $0.pathExtension == "safetensors" })
                {
                    fileLog("Model already cached at \(modelDirectory.path)")
                    return modelDirectory
                }
            }

            // Download required files
            let files = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ]

            for file in files {
                _ = try await hub.snapshot(
                    from: repo,
                    matching: [file],
                    progressHandler: progressHandler
                )
            }

            // Download weight files (may be sharded)
            _ = try await hub.snapshot(
                from: repo,
                matching: ["*.safetensors"],
                progressHandler: progressHandler
            )

            return modelDirectory
        }
    }

    private static func loadWeights(into model: Moondream3, from directory: URL) throws {
        // Find all safetensors files
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let weightFiles = contents.filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }

        guard !weightFiles.isEmpty else {
            throw Moondream3LoaderError.noWeightFiles
        }

        fileLog("[Moondream3Loader] Found \(weightFiles.count) weight files")

        // Process weights incrementally to reduce peak memory
        // Load each file, sanitize, and apply before loading the next
        var totalKeys = 0
        var allRawPrefixes = Set<String>()
        var allSanitizedPrefixes = Set<String>()

        for (index, weightFile) in weightFiles.enumerated() {
            fileLog("[Moondream3Loader] Loading [\(index + 1)/\(weightFiles.count)]: \(weightFile.lastPathComponent)")

            // Load this shard's weights
            let weights = try MLX.loadArrays(url: weightFile)
            totalKeys += weights.count

            // Track prefixes for debugging
            for key in weights.keys {
                let prefix = key.components(separatedBy: ".").prefix(2).joined(separator: ".")
                allRawPrefixes.insert(prefix)
            }

            // Sanitize this shard
            let sanitizedWeights = model.sanitize(weights: weights)

            for key in sanitizedWeights.keys {
                let prefix = key.components(separatedBy: ".").prefix(3).joined(separator: ".")
                allSanitizedPrefixes.insert(prefix)
            }

            // Apply this shard's weights (skip verification since we're loading incrementally)
            let parameters = ModuleParameters.unflattened(sanitizedWeights)
            do {
                try model.update(parameters: parameters, verify: .none)
            } catch {
                fileLog("[Moondream3Loader] ERROR applying weights from \(weightFile.lastPathComponent): \(error)")
                throw error
            }

            // weights and sanitizedWeights go out of scope here, freeing memory
            fileLog("[Moondream3Loader] Applied \(sanitizedWeights.count) weights from \(weightFile.lastPathComponent)")
        }

        fileLog("[Moondream3Loader] Loaded \(totalKeys) total raw weight keys")
        fileLog("[Moondream3Loader] Raw key prefixes: \(allRawPrefixes.sorted())")
        fileLog("[Moondream3Loader] Sanitized key prefixes: \(allSanitizedPrefixes.sorted())")

        // NOTE: Removed eval(model) - MLX will lazily evaluate weights on first inference
        // Calling eval() here caused OOM by forcing all ~2GB weights to GPU at once
        fileLog("[Moondream3Loader] Model ready (weights will evaluate lazily)")
    }

    private static func loadTokenizer(
        from directory: URL,
        configuration: ModelConfiguration,
        hub: HubApi
    ) async throws -> any Tokenizer {
        // Try to load tokenizer from local directory first
        let tokenizerPath = directory.appendingPathComponent("tokenizer.json")
        fileLog("[Moondream3Loader] Checking for tokenizer at: \(tokenizerPath.path)")

        // List directory contents to debug
        if let contents = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
            let files = contents.map { $0.lastPathComponent }
            fileLog("[Moondream3Loader] Directory contains: \(files.joined(separator: ", "))")
        }

        if FileManager.default.fileExists(atPath: tokenizerPath.path) {
            fileLog("[Moondream3Loader] Found local tokenizer.json, loading...")
            do {
                let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
                fileLog("[Moondream3Loader] Tokenizer loaded successfully")
                return tokenizer
            } catch {
                fileLog("[Moondream3Loader] Failed to load local tokenizer: \(error), falling back to Hub")
            }
        } else {
            fileLog("[Moondream3Loader] tokenizer.json NOT found at path")
        }

        // Load tokenizer from starmie-v1 (public repo with moondream3-compatible tokenizer files)
        fileLog("[Moondream3Loader] Loading tokenizer from moondream/starmie-v1...")
        do {
            let tokenizerConfig = ModelConfiguration(id: "moondream/starmie-v1")
            let tokenizer = try await MLXLMCommon.loadTokenizer(configuration: tokenizerConfig, hub: hub)
            fileLog("[Moondream3Loader] Tokenizer loaded successfully from starmie-v1!")
            return tokenizer
        } catch {
            fileLog("[Moondream3Loader] ERROR loading tokenizer: \(error)")
            throw error
        }
    }
}

/// Errors for Moondream3 loading
public enum Moondream3LoaderError: LocalizedError {
    case noWeightFiles
    case configurationMissing
    case tokenizerLoadFailed

    public var errorDescription: String? {
        switch self {
        case .noWeightFiles:
            return "No weight files (.safetensors) found in model directory"
        case .configurationMissing:
            return "Model configuration file (config.json) not found"
        case .tokenizerLoadFailed:
            return "Failed to load tokenizer"
        }
    }
}
