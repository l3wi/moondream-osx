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

    /// Original model configuration (partially quantized MoE, BF16 vision/attention)
    /// Use this for macOS where memory is not as constrained
    public static let originalConfiguration = ModelConfiguration(
        id: "moondream/md3p-int4",
        defaultPrompt: "Describe this image."
    )

    /// Fully quantized model configuration (int4 everywhere)
    /// Use this for iOS where memory is limited to ~6GB
    public static let quantizedConfiguration = ModelConfiguration(
        id: "lewi/md3p-int4-smol",
        defaultPrompt: "Describe this image."
    )

    /// Default configuration - uses quantized model on iOS, original on macOS
    #if os(iOS)
    public static let defaultConfiguration = quantizedConfiguration
    #else
    public static let defaultConfiguration = originalConfiguration
    #endif

    /// Get a ModelConfiguration for a specific model ID or local directory path
    /// - Parameter modelId: The HuggingFace model ID (e.g., "moondream/md3p-int4") or local directory path
    /// - Returns: A ModelConfiguration for the specified model
    public static func configuration(for modelId: String) -> ModelConfiguration {
        // Check if it's a local directory path
        if modelId.hasPrefix("/") || modelId.hasPrefix("./") || modelId.hasPrefix("~") {
            let expandedPath = (modelId as NSString).expandingTildeInPath
            let url = URL(fileURLWithPath: expandedPath)
            return ModelConfiguration(
                directory: url,
                defaultPrompt: "Describe this image."
            )
        }
        // Otherwise treat as HuggingFace model ID
        return ModelConfiguration(
            id: modelId,
            defaultPrompt: "Describe this image."
        )
    }

    /// Model type for loading
    private enum ModelType {
        case standard    // moondream/md3p-int4 (MoE int4, Vision/Attention BF16) - also handles int8 via config.bits
        case quantized   // lewi/md3p-int4-smol (fully int4)
    }

    /// Load the Moondream3 model and create a ModelContext
    /// Automatically uses the appropriate model class based on the configuration
    public static func load(
        configuration: ModelConfiguration = defaultConfiguration,
        hub: HubApi = HubApi(),
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContext {
        fileLog("=== Moondream3Loader.load() started ===")

        // Determine model type from configuration
        let modelType: ModelType
        switch configuration.id {
        case .id(let id, _):
            if id.contains("md3p-int4-smol") || id.contains("lewi/") {
                modelType = .quantized
                fileLog("[Moondream3Loader] Model ID: \(id), type: quantized")
            } else {
                modelType = .standard
                fileLog("[Moondream3Loader] Model ID: \(id), type: standard")
            }
        case .directory(let url):
            // For local directories, check config.json to determine model type
            modelType = detectModelType(from: url)
            fileLog("[Moondream3Loader] Local directory: \(url.path), detected type: \(modelType)")
        }

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

        // Create appropriate model based on type
        let model: any LanguageModel
        switch modelType {
        case .quantized:
            fileLog("[Moondream3Loader] Creating quantized model (Moondream3Quantized)")
            let quantizedModel = Moondream3Quantized(modelConfig)
            try loadWeightsQuantized(into: quantizedModel, from: modelDirectory)
            model = quantizedModel
        case .standard:
            fileLog("[Moondream3Loader] Creating standard model (Moondream3)")
            let standardModel = Moondream3(modelConfig)
            try loadWeights(into: standardModel, from: modelDirectory)
            model = standardModel
        }

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

    /// Detect model type from a local directory by reading config.json
    /// Standard model handles both int4 and int8 via config.text.bits
    private static func detectModelType(from directory: URL) -> ModelType {
        let configURL = directory.appendingPathComponent("config.json")

        guard FileManager.default.fileExists(atPath: configURL.path),
              let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            fileLog("[Moondream3Loader] Could not read config.json, defaulting to standard")
            return .standard
        }

        // Check if text config has group_size=32 (indicates quantization)
        // Also check region for full quantization (lewi/md3p-int4-smol)
        if let textConfig = json["text"] as? [String: Any],
           let groupSize = textConfig["group_size"] as? Int,
           groupSize == 32,
           let regionConfig = json["region"] as? [String: Any],
           let regionGroupSize = regionConfig["group_size"] as? Int,
           regionGroupSize == 32 {
            fileLog("[Moondream3Loader] Detected fully quantized model")
            return .quantized
        }

        // Default to standard (handles int4 and int8 via config.bits)
        fileLog("[Moondream3Loader] Defaulting to standard model type")
        return .standard
    }

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

    private static func loadWeightsQuantized(into model: Moondream3Quantized, from directory: URL) throws {
        // Find all safetensors files
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let weightFiles = contents.filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }

        guard !weightFiles.isEmpty else {
            throw Moondream3LoaderError.noWeightFiles
        }

        fileLog("[Moondream3Loader] Found \(weightFiles.count) weight files (quantized model)")

        // Process weights incrementally to reduce peak memory
        var totalKeys = 0

        for (index, weightFile) in weightFiles.enumerated() {
            fileLog("[Moondream3Loader] Loading [\(index + 1)/\(weightFiles.count)]: \(weightFile.lastPathComponent)")

            // Load this shard's weights
            let weights = try MLX.loadArrays(url: weightFile)
            totalKeys += weights.count

            // Sanitize this shard
            let sanitizedWeights = model.sanitize(weights: weights)

            // Apply this shard's weights
            let parameters = ModuleParameters.unflattened(sanitizedWeights)
            do {
                try model.update(parameters: parameters, verify: .none)
            } catch {
                fileLog("[Moondream3Loader] ERROR applying weights from \(weightFile.lastPathComponent): \(error)")
                throw error
            }

            fileLog("[Moondream3Loader] Applied \(sanitizedWeights.count) weights from \(weightFile.lastPathComponent)")
        }

        fileLog("[Moondream3Loader] Loaded \(totalKeys) total raw weight keys (quantized)")
        fileLog("[Moondream3Loader] Quantized model ready (weights will evaluate lazily)")
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
