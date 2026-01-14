import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

print("=== Moondream3 Model Load Test ===\n")

// Test 1: Load tokenizer from starmie-v1
print("[1] Testing tokenizer load from moondream/starmie-v1...")
do {
    let hub = HubApi()
    let tokenizerConfig = ModelConfiguration(id: "moondream/starmie-v1")
    let tokenizer = try await MLXLMCommon.loadTokenizer(configuration: tokenizerConfig, hub: hub)
    print("    SUCCESS: Tokenizer loaded")

    // Test tokenization
    let testText = "Hello, world!"
    let tokens = tokenizer.encode(text: testText)
    print("    Test encode '\(testText)' -> \(tokens.count) tokens: \(tokens)")

    // Check vocab size by encoding a range of tokens
    let decoded = tokenizer.decode(tokens: tokens)
    print("    Test decode -> '\(decoded)'")

    // Try to determine vocab size by testing edge tokens
    let testTokens = [0, 1, 100, 1000, 10000, 50000, 51199]
    for tokenId in testTokens {
        let decoded = tokenizer.decode(tokens: [tokenId])
        print("    Token \(tokenId) -> '\(decoded)'")
    }

} catch {
    print("    FAILED: \(error)")
}

// Test 2: Download model config from md3p-int4
print("\n[2] Testing model config download from moondream/md3p-int4...")
do {
    let hub = HubApi()
    let repo = Hub.Repo(id: "moondream/md3p-int4")

    // Download config.json
    let modelDirectory = try await hub.snapshot(from: repo, matching: ["config.json"]) { progress in
        print("    Download progress: \(Int(progress.fractionCompleted * 100))%")
    }

    let configURL = modelDirectory.appendingPathComponent("config.json")
    let configData = try Data(contentsOf: configURL)

    if let json = try JSONSerialization.jsonObject(with: configData) as? [String: Any] {
        print("    SUCCESS: Config loaded")
        if let vocabSize = json["vocab_size"] as? Int {
            print("    vocab_size: \(vocabSize)")
        }
        if let textConfig = json["text_config"] as? [String: Any] {
            if let vocabSize = textConfig["vocab_size"] as? Int {
                print("    text_config.vocab_size: \(vocabSize)")
            }
            if let hiddenSize = textConfig["hidden_size"] as? Int {
                print("    text_config.hidden_size: \(hiddenSize)")
            }
        }
        // Print full JSON for debugging
        let prettyData = try JSONSerialization.data(withJSONObject: json, options: .prettyPrinted)
        if let prettyStr = String(data: prettyData, encoding: .utf8) {
            let lines = prettyStr.components(separatedBy: "\n").prefix(30)
            print("    Config preview:\n\(lines.joined(separator: "\n"))")
        }
    }
} catch {
    print("    FAILED: \(error)")
}

// Test 3: Download and load weights (just check file existence)
print("\n[3] Testing weight file download from moondream/md3p-int4...")
do {
    let hub = HubApi()
    let repo = Hub.Repo(id: "moondream/md3p-int4")
    let modelDirectory = hub.localRepoLocation(repo)

    // Check if weights already cached
    let configPath = modelDirectory.appendingPathComponent("config.json")
    if FileManager.default.fileExists(atPath: configPath.path) {
        let contents = try FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)
        let safetensors = contents.filter { $0.pathExtension == "safetensors" }
        if !safetensors.isEmpty {
            print("    Model already cached at: \(modelDirectory.path)")
            print("    Weight files: \(safetensors.map { $0.lastPathComponent })")

            // Try loading one weight file to verify
            print("\n[4] Testing weight loading...")
            let weights = try MLX.loadArrays(url: safetensors[0])
            print("    Loaded \(weights.count) arrays from \(safetensors[0].lastPathComponent)")

            // Show some key prefixes
            let prefixes = Set(weights.keys.map { $0.components(separatedBy: ".").prefix(2).joined(separator: ".") })
            print("    Key prefixes: \(prefixes.sorted().prefix(10))")
        } else {
            print("    No weight files cached, would need to download ~2GB")
        }
    } else {
        print("    Model not cached, would need to download")
    }
} catch {
    print("    FAILED: \(error)")
}

print("\n=== Test Complete ===")
