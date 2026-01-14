import Foundation

/// Downloads the Moondream model from HuggingFace
actor ModelDownloader {
    private let hubBaseURL = "https://huggingface.co/moondream/md3p-int4/resolve/main/"

    /// Files required for the model
    private let requiredFiles = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    /// Get the local model directory path
    static var localModelDirectory: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appendingPathComponent("moondream-md3p-int4", isDirectory: true)
    }

    /// Check if the model is already downloaded
    func isModelDownloaded() -> Bool {
        let modelDir = Self.localModelDirectory

        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            return false
        }

        // Check that all required files exist
        for file in requiredFiles {
            let filePath = modelDir.appendingPathComponent(file)
            if !FileManager.default.fileExists(atPath: filePath.path) {
                return false
            }
        }

        return true
    }

    /// Download the model from HuggingFace
    /// - Parameter progressHandler: Called with progress (0.0 - 1.0)
    func downloadModel(progressHandler: @escaping @Sendable (Double) -> Void) async throws {
        let modelDir = Self.localModelDirectory

        // Create directory if needed
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let totalFiles = Double(requiredFiles.count)

        for (index, file) in requiredFiles.enumerated() {
            let fileURL = URL(string: hubBaseURL + file)!
            let destPath = modelDir.appendingPathComponent(file)

            // Skip if file already exists
            if FileManager.default.fileExists(atPath: destPath.path) {
                let progress = Double(index + 1) / totalFiles
                progressHandler(progress)
                continue
            }

            try await downloadFile(from: fileURL, to: destPath) { fileProgress in
                let overallProgress = (Double(index) + fileProgress) / totalFiles
                progressHandler(overallProgress)
            }
        }

        progressHandler(1.0)
    }

    /// Download a single file with progress tracking
    private func downloadFile(
        from url: URL,
        to destination: URL,
        progressHandler: @escaping @Sendable (Double) -> Void
    ) async throws {
        let (asyncBytes, response) = try await URLSession.shared.bytes(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw ModelDownloadError.downloadFailed(url: url)
        }

        let totalBytes = response.expectedContentLength
        var downloadedBytes: Int64 = 0
        var data = Data()

        // Reserve capacity if known
        if totalBytes > 0 {
            data.reserveCapacity(Int(totalBytes))
        }

        for try await byte in asyncBytes {
            data.append(byte)
            downloadedBytes += 1

            // Report progress every 1MB to avoid excessive updates
            if downloadedBytes % (1024 * 1024) == 0 && totalBytes > 0 {
                progressHandler(Double(downloadedBytes) / Double(totalBytes))
            }
        }

        try data.write(to: destination)
        progressHandler(1.0)
    }

    /// Delete the local model cache
    func deleteLocalModel() throws {
        let modelDir = Self.localModelDirectory
        if FileManager.default.fileExists(atPath: modelDir.path) {
            try FileManager.default.removeItem(at: modelDir)
        }
    }
}

/// Errors from model downloading
enum ModelDownloadError: LocalizedError {
    case downloadFailed(url: URL)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .downloadFailed(let url):
            "Failed to download: \(url.lastPathComponent)"
        case .invalidResponse:
            "Invalid response from server"
        }
    }
}
