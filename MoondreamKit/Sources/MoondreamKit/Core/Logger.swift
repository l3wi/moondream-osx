import Foundation
import os.log

/// Shared logger instance for MoondreamKit
internal let md3Logger = Logger(subsystem: "com.moondream.kit", category: "Moondream3")

/// Log a message to the system logger and optionally to a file on iOS
/// - Parameter message: The message to log
internal func md3Log(_ message: String) {
    md3Logger.info("\(message)")

    #if os(iOS)
    if PlatformConfiguration.fileLoggingEnabled {
        writeToLogFile(message)
    }
    #endif
}

// MARK: - iOS File Logging

#if os(iOS)
/// Date formatter for log timestamps (reused to avoid allocation overhead)
private let logDateFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.dateFormat = "HH:mm:ss.SSS"
    return formatter
}()

/// Write a message to the iOS log file
/// Used for debugging inference issues on device
private func writeToLogFile(_ message: String) {
    let timestamp = logDateFormatter.string(from: Date())
    let logMessage = "[\(timestamp)] [Moondream3] \(message)\n"

    let fileManager = FileManager.default
    guard let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
        return
    }

    let logFile = documentsDir.appendingPathComponent("moondream_inference.log")

    guard let data = logMessage.data(using: .utf8) else {
        return
    }

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

/// Clear the iOS log file
/// Call this to reset logging for a new session
public func clearLogFile() {
    let fileManager = FileManager.default
    guard let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
        return
    }

    let logFile = documentsDir.appendingPathComponent("moondream_inference.log")
    try? fileManager.removeItem(at: logFile)
}
#endif
