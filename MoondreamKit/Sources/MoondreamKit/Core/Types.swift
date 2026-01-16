import Foundation
import CoreGraphics

// MARK: - Skill

/// Available Moondream vision skills
public enum Skill: String, CaseIterable, Identifiable, Codable, Sendable {
    case caption = "caption"
    case query = "query"
    case point = "point"
    case detect = "detect"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .query: "Query"
        case .caption: "Caption"
        case .point: "Point"
        case .detect: "Detect"
        }
    }

    public var icon: String {
        switch self {
        case .query: "bubble.left.and.bubble.right"
        case .caption: "text.alignleft"
        case .point: "scope"
        case .detect: "square.dashed"
        }
    }

    public var description: String {
        switch self {
        case .query:
            "Ask questions about images and get intelligent answers"
        case .caption:
            "Generate natural language descriptions of images"
        case .point:
            "Identify and locate specific elements by coordinates"
        case .detect:
            "Detect and identify objects with bounding boxes"
        }
    }

    /// Whether this skill requires a target object input
    public var requiresObjectInput: Bool {
        switch self {
        case .query, .caption: false
        case .point, .detect: true
        }
    }

    /// Whether this skill requires any text input before running
    public var requiresInput: Bool {
        switch self {
        case .caption: false
        case .query, .point, .detect: true
        }
    }
}

// MARK: - Caption Length

/// Caption length options for the caption skill
public enum CaptionLength: String, CaseIterable, Identifiable, Codable, Sendable {
    case short = "short"
    case normal = "normal"
    case long = "long"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .short: "Short"
        case .normal: "Normal"
        case .long: "Long"
        }
    }
}

// MARK: - Coordinate Types

/// A point in normalized coordinates (0.0-1.0)
public struct NormalizedPoint: Identifiable, Codable, Equatable, Sendable {
    public let id: UUID
    public let x: CGFloat
    public let y: CGFloat
    public let label: String?

    public init(id: UUID = UUID(), x: CGFloat, y: CGFloat, label: String? = nil) {
        self.id = id
        self.x = x
        self.y = y
        self.label = label
    }

    /// Convert to pixel position for a given view size
    public func position(in size: CGSize) -> CGPoint {
        CGPoint(x: x * size.width, y: y * size.height)
    }
}

/// A bounding box in normalized coordinates (0.0-1.0)
public struct NormalizedBox: Identifiable, Codable, Equatable, Sendable {
    public let id: UUID
    public let xMin: CGFloat
    public let yMin: CGFloat
    public let xMax: CGFloat
    public let yMax: CGFloat
    public let label: String?
    public let confidence: Float?

    public init(
        id: UUID = UUID(),
        xMin: CGFloat,
        yMin: CGFloat,
        xMax: CGFloat,
        yMax: CGFloat,
        label: String? = nil,
        confidence: Float? = nil
    ) {
        self.id = id
        self.xMin = xMin
        self.yMin = yMin
        self.xMax = xMax
        self.yMax = yMax
        self.label = label
        self.confidence = confidence
    }

    public var width: CGFloat { xMax - xMin }
    public var height: CGFloat { yMax - yMin }
    public var centerX: CGFloat { (xMin + xMax) / 2 }
    public var centerY: CGFloat { (yMin + yMax) / 2 }

    /// Convert to frame for a given view size
    public func frame(in size: CGSize) -> CGRect {
        CGRect(
            x: xMin * size.width,
            y: yMin * size.height,
            width: width * size.width,
            height: height * size.height
        )
    }

    /// Get center position in pixels
    public func center(in size: CGSize) -> CGPoint {
        CGPoint(x: centerX * size.width, y: centerY * size.height)
    }
}

// MARK: - Result Types

/// Result types for each Moondream skill
public enum MoondreamResult: Equatable, Sendable {
    case query(QueryResult)
    case caption(CaptionResult)
    case point(PointResult)
    case detect(DetectResult)

    public var displayText: String {
        switch self {
        case .query(let result):
            result.answer
        case .caption(let result):
            result.caption
        case .point(let result):
            "Found \(result.points.count) point(s)"
        case .detect(let result):
            "Detected \(result.boxes.count) object(s)"
        }
    }

    public var points: [NormalizedPoint]? {
        if case .point(let result) = self {
            return result.points
        }
        return nil
    }

    public var boxes: [NormalizedBox]? {
        if case .detect(let result) = self {
            return result.boxes
        }
        return nil
    }
}

/// Result from query skill
public struct QueryResult: Equatable, Sendable {
    public let answer: String
    public let reasoning: String?
    public let rawOutput: String

    public init(answer: String, reasoning: String? = nil, rawOutput: String = "") {
        self.answer = answer
        self.reasoning = reasoning
        self.rawOutput = rawOutput
    }
}

/// Result from caption skill
public struct CaptionResult: Equatable, Sendable {
    public let caption: String
    public let rawOutput: String

    public init(caption: String, rawOutput: String = "") {
        self.caption = caption
        self.rawOutput = rawOutput
    }
}

/// Result from point skill
public struct PointResult: Equatable, Sendable {
    public let points: [NormalizedPoint]
    public let rawOutput: String

    public init(points: [NormalizedPoint], rawOutput: String = "") {
        self.points = points
        self.rawOutput = rawOutput
    }
}

/// Result from detect skill
public struct DetectResult: Equatable, Sendable {
    public let boxes: [NormalizedBox]
    public let rawOutput: String

    public init(boxes: [NormalizedBox], rawOutput: String = "") {
        self.boxes = boxes
        self.rawOutput = rawOutput
    }
}

// MARK: - Model Info

/// Information about an available Moondream model
public struct ModelInfo: Identifiable, Sendable, Equatable, Hashable {
    /// HuggingFace repository ID (e.g., "moondream/md3p-int4")
    public let id: String

    /// Display name for the UI (e.g., "Moondream 3 Standard")
    public let displayName: String

    /// Short description of the model
    public let description: String

    /// Quantization details (e.g., "MoE int4, Vision BF16")
    public let quantization: String

    /// Model size in bytes
    public let sizeBytes: Int64

    /// Human-readable size string (e.g., "6.48 GB")
    public var sizeDisplay: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }

    /// Maximum model size recommended for iOS (5.5 GB)
    public static let iOSMaxSizeBytes: Int64 = 5_905_580_032  // 5.5 GB

    /// Maximum model size for typical 16GB macOS machines
    public static let macOS16GBMaxSizeBytes: Int64 = 10_000_000_000  // ~10 GB

    /// Whether this model is compatible with iOS (under 5.5 GB)
    public var isCompatibleWithiOS: Bool {
        sizeBytes <= Self.iOSMaxSizeBytes
    }

    /// Whether this model requires high-memory macOS (32GB+ recommended)
    public var requiresHighMemoryMac: Bool {
        sizeBytes > Self.macOS16GBMaxSizeBytes
    }

    public init(
        id: String,
        displayName: String,
        description: String,
        quantization: String,
        sizeBytes: Int64
    ) {
        self.id = id
        self.displayName = displayName
        self.description = description
        self.quantization = quantization
        self.sizeBytes = sizeBytes
    }
}

/// Available Moondream models
public enum AvailableModels {
    /// All available models
    public static let all: [ModelInfo] = [
        ModelInfo(
            id: "lewi/md3p-int8",
            displayName: "Moondream 3 Int8",
            description: "Highest quality, requires 16GB+ RAM (macOS only)",
            quantization: "Int8",
            sizeBytes: 10_960_000_000  // ~10.2 GB
        ),
        ModelInfo(
            id: "moondream/md3p-int4",
            displayName: "Moondream 3 Standard",
            description: "Best quality, recommended for devices with 8GB+ RAM",
            quantization: "MoE int4, Vision BF16",
            sizeBytes: 6_960_000_000  // ~6.48 GB
        ),
        ModelInfo(
            id: "lewi/md3p-int4-smol",
            displayName: "Moondream 3 Compact",
            description: "Optimized for memory, recommended for iOS",
            quantization: "Full int4",
            sizeBytes: 5_830_000_000  // ~5.43 GB
        )
    ]

    /// Models compatible with iOS (filtered by size)
    public static var iOSCompatible: [ModelInfo] {
        all.filter { $0.isCompatibleWithiOS }
    }

    /// Default model ID for each platform
    public static var defaultId: String {
        #if os(iOS)
        return "lewi/md3p-int4-smol"
        #else
        return "moondream/md3p-int4"
        #endif
    }

    /// Get model info by ID
    public static func model(for id: String) -> ModelInfo? {
        all.first { $0.id == id }
    }

    /// Get model info for the default model
    public static var defaultModel: ModelInfo {
        model(for: defaultId) ?? all[0]
    }
}
