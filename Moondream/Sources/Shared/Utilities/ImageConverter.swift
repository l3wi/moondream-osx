// ImageConverter.swift
// MoondreamCore
//
// Native image loading utility supporting multiple formats.

import Foundation
import CoreImage
import UniformTypeIdentifiers

#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// Errors from image loading
public enum ImageConversionError: LocalizedError {
    case fileNotFound(URL)
    case unsupportedFormat(String)
    case decodingFailed(URL)
    case invalidImageData

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            return "Image file not found: \(url.lastPathComponent)"
        case .unsupportedFormat(let ext):
            return "Unsupported image format: .\(ext)"
        case .decodingFailed(let url):
            return "Failed to decode image: \(url.lastPathComponent)"
        case .invalidImageData:
            return "Invalid or corrupted image data"
        }
    }
}

/// Native image loading utility
///
/// Supports loading images from URLs, NSImage, and raw Data.
/// Uses macOS 15's native format support for AVIF, HEIC, WebP, and other formats.
public struct ImageConverter {

    /// All supported content types for file picker
    public static var supportedContentTypes: [UTType] {
        [
            .png,
            .jpeg,
            .gif,
            .webP,
            .heic,
            .tiff,
            .bmp,
            .pdf,
            .ico,
            UTType("public.heif"),
            UTType("public.avif"),
            UTType("public.jpeg-2000"),
        ].compactMap { $0 }
    }

    /// Supported file extensions
    public static var supportedExtensions: [String] {
        ["png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "avif", "tiff", "tif", "bmp", "jp2", "ico", "pdf"]
    }

    /// Load image from file URL
    /// - Parameter url: URL to the image file
    /// - Returns: CIImage ready for processing
    /// - Throws: ImageConversionError if loading fails
    public static func loadImage(from url: URL) throws -> CIImage {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ImageConversionError.fileNotFound(url)
        }

        // Try CIImage directly (handles most formats)
        if let ciImage = CIImage(contentsOf: url) {
            return ciImage
        }

        #if os(macOS)
        // Fallback to NSImage -> CIImage
        if let nsImage = NSImage(contentsOf: url) {
            return try loadImage(from: nsImage)
        }
        #else
        // Fallback to UIImage -> CIImage
        if let data = try? Data(contentsOf: url),
           let uiImage = UIImage(data: data) {
            return try loadImage(from: uiImage)
        }
        #endif

        throw ImageConversionError.decodingFailed(url)
    }

    #if os(macOS)
    /// Load image from NSImage
    /// - Parameter nsImage: NSImage to convert
    /// - Returns: CIImage ready for processing
    /// - Throws: ImageConversionError if conversion fails
    public static func loadImage(from nsImage: NSImage) throws -> CIImage {
        guard let tiffData = nsImage.tiffRepresentation,
              let ciImage = CIImage(data: tiffData) else {
            throw ImageConversionError.invalidImageData
        }
        return ciImage
    }
    #else
    /// Load image from UIImage
    /// - Parameter uiImage: UIImage to convert
    /// - Returns: CIImage ready for processing
    /// - Throws: ImageConversionError if conversion fails
    public static func loadImage(from uiImage: UIImage) throws -> CIImage {
        guard let ciImage = CIImage(image: uiImage) else {
            throw ImageConversionError.invalidImageData
        }
        return ciImage
    }
    #endif

    /// Load image from raw data
    /// - Parameter data: Image data
    /// - Returns: CIImage ready for processing
    /// - Throws: ImageConversionError if loading fails
    public static func loadImage(from data: Data) throws -> CIImage {
        // Try CIImage directly
        if let ciImage = CIImage(data: data) {
            return ciImage
        }

        #if os(macOS)
        // Try via NSImage
        if let nsImage = NSImage(data: data) {
            return try loadImage(from: nsImage)
        }
        #else
        // Try via UIImage
        if let uiImage = UIImage(data: data) {
            return try loadImage(from: uiImage)
        }
        #endif

        throw ImageConversionError.invalidImageData
    }

    /// Check if a file extension is supported
    /// - Parameter extension: File extension (without dot)
    /// - Returns: true if the format is supported
    public static func isSupported(extension ext: String) -> Bool {
        supportedExtensions.contains(ext.lowercased())
    }
}
