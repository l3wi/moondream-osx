# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-15

### Added
- **iOS target**: Camera-first UI for iPhone with live preview and capture
- **Unified project**: Single Xcode project with both macOS and iOS targets
- **Cross-platform shared code**: Models, Services, and ML core shared between platforms
- **iOS-specific features**: CameraService, CaptureButton, SettingsSheet, ResultsSheet
- **Platform-conditional compilation**: ImageConverter supports both AppKit and UIKit

### Changed
- **Renamed project**: MoondreamMac → Moondream
- **Reorganized source structure**: Sources/Shared/, Sources/macOS/, Sources/iOS/
- **Updated schemes**: Moondream-macOS and Moondream-iOS

### Removed
- Merged and deleted old MoondreamCamera project

## [1.0.0] - 2024-01-14

### Added
- Initial release of MoondreamMac CLI application
- Support for Moondream3 vision-language model on Apple Silicon
- Four inference skills: caption, query, point, detect
- Caption length options: short, normal, long
- Automatic model download from Hugging Face
- Comprehensive test suite with unit and integration tests

### Fixed
- **Critical**: RoPE dimension calculation (was `dim/n_heads`, now correctly `dim/(2*n_heads)`)
- **Critical**: KV cache update logic now properly inserts at cache position
- **Critical**: `generateLogits` output shape now correctly returns `[B, vocab_size]`
- Token sampling no longer squeezes wrong dimensions

### Known Limitations
- MoE `gatherSort` disabled due to shape broadcast issues (inference still works)
- Requires Xcode build for Metal shader bundling (`swift build` not supported)
- Tests require Xcode for MLX/Metal operations

### Dependencies
- mlx-swift 0.21.0+
- mlx-swift-examples 2.21.0+
- swift-transformers 1.0.0+
