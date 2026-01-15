# Moondream

[![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%2015+%20|%20iOS%2018+-blue.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A native Swift implementation of [Moondream3](https://moondream.ai) vision-language model using Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework. Runs entirely on Apple Silicon with Metal acceleration.

## Platforms

| Platform | UI Design | Primary Input |
|----------|-----------|---------------|
| **macOS** | Liquid Glass 2-panel | Drag & drop images |
| **iOS** | Camera-first | Live camera capture |

## Features

- **Caption**: Generate image descriptions (short, normal, or long)
- **Query**: Ask questions about images
- **Point**: Locate objects by returning coordinates
- **Detect**: Find objects with bounding boxes

## Requirements

- macOS 15.0+ / iOS 18.0+
- Apple Silicon (M1/M2/M3/M4 for Mac, A12+ for iOS)
- Xcode 16.0+ (for building)

## Building

> **Important**: You must build with Xcode, not `swift build`. MLX requires Metal shader libraries that are only properly bundled in Xcode builds.

```bash
# Open workspace
open Moondream.xcworkspace

# Build macOS (Debug)
xcodebuild -scheme Moondream-macOS -configuration Debug build

# Build macOS (Release)
xcodebuild -scheme Moondream-macOS -configuration Release build

# Build iOS Simulator
xcodebuild -scheme Moondream-iOS -destination 'platform=iOS Simulator,name=iPhone 16' build

# Build iOS Device
xcodebuild -scheme Moondream-iOS -destination 'generic/platform=iOS' build
```

## Usage

### macOS GUI

Launch the app and drag-and-drop an image onto the window. Use the toolbar to select skills (Caption, Query, Point, Detect) and run inference.

### macOS CLI

```bash
# Caption an image
./Moondream /path/to/image.png caption:normal

# Ask a question
./Moondream /path/to/image.png query "What is in this image?"

# Locate an object
./Moondream /path/to/image.png point "the red button"

# Detect objects
./Moondream /path/to/image.png detect "people"

# Caption lengths: short, normal, long
./Moondream /path/to/image.png caption:short
./Moondream /path/to/image.png caption:long
```

### iOS

Launch the app and grant camera permission. Point at objects and tap capture to analyze. Results display as overlays on the captured image.

## Architecture

```
Moondream/
├── App/
│   ├── macOS/                   # macOS Info.plist, entitlements
│   └── iOS/                     # iOS Info.plist, entitlements
├── Sources/
│   ├── Shared/                  # Cross-platform code
│   │   ├── Models/              # Skill, CaptionLength, MoondreamResult
│   │   ├── Services/            # MoondreamService
│   │   ├── Utilities/           # ImageConverter
│   │   └── Moondream/           # Core ML model
│   │       ├── Moondream3.swift
│   │       └── Moondream3Loader.swift
│   ├── macOS/                   # Mac-specific UI
│   │   ├── MoondreamApp.swift
│   │   ├── Views/               # ContentView, ImagePanel, etc.
│   │   └── Components/          # ToolbarPanel, SkillTabBar, etc.
│   └── iOS/                     # iOS-specific UI
│       ├── MoondreamApp.swift
│       ├── Services/            # CameraService
│       ├── Views/               # CameraView, ResultsSheet, etc.
│       └── Components/          # CaptureButton, SettingsButton, etc.
└── Assets/
    ├── Assets.xcassets          # Shared assets
    └── AppIcon.appiconset       # App icons
```

### Model Architecture

- **Vision Encoder**: 27-layer ViT, 1152-dim, 16 heads
- **Text Model**: 24-layer transformer, 2048-dim, 32 heads
- **MoE**: 64 experts, 8 active per token (layers 4-23)
- **Quantization**: int4 (~2GB model size)

## Known Limitations

1. **MoE Sorting Disabled**: The `gatherSort` function for MoE expert selection is disabled due to shape broadcast issues. Inference still works but may be slightly less efficient.

2. **Metal Required**: MLX operations require Metal GPU acceleration. The model cannot run on Intel Macs or in environments without Metal support.

3. **swift test Limitations**: MLX-dependent tests crash with `swift test` because Metal shader libraries aren't bundled. Use Xcode for full test suite.

## Model

The model weights are automatically downloaded from Hugging Face:
- **Repository**: `moondream/md3p-int4`
- **Size**: ~2GB (int4 quantized)
- **Tokenizer**: `moondream/starmie-v1` (51,200 vocab)

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) - Apple's ML framework
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) - MLXVLM, MLXLMCommon
- [swift-transformers](https://github.com/huggingface/swift-transformers) - Tokenizers, Hub

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Moondream AI](https://moondream.ai) for the model architecture
- [Apple MLX Team](https://github.com/ml-explore) for the Swift ML framework
