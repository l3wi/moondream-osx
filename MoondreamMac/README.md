# MoondreamMac

[![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%2015+-blue.svg)](https://developer.apple.com/macos/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A native Swift implementation of [Moondream3](https://moondream.ai) vision-language model using Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework. Runs entirely on Apple Silicon with Metal acceleration.

## Features

- **Caption**: Generate image descriptions (short, normal, or long)
- **Query**: Ask questions about images
- **Point**: Locate objects by returning coordinates
- **Detect**: Find objects with bounding boxes

## Requirements

- macOS 15.0+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 16.0+ (for building)

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/moondream/MoondreamMac.git", from: "1.0.0")
]
```

Then add `MoondreamCore` to your target dependencies:

```swift
.target(
    name: "YourApp",
    dependencies: ["MoondreamCore"]
)
```

## Usage

### Command Line

```bash
# Build with Xcode (required for Metal shaders)
xcodebuild -scheme MoondreamMac -configuration Release build

# Caption an image
./MoondreamMac /path/to/image.png caption:normal

# Ask a question
./MoondreamMac /path/to/image.png query "What is in this image?"

# Caption lengths: short, normal, long
./MoondreamMac /path/to/image.png caption:short
./MoondreamMac /path/to/image.png caption:long
```

### Programmatic

```swift
import MoondreamCore

// Load the model
let container = try await Moondream3Loader.loadContainer(
    configuration: Moondream3Loader.defaultConfiguration
)

// Caption an image
let result = try await container.perform { context in
    guard let model = context.model as? Moondream3 else {
        throw MoondreamError.inferenceError("Invalid model")
    }

    return model.caption(
        pixels: processedImage,
        length: "normal",
        tokenizer: context.tokenizer,
        maxTokens: 512,
        temperature: 0.0
    )
}
```

## Building

> **Important**: You must build with Xcode, not `swift build`. MLX requires Metal shader libraries that are only properly bundled in Xcode builds.

```bash
# Debug build
xcodebuild -scheme MoondreamMac -configuration Debug build

# Release build
xcodebuild -scheme MoondreamMac -configuration Release build

# Run tests (ConfigurationTests work with swift test)
swift test --filter ConfigurationTests

# Full test suite requires Xcode
xcodebuild -scheme MoondreamMac test
```

## Architecture

```
MoondreamMac/
├── Sources/
│   ├── MoondreamCore/           # Core model library
│   │   ├── Moondream3.swift     # Model implementation
│   │   └── Moondream3Loader.swift
│   └── MoondreamMac/            # CLI application
│       ├── Models/              # Data types
│       └── Services/            # Business logic
├── Tests/
│   └── MoondreamMacTests/       # Unit & integration tests
└── Package.swift
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
