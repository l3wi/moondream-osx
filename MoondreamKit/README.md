# MoondreamKit

A Swift package for running the Moondream3 vision-language model on Apple Silicon using MLX.

## Requirements

- macOS 15.0+ / iOS 18.0+
- Apple Silicon (M1 or later)
- Xcode 16.0+

## Installation

Add MoondreamKit to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/l3wi/moondream-osx.git", branch: "main")
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "MoondreamKit", package: "moondream-osx")
        ]
    )
]
```

Or add it via Xcode: File → Add Package Dependencies → enter `https://github.com/l3wi/moondream-osx.git` → select "MoondreamKit" product.

## Usage

### Loading the Model

```swift
import MoondreamKit

// Load the model container
let container = try await Moondream3Loader.loadContainer(
    configuration: Moondream3Loader.defaultConfiguration
) { progress in
    print("Loading: \(Int(progress.fractionCompleted * 100))%")
}

// The model weights will be downloaded from HuggingFace on first launch
```

### Skills

MoondreamKit provides four vision-language skills:

#### Caption - Generate image descriptions

```swift
container.perform { context in
    let result = model.caption(
        pixels: imagePixels,
        length: "normal",  // "short", "normal", or "long"
        tokenizer: context.tokenizer,
        maxTokens: 768,
        temperature: 0.0
    )
    print(result)
}
```

#### Query - Ask questions about images

```swift
container.perform { context in
    let result = model.query(
        pixels: imagePixels,
        question: "What color is the car?",
        tokenizer: context.tokenizer,
        maxTokens: 768,
        temperature: 0.0
    )
    print(result)
}
```

#### Point - Locate objects in images

```swift
container.perform { context in
    let result = model.point(
        pixels: imagePixels,
        object: "the red button",
        tokenizer: context.tokenizer,
        maxTokens: 256,
        temperature: 0.0
    )
    // Returns JSON with normalized coordinates
}
```

#### Detect - Find and bound objects

```swift
container.perform { context in
    let result = model.detect(
        pixels: imagePixels,
        object: "cars",
        tokenizer: context.tokenizer,
        maxTokens: 256,
        temperature: 0.0
    )
    // Returns JSON with bounding boxes
}
```

### Result Types

```swift
// Point results
struct NormalizedPoint: Identifiable, Equatable, Sendable {
    let x: CGFloat      // 0.0 - 1.0
    let y: CGFloat      // 0.0 - 1.0
    let label: String?
}

// Detection results
struct NormalizedBox: Identifiable, Equatable, Sendable {
    let xMin: CGFloat   // 0.0 - 1.0
    let yMin: CGFloat
    let xMax: CGFloat
    let yMax: CGFloat
    let label: String?
    let confidence: Float?
}
```

## Model Details

- **Model**: Moondream3 (int4 quantized)
- **Size**: ~2GB
- **Source**: `moondream/md3p-int4` on HuggingFace

The model weights are automatically downloaded on first use and cached locally.

## Building

**Important**: Due to Metal shader bundling requirements, this package must be built via Xcode rather than `swift build`. When integrating into your project, use the Xcode build system.

```bash
# Build in Xcode
xcodebuild -scheme YourApp -configuration Debug build
```

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (0.21.0+)
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) (2.21.0+)
- [swift-transformers](https://github.com/huggingface/swift-transformers) (1.0.0+)

## License

See LICENSE file for details.
