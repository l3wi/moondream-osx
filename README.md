# Moondream MLX

Vision-language model implementations using Apple's MLX framework for on-device inference.

## Components

| Component | Platform | Status |
|-----------|----------|--------|
| **MoondreamKit** | Swift Package | Working - Reusable model package |
| **Moondream** | macOS + iOS | Working - Unified app |

## Models

| Model | ID | Size | Notes |
|-------|-----|------|-------|
| Standard | `moondream/md3p-int4` | 6.48 GB | MoE int4, Vision BF16 |
| Compact | `lewi/md3p-int4-smol` | 5.43 GB | Full int4, iOS optimized |

## Features

- **Query**: Ask questions about images
- **Caption**: Generate descriptions (short/normal/long)
- **Point**: Locate objects by coordinates
- **Detect**: Find objects with bounding boxes

## Moondream App

Unified Swift app with both macOS and iOS targets, built on MoondreamKit.

### macOS Features

- **Liquid Glass UI** - Modern semi-transparent design with hidden title bar
- **Drag & Drop** - Drop images directly onto the app
- **4 Skills** - Caption, Query, Point, Detect
- **Real-time Processing** - Progress indicators and streaming results
- **CLI Mode** - Command-line interface for scripting

### iOS Features

- **iOS 26 Liquid Glass** - Native `glassEffect` and `GlassEffectContainer` APIs
- **Camera-First UI** - Live camera preview with capture
- **Model Selection** - Download and switch between Standard/Compact models
- **Download Progress** - Visual progress with cancel support
- **Results Overlay** - Display bounding boxes and point markers

### Requirements

- macOS 15+ / iOS 26+
- Apple Silicon (M1/M2/M3/M4) for Mac
- iPhone 15 Pro or newer for iOS (8GB+ RAM recommended)
- Xcode 17+

### Build & Run

```bash
# Open in Xcode (recommended)
open Moondream/Moondream.xcworkspace

# Or build from command line
cd Moondream

# Build macOS
xcodebuild -scheme Moondream-macOS -configuration Debug build

# Build iOS
xcodebuild -scheme Moondream-iOS -destination 'generic/platform=iOS Simulator' build
```

**Note:** `swift run` does not work due to Metal shader bundling issues. Must use Xcode build.

### CLI Mode (macOS)

```bash
# Caption
./Moondream image.png caption:normal

# Query
./Moondream image.png query "What is in this image?"

# Point (locate object)
./Moondream image.png point "the red button"

# Detect (bounding boxes)
./Moondream image.png detect "people"
```

## Architecture

```
moondream-mlx/
├── MoondreamKit/                    # Swift Package (reusable)
│   ├── Package.swift
│   ├── Sources/MoondreamKit/
│   │   ├── Moondream3.swift         # Standard model
│   │   ├── Moondream3Quantized.swift # Compact model (in same file)
│   │   ├── Moondream3Loader.swift   # Model loading from HuggingFace
│   │   ├── ModelCache.swift         # Cache detection/deletion
│   │   └── Models/
│   │       ├── ModelInfo.swift      # Model metadata
│   │       ├── Skill.swift          # caption/query/point/detect
│   │       └── ...
│   └── Tests/
├── Moondream/                       # Unified Swift app (macOS + iOS)
│   ├── Moondream.xcworkspace
│   ├── Sources/
│   │   ├── Shared/                  # Cross-platform code
│   │   │   └── Services/MoondreamService.swift
│   │   ├── macOS/                   # Mac-specific UI
│   │   └── iOS/                     # iOS-specific UI
│   │       ├── Views/
│   │       │   ├── CameraView.swift
│   │       │   ├── ModelDownloadView.swift
│   │       │   └── ...
│   │       └── Components/
└── docs/tasks/                      # Task documentation
```

## License

MIT
