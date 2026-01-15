# Moondream MLX

Vision-language model implementations using Apple's MLX framework for on-device inference.

## Components

| Component | Platform | Status |
|-----------|----------|--------|
| **Moondream** | macOS + iOS (Swift) | Working |

## Model

Uses [moondream/md3p-int4](https://huggingface.co/moondream/md3p-int4) - an int4 quantized Moondream3 model (~2GB).

## Features

- **Query**: Ask questions about images
- **Caption**: Generate descriptions (short/normal/long)
- **Point**: Locate objects by coordinates
- **Detect**: Find objects with bounding boxes

## Moondream App

Unified Swift app with both macOS and iOS targets.

### macOS Features

- **Liquid Glass UI** - Modern semi-transparent design with hidden title bar
- **Drag & Drop** - Drop images directly onto the app
- **4 Skills** - Caption, Query, Point, Detect
- **Real-time Processing** - Progress indicators and streaming results

### iOS Features

- **Camera-First UI** - Live camera preview with capture
- **Real-time Analysis** - Analyze captured images instantly
- **Settings Sheet** - Configure caption length and other options
- **Results Overlay** - Display bounding boxes and point markers

### Requirements

- macOS 15+ / iOS 18+
- Apple Silicon (M1/M2/M3/M4) for Mac, A12+ for iOS
- Xcode 16+

### Build & Run

```bash
# Open in Xcode (recommended)
open Moondream/Moondream.xcworkspace

# Or build from command line
cd Moondream

# Build macOS
xcodebuild -scheme Moondream-macOS -configuration Debug -destination 'platform=macOS' build

# Build iOS
xcodebuild -scheme Moondream-iOS -destination 'platform=iOS Simulator,name=iPhone 16' build
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
├── Moondream/                       # Swift app (macOS + iOS)
│   ├── Moondream.xcworkspace
│   ├── Moondream.xcodeproj
│   ├── App/
│   │   ├── macOS/                   # macOS Info.plist, entitlements
│   │   └── iOS/                     # iOS Info.plist, entitlements
│   ├── Sources/
│   │   ├── Shared/                  # Cross-platform code
│   │   │   ├── Models/              # Skill, CaptionLength, etc.
│   │   │   ├── Services/            # MoondreamService
│   │   │   ├── Utilities/           # ImageConverter
│   │   │   └── Moondream/           # Core ML model
│   │   │       ├── Moondream3.swift
│   │   │       └── Moondream3Loader.swift
│   │   ├── macOS/                   # Mac-specific UI
│   │   │   ├── MoondreamApp.swift
│   │   │   ├── Views/
│   │   │   └── Components/
│   │   └── iOS/                     # iOS-specific UI
│   │       ├── MoondreamApp.swift
│   │       ├── Services/            # CameraService
│   │       ├── Views/
│   │       └── Components/
│   └── Assets/
└── CLAUDE.md                        # Development notes
```

## License

MIT
