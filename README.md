# Moondream MLX

Vision-language model implementations using Apple's MLX framework for on-device inference.

## Components

| Component | Platform | Status |
|-----------|----------|--------|
| **MoondreamMac** | macOS (Swift) | Working |

## Model

Uses [moondream/md3p-int4](https://huggingface.co/moondream/md3p-int4) - an int4 quantized Moondream3 model (~2GB).

## Features

- **Query**: Ask questions about images
- **Caption**: Generate descriptions (short/normal/long)
- **Point**: Locate objects by coordinates
- **Detect**: Find objects with bounding boxes

## MoondreamMac

Native macOS app with Liquid Glass UI design. Supports both GUI and CLI modes.

### Features

- **Liquid Glass UI** - Modern semi-transparent design with hidden title bar
- **Drag & Drop** - Drop images directly onto the app
- **4 Skills** - Caption, Query, Point, Detect
- **Real-time Processing** - Progress indicators and streaming results

### Requirements

- macOS 14+ (macOS 26+ for full Liquid Glass effects)
- Apple Silicon (M1/M2/M3)
- Xcode 16+

### Build & Run

```bash
# Open in Xcode (recommended)
open MoondreamMac/MoondreamMac.xcworkspace

# Or build from command line
cd MoondreamMac
xcodebuild -scheme MoondreamMac -configuration Debug -destination 'platform=macOS' build
```

**Note:** `swift run` does not work due to Metal shader bundling issues. Must use Xcode build.

### CLI Mode

```bash
# Caption
./MoondreamMac image.png caption:normal

# Query
./MoondreamMac image.png query "What is in this image?"

# Point (locate object)
./MoondreamMac image.png point "the red button"

# Detect (bounding boxes)
./MoondreamMac image.png detect "people"
```

## Architecture

```
moondream-mlx/
├── MoondreamMac/                    # Swift macOS app
│   └── Sources/
│       ├── MoondreamCore/           # ML model library
│       │   ├── Moondream3.swift     # Model implementation
│       │   └── Moondream3Loader.swift
│       └── MoondreamMac/            # GUI app
│           ├── MoondreamMacApp.swift
│           ├── ContentView.swift
│           ├── Views/               # Main view components
│           │   ├── EmptyStateView.swift
│           │   ├── ImageLoadedView.swift
│           │   ├── ImagePanel.swift
│           │   └── ToolbarPanel.swift
│           ├── Components/          # Reusable UI components
│           │   ├── SkillTabBar.swift
│           │   ├── InputField.swift
│           │   ├── ResultsView.swift
│           │   └── RunButton.swift
│           ├── Models/
│           ├── Services/
│           └── Utilities/
└── CLAUDE.md                        # Development notes
```

## License

MIT
