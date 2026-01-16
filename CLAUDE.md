# Moondream MLX - Vision Language Model

## Project Overview

Cross-platform **Moondream3** vision-language model implementation using Apple's MLX framework:
- **MoondreamKit** - Swift Package containing the core ML model (reusable by other developers)
- **Moondream** - Unified Swift app with macOS + iOS targets (uses MoondreamKit)

The int4-quantized Moondream3 model supports image captioning, visual Q&A, object detection, and pointing.

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| MoondreamKit Package | **Working** | Standalone Swift Package for model inference |
| Swift macOS App | **Working** | Liquid Glass UI, drag-drop, all 4 skills |
| Swift iOS App | **Working** | iOS 26 Liquid Glass, camera-first UI, model selection |

## Models

| Model | ID | Size | Notes |
|-------|-----|------|-------|
| Standard | `moondream/md3p-int4` | 6.48 GB | MoE int4, Vision BF16, best quality |
| Compact | `lewi/md3p-int4-smol` | 5.43 GB | Full int4, iOS optimized |

## Architecture

```
moondream-mlx/
├── MoondreamKit/                    # Swift Package (reusable)
│   ├── Package.swift
│   ├── README.md
│   ├── Sources/MoondreamKit/
│   │   ├── Moondream3.swift         # Core model (standard + quantized)
│   │   ├── Moondream3Loader.swift   # Model loading from HuggingFace
│   │   ├── ModelCache.swift         # Cache detection/deletion utilities
│   │   └── Models/
│   │       ├── ModelInfo.swift      # Model metadata for UI
│   │       ├── Skill.swift          # caption/query/point/detect
│   │       ├── CaptionLength.swift  # short/normal/long
│   │       ├── CoordinateTypes.swift # NormalizedPoint, NormalizedBox
│   │       └── MoondreamResult.swift # Result types
│   └── Tests/MoondreamKitTests/
├── Moondream/                       # Unified Swift app (macOS + iOS)
│   ├── Moondream.xcodeproj/
│   ├── Moondream.xcworkspace/       # Includes MoondreamKit package
│   ├── Assets.xcassets/
│   ├── App/
│   │   ├── Info.plist               # macOS app config
│   │   └── iOS/Info.plist           # iOS app config
│   └── Sources/
│       ├── Shared/                  # Cross-platform code
│       │   ├── Models/
│       │   │   └── OverlayModels.swift  # JSON parsing helpers
│       │   ├── Services/
│       │   │   └── MoondreamService.swift
│       │   └── Utilities/
│       │       └── ImageConverter.swift
│       ├── macOS/                   # Mac-specific UI
│       │   ├── MoondreamApp.swift
│       │   ├── Views/
│       │   └── Components/
│       └── iOS/                     # iOS-specific UI
│           ├── MoondreamApp.swift
│           ├── Models/AppState.swift
│           ├── Services/CameraService.swift
│           ├── Views/
│           │   ├── CameraView.swift      # Main camera with glass controls
│           │   ├── ModelDownloadView.swift # Model selection screen
│           │   ├── DownloadProgressView.swift
│           │   ├── ResultsSheet.swift
│           │   └── SettingsSheet.swift
│           └── Components/
│               ├── SettingsButton.swift  # Glass circle button
│               ├── CaptureButton.swift
│               └── ModelRow.swift
├── docs/tasks/                      # Task documentation
└── CLAUDE.md                        # This file
```

---

## Building & Running

### macOS App

```bash
cd /Users/lewi/Documents/ai/moondream-mlx/Moondream

# Build macOS target
xcodebuild -project Moondream.xcodeproj -scheme Moondream-macOS -configuration Debug build

# Or open in Xcode
open Moondream.xcworkspace
# Select "Moondream-macOS" scheme and press ⌘R
```

### iOS App

```bash
cd /Users/lewi/Documents/ai/moondream-mlx/Moondream

# Build iOS target (simulator)
xcodebuild -project Moondream.xcodeproj -scheme Moondream-iOS \
  -destination 'generic/platform=iOS Simulator' -configuration Debug build

# Or open in Xcode
open Moondream.xcworkspace
# Select "Moondream-iOS" scheme and press ⌘R
```

### CLI Usage (macOS only)

The macOS app supports both GUI mode (no args) and CLI mode (with args):

```bash
# GUI mode - opens window
./Moondream

# CLI mode - process and exit
./Moondream <image> <skill> [query]

# Caption (short/normal/long)
./Moondream image.png caption:normal

# Query
./Moondream image.png query "What painting is shown?"

# Point (locate object)
./Moondream image.png point "cat"

# Detect (bounding boxes)
./Moondream image.png detect "person"
```

### Test Image

```
/Users/lewi/Documents/ai/moondream-mlx/monalisa-on-a-gallery-wall.png
```

---

## Platform Differences

| Feature | macOS | iOS |
|---------|-------|-----|
| UI Design | Liquid Glass 2-panel layout | iOS 26 Liquid Glass camera UI |
| Glass APIs | `.ultraThinMaterial` | `glassEffect`, `GlassEffectContainer` |
| Image Input | Drag-drop or file picker | Live camera capture |
| Overlays | Hover to show labels | Tap to show details |
| Model Selection | Auto (Standard) | User choice (Standard/Compact) |
| CLI Support | Yes | No |
| Min Version | macOS 15.0 | iOS 26.0 |

---

## Liquid Glass UI

### macOS Approach

macOS uses SwiftUI's built-in material system for the glass effect:

```swift
// Background material for panels
.background(.ultraThinMaterial)

// Toolbar and sidebar backgrounds
VStack { ... }
    .background(.ultraThinMaterial)

// Hidden title bar for clean look
.windowStyle(.hiddenTitleBar)
```

**Files Using Materials:**
| File | Usage |
|------|-------|
| `ContentView.swift` | Main window with hidden title bar |
| `ToolbarPanel.swift` | Right sidebar with `.ultraThinMaterial` |
| `ImagePanel.swift` | Image display area |
| `WebcamCaptureView.swift` | Full-screen camera preview |

### iOS 26 Approach

The iOS app uses native iOS 26 Liquid Glass APIs:

```swift
// Glass effect on buttons and cards
.glassEffect(.regular, in: .circle)
.glassEffect(.regular, in: .rect(cornerRadius: 12))
.glassEffect(.clear, in: .capsule)  // Subtle variant

// Container for coordinated glass effects
GlassEffectContainer {
    SkillIndicator(...)
    CaptureButton(...)
    SettingsButton(...)
}
```

**Files Using Glass Effects:**
| File | Usage |
|------|-------|
| `CameraView.swift` | Bottom controls in `GlassEffectContainer`, SkillIndicator, ProcessingOverlay |
| `SettingsButton.swift` | `.glassEffect(.regular, in: .circle)` |
| `ModelDownloadView.swift` | Model cards, skip button capsule |
| `DownloadProgressView.swift` | Progress container |
| `ResultsSheet.swift` | Detail cards, debug section |

---

## Known Issues & Quirks

### 1. gatherSort Shape Mismatch (Swift)

**Status:** Disabled as workaround

The `gatherSort` function in MoE has a broadcast shape error.

**Workaround:** `doSort = false` in `QuantizedMoEMLP.callAsFunction()`

### 2. Metal Library Not Found with swift run

SwiftPM CLI builds don't bundle the Metal shader library correctly.

**Fix:** Always build with `xcodebuild`, not `swift build` / `swift run`

### 3. Quantization Shape Error for patchEmb

**Status:** Fixed

The vision encoder's `patchEmb` layer had dimension 588 (14×14×3) which isn't divisible by 64.

**Fix:** Changed `patchEmb` from `QuantizedLinear` to `Linear` in `Moondream3.swift`

---

## Moondream3 Architecture Reference

### Token IDs (from config)

| Token | ID | Notes |
|-------|-----|-------|
| BOS | 0 | Beginning of sequence |
| EOS | 0 | End of sequence (same as BOS!) |
| Answer | 3 | End of reasoning section |
| Thinking | 4 | Start thinking/reasoning |
| Coord | 5 | Coordinate token for pointing |

### Model Architecture

- **Vision Encoder**: 27 ViT layers, 1152-dim, 16 heads
- **Text Model**: 24 layers, 2048-dim, 32 heads
- **MoE**: 64 experts, 8 active per token (layers 4-23)
- **Layers 0-3**: Standard MLP (not MoE)

---

## Dependencies

- **mlx-swift** (0.21.0+)
- **mlx-swift-examples** (2.21.0+) - MLXVLM, MLXLMCommon
- **swift-transformers** (1.0.0+) - Hub, Tokenizers

## Model Files

- **Standard ID**: `moondream/md3p-int4` (~6.48 GB)
- **Compact ID**: `lewi/md3p-int4-smol` (~5.43 GB)
- **Tokenizer**: `moondream/starmie-v1` (vocab_size 51200)

---

## Agent Development Guide

### Code Style & Conventions

**Naming:**
| Category | Convention | Examples |
|----------|------------|----------|
| Files | PascalCase | `CameraService.swift`, `MoondreamService.swift` |
| Types/Classes | PascalCase | `Skill`, `NormalizedPoint`, `MoondreamResult` |
| Functions/Methods | camelCase | `captureCurrentFrame()`, `loadModel()` |
| Variables/Properties | camelCase | `selectedSkill`, `isFrozen`, `isProcessing` |
| Enum cases | lowercase | `.caption`, `.query`, `.point`, `.detect` |

**File Organization:**
```swift
// Use MARK sections to organize code
// MARK: - Properties
// MARK: - Initialization
// MARK: - Public Methods
// MARK: - Private Methods
// MARK: - View Body (for SwiftUI views)
```

**Error Handling:**
```swift
// Custom errors implement LocalizedError + Identifiable
enum MoondreamError: LocalizedError {
    case modelNotLoaded
    case imageConversionFailed
    case inferenceError(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: "Model not loaded"
        // ...
        }
    }
}
```

**Async/Await:**
```swift
// Use @MainActor for UI-related classes
@MainActor
final class MoondreamService: ObservableObject { ... }

// Use Task { @MainActor in } for callbacks
AVCaptureDevice.requestAccess(for: .video) { granted in
    Task { @MainActor in
        self?.handlePermission(granted)
    }
}
```

---

### SwiftUI Patterns

**iOS (Modern @Observable):**
```swift
@MainActor
@Observable
final class AppState {
    var selectedSkill: Skill = .caption
    var isProcessing: Bool = false
}

// In views:
@Environment(AppState.self) private var appState
```

**macOS (Traditional ObservableObject):**
```swift
@MainActor
final class MoondreamService: ObservableObject {
    @Published var isLoaded = false
    static let shared = MoondreamService()
}

// In views:
@EnvironmentObject var service: MoondreamService
@StateObject private var cameraService = MacCameraService()
```

**Bindings for Component Props:**
```swift
// Parent creates binding
SkillTabBar(selection: $selectedSkill)

// Or manual binding
SkillSelectorButton(selectedSkill: Binding(
    get: { appState.selectedSkill },
    set: { appState.selectedSkill = $0 }
))
```

**Platform-Specific Code:**
```swift
#if os(iOS)
private let maxTokens = 128  // Conservative for iOS memory
#else
private let maxTokens = 768
#endif
```

---

### Testing

**Framework:** XCTest

**Test Location:** `Moondream/Tests/MoondreamMacTests/`

**Running Tests:**
- **Xcode:** Select test target → Cmd+U
- **CLI:** `xcodebuild test -scheme Moondream-macOS` (Metal required)
- **Note:** `swift test` won't work due to Metal shader requirements

**Test Files:**
- `ConfigurationTests.swift` - Model constants validation
- `IntegrationTests.swift` - End-to-end with real images
- `RoPETests.swift` - Rotary position embedding tests
- `KVCacheTests.swift` - KV cache shape tests

**Test Pattern:**
```swift
import XCTest
@testable import MoondreamMac

final class MyTests: XCTestCase {
    func testExample() throws {
        // Arrange
        let input = ...
        // Act
        let result = ...
        // Assert
        XCTAssertEqual(result, expected)
    }
}
```

---

### Development Workflow

**Critical: Always Use Xcode Build**
```bash
# CORRECT - use xcodebuild
xcodebuild -scheme Moondream-macOS -configuration Debug build

# WRONG - swift build doesn't bundle Metal shaders
swift build  # Will fail at runtime!
```

**Project Structure:**
- `Moondream.xcworkspace` - Open this (includes MoondreamKit)
- `Moondream.xcodeproj` - The app project
- Schemes: `Moondream-macOS`, `Moondream-iOS`

**Adding Files to Xcode:**
When creating new Swift files, they must be added to `project.pbxproj`:
1. Add `PBXFileReference` entry
2. Add `PBXBuildFile` entry for target
3. Add to appropriate `PBXGroup`
4. Add to `PBXSourcesBuildPhase`

Or just add files via Xcode UI (File → New File).

---

### Common Tasks

**Adding a New macOS View:**
1. Create `Sources/macOS/Views/MyView.swift`
2. Add to Xcode project (macOS target)
3. Follow existing view patterns (GeometryReader, ZStack, etc.)

**Adding a New iOS View:**
1. Create `Sources/iOS/Views/MyView.swift`
2. Add to Xcode project (iOS target)
3. Use `@Environment(AppState.self)` for state
4. Use `.glassEffect()` for iOS 26 Liquid Glass

**Adding a Shared Component:**
1. Create `Sources/Shared/Components/MyComponent.swift`
2. Add to both macOS and iOS targets in Xcode
3. Use `#if os()` for platform differences

**Modifying Model Inference:**
1. Core model: `MoondreamKit/Sources/MoondreamKit/Moondream3.swift`
2. Service layer: `Moondream/Sources/Shared/Services/MoondreamService.swift`
3. Test changes with `images/monalisa-on-a-gallery-wall.png`

**Adding a New Skill:**
1. Add case to `Skill` enum in `MoondreamKit/Sources/MoondreamKit/Models/Skill.swift`
2. Add method to `Moondream3.swift`
3. Add service method to `MoondreamService.swift`
4. Add UI for skill in both macOS and iOS views

---

## TODO

- [x] Fix Swift garbage token output
- [x] Implement Liquid Glass UI (macOS)
- [x] Add iOS camera app
- [x] Merge macOS + iOS into unified project
- [x] Extract core model into MoondreamKit package
- [x] Add model download/selection screen to iOS
- [x] Implement iOS 26 Liquid Glass APIs
- [x] Fix patchEmb quantization shape error
- [x] Add webcam capture to macOS app
- [ ] Fix gatherSort shape issue for MoE
- [ ] Performance profiling
