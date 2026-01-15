# iOS Model Download & Selection Screen

## Task Overview

Added an explicit model download/selection screen for the iOS app that matches the macOS Liquid Glass styling. Users can choose which model to download, see download progress, and manage models in Settings.

## Changes Made

### New Files Created

#### MoondreamKit

1. **`MoondreamKit/Sources/MoondreamKit/Models/ModelInfo.swift`**
   - `ModelInfo` struct with id, displayName, description, quantization, sizeBytes, sizeDisplay
   - `AvailableModels` enum with `all` (list of models), `defaultId` (platform-specific), `model(for:)` helper

2. **`MoondreamKit/Sources/MoondreamKit/ModelCache.swift`**
   - `ModelCache.isDownloaded(_:)` - check if model exists in HF cache
   - `ModelCache.getDownloadedModelIds()` - list downloaded model IDs
   - `ModelCache.deleteModel(_:)` - remove model from cache
   - `ModelCache.cacheDirectory(for:)` - get cache URL for model

#### iOS App

3. **`Moondream/Sources/iOS/Views/ModelDownloadView.swift`**
   - Initial download screen with Liquid Glass styling
   - Moon icon (filled), "Moondream" title, "No models found" message
   - List of available models with download buttons
   - Shows download progress with percentage
   - Skip button to defer download

4. **`Moondream/Sources/iOS/Components/ModelRow.swift`**
   - Reusable model row component for Settings
   - Shows model name, quantization, size
   - Selection indicator, download button, or progress indicator

### Modified Files

5. **`MoondreamKit/Sources/MoondreamKit/Moondream3Loader.swift`**
   - Added `configuration(for:)` helper to get ModelConfiguration by model ID

6. **`Moondream/Sources/iOS/Models/AppState.swift`**
   - Added `selectedModelId: String` (persisted to UserDefaults)
   - Added `downloadingModelId: String?` (current download)
   - Added `downloadedModelIds: Set<String>` (cache state)
   - Added `hasDownloadedModels` computed property
   - Added `refreshDownloadedModels()` function

7. **`Moondream/Sources/iOS/Views/ContentView.swift`**
   - Updated flow: check downloaded → show download screen OR load model OR camera
   - Added `NoModelView` for when user skips download
   - Passes selected model ID to MoondreamService

8. **`Moondream/Sources/iOS/Views/SettingsSheet.swift`**
   - Added "Models" section at top of Form
   - Shows downloaded models with selection checkmark
   - Swipe to delete models
   - Shows available models with download buttons
   - Auto-selects another model if selected one is deleted

9. **`Moondream/Sources/Shared/Services/MoondreamService.swift`**
   - Added `loadModel(modelId:)` to load specific model by ID
   - Added `unloadModel()` to free memory
   - Updated inference methods to support both `Moondream3` and `Moondream3Quantized`

10. **`Moondream/Moondream.xcodeproj/project.pbxproj`**
    - Added new files to iOS target

## Model Information

| Model | ID | Quantization | Size |
|-------|-----|--------------|------|
| Moondream 3 Standard | `moondream/md3p-int4` | MoE int4, Vision BF16 | 6.48 GB |
| Moondream 3 Compact | `lewi/md3p-int4-smol` | Full int4 | 5.43 GB |

## App Flow

```
1. Launch App
   ├── No models downloaded?
   │   └── Show ModelDownloadView
   │       ├── User downloads model → Load model → CameraView
   │       └── User skips → NoModelView (prompt to download)
   └── Model downloaded?
       └── Load selected model → CameraView

2. Settings
   └── Models section
       ├── Downloaded models (tap to select, swipe to delete)
       └── Available models (tap to download)
```

## Testing

1. **Fresh Install Test**
   - Delete app/cache
   - Launch app → ModelDownloadView appears
   - Download model → progress visible → navigates to CameraView

2. **Settings Model Management**
   - Open Settings → Models section visible
   - Tap model → selection changes
   - Swipe to delete → model removed, auto-selects another if needed
   - Download available model → progress shown

3. **Model Switching**
   - Download both models
   - Switch between them in Settings
   - Verify inference works with each

## Bug Fixes

### Quantization Shape Error (Task 5)

**Error:** `Fatal error: [quantize] The last dimension of the matrix needs to be divisible by the quantization group size 64. However the provided matrix has shape (1152,588)`

**Cause:** The `QuantizedVision.Encoder.patchEmb` layer was defined as `QuantizedLinear` with input dimension `588` (= 14×14×3 for patch size 14 with 3 RGB channels), which is not divisible by 64.

**Fix:** Changed `patchEmb` from `QuantizedLinear` to regular `Linear` layer in `Moondream3.swift`:
```swift
// Before:
@ModuleInfo(key: "patch_emb") var patchEmb: QuantizedLinear
self._patchEmb.wrappedValue = QuantizedLinear(patchDim, config.encDim, bias: true, groupSize: 64, bits: 4)

// After:
@ModuleInfo(key: "patch_emb") var patchEmb: Linear
self._patchEmb.wrappedValue = Linear(patchDim, config.encDim, bias: true)
```

This follows the same pattern used for `fc2` in `QuantizedVision.MLP` which also uses `Linear` due to `4304 % 64 != 0`.

---

## iOS 26 Liquid Glass Update

### Overview

Updated the iOS app from `.ultraThinMaterial` styling to iOS 26 native Liquid Glass APIs.

### APIs Used

- `GlassEffectContainer` - wraps related views for coordinated glass effects
- `glassEffect(.regular, in: Shape)` - applies glass effect with shape
- `glassEffect(.clear, in: Shape)` - subtle glass for secondary elements

### Files Updated

| File | Changes |
|------|---------|
| `CameraView.swift` | Bottom controls wrapped in `GlassEffectContainer`, SkillIndicator and ProcessingOverlay use `glassEffect` |
| `SettingsButton.swift` | Uses `glassEffect(.regular, in: .circle)` |
| `ModelDownloadView.swift` | Model cards use `glassEffect(.regular, in: .rect)`, skip button uses `.clear` capsule |
| `DownloadProgressView.swift` | Progress/error section wrapped in `GlassEffectContainer` with glass effect |
| `ResultsSheet.swift` | QueryInputSection, PointDetail, BoxDetail use glass; DebugOutput uses `.clear` |
| `project.pbxproj` | iOS deployment target updated to 26.0 |

### Files NOT Updated

- `PointMarkerView.swift` / `BoundingBoxView.swift` - functional detection overlays need contrast
- `CaptureButton.swift` / `CloseButton.swift` - traditional camera button design
- `SettingsSheet.swift` - Form gets automatic iOS 26 styling
