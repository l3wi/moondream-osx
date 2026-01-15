# MoondreamMac Tests

## Test Categories

### ConfigurationTests
Pure Swift tests that verify architecture constants and bug fix calculations.
**Can run with `swift test`.**

### RoPETests
Tests for Rotary Position Embedding implementation.
**Requires Xcode** (Metal/MLX dependency).

### KVCacheTests
Tests for KV cache allocation and update operations.
**Requires Xcode** (Metal/MLX dependency).

### IntegrationTests
End-to-end tests including image processing with the Mona Lisa test image.
**Partially requires Xcode** - some tests are pure Swift, others require MLX.

## Running Tests

### Swift Package Manager (limited)
Only runs pure-Swift tests (ConfigurationTests):
```bash
cd MoondreamMac
swift test --filter ConfigurationTests
```

### Xcode (full test suite)
Required for MLX-dependent tests:
```bash
cd MoondreamMac
xcodebuild -scheme MoondreamMac -configuration Debug \
  -destination 'platform=macOS' test
```

Or open in Xcode and run tests with Cmd+U.

## Why Xcode is Required for MLX Tests

`swift test` doesn't bundle Metal shader libraries correctly. MLX requires Metal for GPU operations, and attempting to run MLX tests via `swift test` will crash with:
```
MLX error: Failed to load the default metallib. library not found
```

The tests include a `requireMLX()` helper that attempts to skip Metal tests when running via `swift test`, but due to MLX library loading order, crashes may still occur.

## Test Image

The integration tests use a Mona Lisa test image located at:
```
/Users/lewi/Documents/ai/moondream-mlx/monalisa-on-a-gallery-wall.png
```

Expected captions include keywords: "Mona Lisa", "Louvre", "painting", "museum", "gallery"
