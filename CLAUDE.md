# Moondream MLX - Vision Language Model

## Project Overview

Cross-platform **Moondream3** vision-language model implementation using Apple's MLX framework:
- **MoondreamMac** - macOS CLI app (Swift/MLX) - **WIP, not working yet**
- **moondream-station** - Python backend server (MLX Python) - **Working**

Both implementations run the int4-quantized Moondream3 model for image captioning, visual Q&A, object detection, and pointing.

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Python Backend | Working | Produces coherent captions |
| Swift MoondreamMac | **Working** | Produces coherent captions |

## Architecture

```
moondream-mlx/
├── MoondreamMac/                    # macOS Swift CLI app
│   └── Sources/MoondreamMac/
│       ├── Moondream/
│       │   ├── Moondream3.swift     # Model implementation + inference
│       │   └── Moondream3Loader.swift
│       └── Services/
│           └── MoondreamService.swift
├── moondream-station/               # Python backend (working)
│   └── backends/mlx_backend/
│       ├── backend.py               # API endpoints
│       └── md3/
│           ├── model.py             # Moondream model class
│           ├── text.py              # Text transformer
│           ├── attention.py         # Attention + KV cache
│           └── moe.py               # Mixture of Experts
└── CLAUDE.md                        # This file
```

---

## Building & Testing Swift MoondreamMac

### IMPORTANT: Use Xcode Build, Not `swift run`

**`swift run` DOES NOT WORK** - Metal shaders are not bundled correctly with SwiftPM command-line builds. You will get:

```
Library 'mlx_ext' not found
```

### Build Command

```bash
cd /Users/lewi/Documents/ai/moondream-mlx/MoondreamMac

xcodebuild -scheme MoondreamMac -configuration Debug \
  -destination 'platform=macOS' build
```

### Run Command

After building with Xcode, run the binary from DerivedData:

```bash
# Find the built binary
/Users/lewi/Library/Developer/Xcode/DerivedData/MoondreamMac-cwetnnqvbrdycxdurstgfuteenrk/Build/Products/Debug/MoondreamMac.app/Contents/MacOS/MoondreamMac <image> <skill>

# Example with test image
/Users/lewi/Library/Developer/Xcode/DerivedData/MoondreamMac-cwetnnqvbrdycxdurstgfuteenrk/Build/Products/Debug/MoondreamMac.app/Contents/MacOS/MoondreamMac \
  /Users/lewi/Documents/ai/moondream-mlx/monalisa-on-a-gallery-wall.png caption:normal
```

### Skills

```bash
# Caption (short/normal/long)
./MoondreamMac <image> caption:normal
./MoondreamMac <image> caption:short
./MoondreamMac <image> caption:long

# Query
./MoondreamMac <image> query "What painting is shown?"
```

### Test Image

```
/Users/lewi/Documents/ai/moondream-mlx/monalisa-on-a-gallery-wall.png
```

---

## Known Issues & Quirks

### 1. Garbage Token Output (Swift)

**Status:** ✅ RESOLVED

**Root causes found and fixed:**
1. **RoPE dimension bug**: Swift used `dim / n_heads = 64` but Python uses `dim / (2 * n_heads) = 32`
2. **KV cache update**: Swift used concatenation instead of proper in-place update semantics
3. **generateLogits slicing**: `(-1)...` produced `[B, 1, dim]` instead of `[B, dim]`

All three bugs were fixed and the model now produces coherent captions.

### 2. gatherSort Shape Mismatch (Swift)

**Status:** Disabled as workaround

The `gatherSort` function in MoE has a broadcast shape error:
```
(5840) and (730,8) cannot be broadcast
```

**Workaround:** `doSort = false` in `QuantizedMoEMLP.callAsFunction()`

### 3. argPartition Returns uint32 (Swift)

MLX Swift's `argPartition` returns `uint32`, but `take()` requires `int32` indices.

**Fix applied:** Cast indices with `.asType(.int32)` before passing to `take()`

### 4. Metal Library Not Found with swift run

SwiftPM CLI builds don't bundle the Metal shader library correctly.

**Fix:** Always build with `xcodebuild`, not `swift build` / `swift run`

---

## Testing Python Backend

```bash
cd moondream-station

# Clear bytecode cache first
find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} +

# Run test
python -c "
from PIL import Image
from backends.mlx_backend import backend
import base64, io

backend._model = None
backend._quantize_mode = True

image = Image.open('/Users/lewi/Documents/ai/moondream-mlx/monalisa-on-a-gallery-wall.png').convert('RGB')
buf = io.BytesIO()
image.save(buf, format='PNG')
b64 = base64.b64encode(buf.getvalue()).decode()

result = backend.caption(f'data:image/png;base64,{b64}')
print(result)
"
```

**Expected:** Coherent description like "The image shows the Mona Lisa painting..."

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

### Prompt Templates

```python
# Caption prompts (token sequences)
caption_short:  [1, 32708, 2, 12492, 3]
caption_normal: [1, 32708, 2, 6382, 3]
caption_long:   [1, 32708, 2, 4059, 3]

# Query prompt: prefix + question_tokens + suffix
query_prefix: [1, 15381, 2]
query_suffix: [3]
```

### Inference Flow

1. **Encode image** → `img_emb` (shape: `[1, 729, 2048]`)
2. **Create BOS embedding** → embed token 0
3. **Concatenate** → `[BOS, img_emb]` (shape: `[1, 730, 2048]`)
4. **Allocate KV cache** → pre-sized arrays for all 24 layers
5. **Prefill image** → run forward pass, populate cache
6. **Prefill prompt tokens** → continue forward pass with cache
7. **Generate loop:** sample token, embed, decode with cache, repeat until EOS

### Model Architecture

- **Vision Encoder**: 27 ViT layers, 1152-dim, 16 heads
- **Text Model**: 24 layers, 2048-dim, 32 heads
- **MoE**: 64 experts, 8 active per token (layers 4-23)
- **Layers 0-3**: Standard MLP (not MoE)

### Weight Key Mappings (HF → Swift)

| HuggingFace Key | Swift Model Key |
|-----------------|-----------------|
| `text.blocks.N.*` | `text.layers.N.*` |
| `text.wte.*` | `text.embed_tokens.*` |
| `vision.blocks.N.*` | `vision.layers.N.*` |

---

## Dependencies

- **mlx-swift** (0.21.0+)
- **mlx-swift-examples** (2.21.0+) - MLXVLM, MLXLMCommon
- **swift-transformers** (1.0.0+) - Hub, Tokenizers

## Model

- **ID**: `moondream/md3p-int4`
- **Size**: ~2GB (int4 quantized)
- **Tokenizer**: `moondream/starmie-v1` (vocab_size 51200)

---

## TODO

- [x] Fix Python mx.compile() mask issue
- [x] Implement Swift KV cache infrastructure
- [x] Fix gather indices dtype crash
- [x] **Fix Swift garbage token output** (RoPE dim, KV cache, generateLogits bugs)
- [x] Verify Swift produces correct captions
- [ ] Fix gatherSort shape issue for MoE (currently disabled)
- [ ] Performance profiling
- [ ] Remove debug print statements
