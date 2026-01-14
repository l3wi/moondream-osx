# Moondream MLX

Vision-language model implementations using Apple's MLX framework for on-device inference.

## Components

| Component | Platform | Status |
|-----------|----------|--------|
| **moondream-station** | macOS (Python) | Working |
| **MoondreamMac** | macOS (Swift) | WIP - Not working |

## Model

Uses [moondream/md3p-int4](https://huggingface.co/moondream/md3p-int4) - an int4 quantized Moondream3 model (~2GB).

## Features

- **Query**: Ask questions about images
- **Caption**: Generate descriptions (short/normal/long)
- **Point**: Locate objects by coordinates
- **Detect**: Find objects with bounding boxes

## Python Backend (Working)

```bash
cd moondream-station

# Test caption
python -c "
from PIL import Image
from backends.mlx_backend import backend
import base64, io

image = Image.open('path/to/image.png').convert('RGB')
buf = io.BytesIO()
image.save(buf, format='PNG')
b64 = base64.b64encode(buf.getvalue()).decode()

result = backend.caption(f'data:image/png;base64,{b64}')
print(result)
"
```

## Swift MoondreamMac (WIP)

The Swift implementation builds and runs but produces garbage output. See [CLAUDE.md](CLAUDE.md) for debugging details.

### Requirements

- macOS 14+
- Apple Silicon (M1/M2/M3)
- Xcode 16+

### Build

```bash
cd MoondreamMac
xcodebuild -scheme MoondreamMac -configuration Debug -destination 'platform=macOS' build
```

**Note:** `swift run` does not work due to Metal shader bundling issues. Must use Xcode build.

### Run

```bash
# Binary location after build
/Users/<you>/Library/Developer/Xcode/DerivedData/MoondreamMac-*/Build/Products/Debug/MoondreamMac.app/Contents/MacOS/MoondreamMac <image> caption:normal
```

## Architecture

```
moondream-mlx/
├── MoondreamMac/           # Swift CLI app (WIP)
│   └── Sources/
│       └── MoondreamMac/
│           ├── Moondream/  # Model implementation
│           └── Services/   # Inference service
├── moondream-station/      # Python backend (working)
│   └── backends/
│       └── mlx_backend/
│           └── md3/        # Model components
└── CLAUDE.md               # Development notes
```

## License

MIT
