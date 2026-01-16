import Foundation
import MLX

// MARK: - Token Sampling

/// Sample a token from logits
/// - Parameters:
///   - logits: Logits tensor of shape [B, vocab_size] or [vocab_size]
///   - temperature: Sampling temperature (0 = greedy)
/// - Returns: Sampled token ID
internal func sampleToken(logits: MLXArray, temperature: Float) -> Int {
    // logits is [1, vocab_size], squeeze batch dim to get [vocab_size]
    let logits1D = logits.ndim > 1 ? logits.squeezed(axis: 0) : logits

    if temperature == 0 {
        // Greedy decoding
        return Int(argMax(logits1D, axis: -1).item(Int.self))
    } else {
        // Temperature sampling with softmax
        // NOTE: Using argmax after softmax (deterministic). For stochastic sampling,
        // implement multinomial sampling from probability distribution.
        let probs = softmax(logits1D / temperature, axis: -1)
        return Int(argMax(probs, axis: -1).item(Int.self))
    }
}

// MARK: - Coordinate Generation

/// Protocol for models that support coordinate generation
/// Both standard and quantized region models conform to this
internal protocol CoordinateDecoder {
    func encodeCoordinate(_ coord: MLXArray) -> MLXArray
    func decodeCoordinate(_ hiddenState: MLXArray) -> MLXArray
    func encodeSize(_ size: MLXArray) -> MLXArray
    func decodeSize(_ hiddenState: MLXArray) -> MLXArray
}

extension RegionModel: CoordinateDecoder {}
extension QuantizedRegionModel: CoordinateDecoder {}

/// Result type for generated coordinates
internal struct GeneratedCoordinate {
    let x: Float
    let y: Float
    let width: Float?
    let height: Float?

    /// Format as point string "(x, y)"
    var pointString: String {
        String(format: "(%.4f, %.4f)", x, y)
    }

    /// Format as bounding box string "[xmin, ymin, xmax, ymax]"
    var boxString: String {
        guard let w = width, let h = height else {
            return pointString
        }
        let xMin = x - w / 2
        let yMin = y - h / 2
        let xMax = x + w / 2
        let yMax = y + h / 2
        return String(format: "[%.4f, %.4f, %.4f, %.4f]", xMin, yMin, xMax, yMax)
    }
}

/// Decode function type for single token decoding
internal typealias DecodeOneFunc = (
    _ embedding: MLXArray,
    _ cachePos: Int,
    _ cache: inout [(MLXArray, MLXArray)]
) -> (logits: MLXArray, hidden: MLXArray)

/// Generate points/bounding boxes using coordinate decoding from hidden states
/// This is the shared implementation used by both model variants
/// - Parameters:
///   - hidden: Last hidden state from prefill (shape: [1, seq_len, dim])
///   - firstToken: The first sampled token (should be coord_id to start coordinate gen)
///   - startPos: Current position in KV cache
///   - cache: KV cache (modified in place)
///   - regionModel: Region model for coordinate decoding
///   - decodeOne: Function to decode one token
///   - includeSize: If true, decode width/height for bounding boxes (detect mode)
///   - maxObjects: Maximum number of objects to detect
/// - Returns: Array of generated coordinates
internal func generateCoordinates(
    hidden: MLXArray,
    firstToken: Int,
    startPos: Int,
    cache: inout [(MLXArray, MLXArray)],
    regionModel: CoordinateDecoder,
    decodeOne: DecodeOneFunc,
    includeSize: Bool,
    maxObjects: Int
) -> [GeneratedCoordinate] {
    var results: [GeneratedCoordinate] = []
    var pos = startPos
    var currentHidden = hidden
    var currentToken = firstToken

    // Only proceed if we got the coordinate token
    while currentToken != TokenConstants.eos && results.count < maxObjects {
        // Decode X coordinate from hidden state
        let xLogits = regionModel.decodeCoordinate(currentHidden)
        let xIdx = Int(argMax(xLogits.reshaped(-1), axis: -1).item(Int.self))
        let xCenter = Float(xIdx) / Float(xLogits.dim(-1))

        // Create coordinate array and encode
        let xCoord = MLXArray([xCenter]).expandedDimensions(axis: 0)
        var nextEmb = regionModel.encodeCoordinate(xCoord).reshaped(1, 1, -1)

        // Decode step to get hidden state for Y
        let (_, yHidden) = decodeOne(nextEmb, pos, &cache)
        pos += 1

        // Decode Y coordinate
        let yLogits = regionModel.decodeCoordinate(yHidden)
        let yIdx = Int(argMax(yLogits.reshaped(-1), axis: -1).item(Int.self))
        let yCenter = Float(yIdx) / Float(yLogits.dim(-1))
        let yCoord = MLXArray([yCenter]).expandedDimensions(axis: 0)
        nextEmb = regionModel.encodeCoordinate(yCoord).reshaped(1, 1, -1)

        var width: Float? = nil
        var height: Float? = nil

        if includeSize {
            // Decode step to get hidden state for size
            let (_, sizeHidden) = decodeOne(nextEmb, pos, &cache)
            pos += 1

            // Decode size (width and height)
            let sizeLogits = regionModel.decodeSize(sizeHidden[0..., (-1)..., 0...].squeezed(axis: 1))

            let wBin = Int(argMax(sizeLogits[0], axis: -1).item(Int.self))
            let hBin = Int(argMax(sizeLogits[1], axis: -1).item(Int.self))
            width = Float(FourierFeatures.binToSize(MLXArray([wBin])).item(Float.self))
            height = Float(FourierFeatures.binToSize(MLXArray([hBin])).item(Float.self))

            // Create size array with both width and height
            let sizeArr = MLXArray([width!, height!]).expandedDimensions(axis: 0)
            nextEmb = regionModel.encodeSize(sizeArr).reshaped(1, 1, -1)
        }

        results.append(GeneratedCoordinate(x: xCenter, y: yCenter, width: width, height: height))

        // Get next token decision (continue with more coords or stop)
        let (nextLogits, nextHidden) = decodeOne(nextEmb, pos, &cache)
        pos += 1
        currentHidden = nextHidden

        // Choose between coord_id (continue) and eos_id (stop)
        let coordLogit = nextLogits[0..., TokenConstants.coord].item(Float.self)
        let eosLogit = nextLogits[0..., TokenConstants.eos].item(Float.self)
        currentToken = coordLogit > eosLogit ? TokenConstants.coord : TokenConstants.eos
    }

    return results
}

/// Format generated points as string output
/// - Parameters:
///   - coordinates: Array of generated coordinates
///   - includeSize: Whether to format as boxes (with size) or points
/// - Returns: Formatted string suitable for parsing
internal func formatCoordinates(_ coordinates: [GeneratedCoordinate], includeSize: Bool) -> String {
    if coordinates.isEmpty {
        return ""
    }

    if includeSize {
        return coordinates.map { $0.boxString }.joined(separator: ", ")
    } else {
        return coordinates.map { $0.pointString }.joined(separator: ", ")
    }
}
