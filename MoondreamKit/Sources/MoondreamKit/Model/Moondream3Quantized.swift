import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

/// Moondream3 variant with fully quantized vision encoder, text attention, and region model
/// Use this for lewi/md3p-int4-smol model to fit in iOS memory constraints
public class Moondream3Quantized: Module, MoondreamModel {
    @ModuleInfo(key: "vision") private var visionEncoder: QuantizedVisionEncoder
    @ModuleInfo(key: "text") private var textModel: QuantizedTextModel
    @ModuleInfo(key: "region") private var regionModel: QuantizedRegionModel

    public let config: Moondream3Configuration

    // Attention mask (precomputed once)
    private var attnMask: MLXArray?

    public var vocabularySize: Int { config.text.vocabSize }
    public var kvHeads: [Int] { textModel.kvHeads }

    public init(_ config: Moondream3Configuration) {
        self.config = config
        self._visionEncoder.wrappedValue = QuantizedVisionEncoder(config.vision)
        self._textModel.wrappedValue = QuantizedTextModel(config.text)
        self._regionModel.wrappedValue = QuantizedRegionModel(config.region)

        let maxCtx = config.text.maxContext
        let mask = tril(MLXArray.ones([1, 1, maxCtx, maxCtx]).asType(.bool))
        self.attnMask = mask
    }

    // MARK: - Vision Encoding

    public func encodeImage(_ pixels: MLXArray) -> MLXArray {
        visionEncoder(pixels)
    }

    // MARK: - Internal Forward Pass

    private func prefill(
        embeddings: MLXArray,
        cachePos: Int,
        cache: inout [(MLXArray, MLXArray)]
    ) -> MLXArray {
        let seqLen = embeddings.dim(1)
        let positions = MLXArray((cachePos..<(cachePos + seqLen)).map { Int32($0) })
        let mask = attnMask?[0..., 0..., cachePos..<(cachePos + seqLen), 0...]

        let (hidden, newCaches) = textModel(
            embeddings,
            positions: positions,
            mask: mask,
            cache: cache,
            cachePos: cachePos
        )

        cache = newCaches
        return hidden
    }

    private func decodeOne(
        embedding: MLXArray,
        cachePos: Int,
        cache: inout [(MLXArray, MLXArray)]
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let positions = MLXArray([Int32(cachePos)])

        let (hidden, newCaches) = textModel(
            embedding,
            positions: positions,
            mask: nil,
            cache: cache,
            cachePos: cachePos
        )

        cache = newCaches
        return (textModel.generateLogits(hidden), hidden)
    }

    // MARK: - Generation Methods

    public func caption(
        pixels: MLXArray,
        length: String = "normal",
        tokenizer: any Tokenizer,
        maxTokens: Int = 768,
        temperature: Float = 0.0
    ) -> String {
        md3Log("caption() [quantized] started - length: \(length)")
        eval(pixels)

        let promptTokens = TokenConstants.Caption.template(for: length)

        let imgEmb = visionEncoder(pixels)
        let bosTokens = MLXArray([Int32(TokenConstants.bos)]).expandedDimensions(axis: 0)
        let bosEmb = textModel.embed(bosTokens)
        let inputsEmbeds = concatenated([bosEmb, imgEmb], axis: 1)

        var cache = KVCacheManager.allocateCache(
            layers: config.text.nLayers,
            kvHeads: config.text.nKvHeads,
            headDim: config.text.dim / config.text.nHeads
        )

        _ = prefill(embeddings: inputsEmbeds, cachePos: 0, cache: &cache)
        var pos = inputsEmbeds.dim(1)

        let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let promptEmb = textModel.embed(promptArray)
        let hidden = prefill(embeddings: promptEmb, cachePos: pos, cache: &cache)
        var logits = textModel.generateLogits(hidden)
        pos += promptEmb.dim(1)

        var nextToken = sampleToken(logits: logits, temperature: temperature)
        var tokens: [Int] = []
        var generated = 0

        while nextToken != TokenConstants.eos && generated < maxTokens {
            tokens.append(nextToken)
            let tokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            let nextEmb = textModel.embed(tokenArray)
            logits = decodeOne(embedding: nextEmb, cachePos: pos, cache: &cache).logits
            nextToken = sampleToken(logits: logits, temperature: temperature)
            pos += 1
            generated += 1
        }

        return tokenizer.decode(tokens: tokens)
    }

    public func query(
        pixels: MLXArray,
        question: String,
        tokenizer: any Tokenizer,
        maxTokens: Int = 768,
        temperature: Float = 0.0
    ) -> String {
        md3Log("query() [quantized] started - question: \(question.prefix(50))...")

        let promptTokens = TokenConstants.Query.prefix + tokenizer.encode(text: question) + TokenConstants.Query.suffix

        let imgEmb = visionEncoder(pixels)
        let bosTokens = MLXArray([Int32(TokenConstants.bos)]).expandedDimensions(axis: 0)
        let bosEmb = textModel.embed(bosTokens)
        let inputsEmbeds = concatenated([bosEmb, imgEmb], axis: 1)

        var cache = KVCacheManager.allocateCache(
            layers: config.text.nLayers,
            kvHeads: config.text.nKvHeads,
            headDim: config.text.dim / config.text.nHeads
        )

        _ = prefill(embeddings: inputsEmbeds, cachePos: 0, cache: &cache)
        var pos = inputsEmbeds.dim(1)

        let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let promptEmb = textModel.embed(promptArray)
        let hidden = prefill(embeddings: promptEmb, cachePos: pos, cache: &cache)
        var logits = textModel.generateLogits(hidden)
        pos += promptEmb.dim(1)

        var nextToken = sampleToken(logits: logits, temperature: temperature)
        var tokens: [Int] = []
        var generated = 0

        while nextToken != TokenConstants.eos && generated < maxTokens {
            tokens.append(nextToken)
            let tokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            let nextEmb = textModel.embed(tokenArray)
            logits = decodeOne(embedding: nextEmb, cachePos: pos, cache: &cache).logits
            nextToken = sampleToken(logits: logits, temperature: temperature)
            pos += 1
            generated += 1
        }

        return tokenizer.decode(tokens: tokens)
    }

    public func point(
        pixels: MLXArray,
        object: String,
        tokenizer: any Tokenizer,
        maxTokens: Int = 256,
        temperature: Float = 0.0
    ) -> String {
        let promptTokens = TokenConstants.Point.prefix + tokenizer.encode(text: " " + object) + TokenConstants.Point.suffix

        let imgEmb = visionEncoder(pixels)
        let bosTokens = MLXArray([Int32(TokenConstants.bos)]).expandedDimensions(axis: 0)
        let bosEmb = textModel.embed(bosTokens)
        let inputsEmbeds = concatenated([bosEmb, imgEmb], axis: 1)

        var cache = KVCacheManager.allocateCache(
            layers: config.text.nLayers,
            kvHeads: config.text.nKvHeads,
            headDim: config.text.dim / config.text.nHeads
        )

        _ = prefill(embeddings: inputsEmbeds, cachePos: 0, cache: &cache)
        var pos = inputsEmbeds.dim(1)

        let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let promptEmb = textModel.embed(promptArray)
        let hidden = prefill(embeddings: promptEmb, cachePos: pos, cache: &cache)
        let logits = textModel.generateLogits(hidden)
        pos += promptEmb.dim(1)

        let firstToken = sampleToken(logits: logits, temperature: 0)

        if firstToken != TokenConstants.coord {
            return ""
        }

        let lastHidden = hidden[0..., (-1)..., 0...].squeezed(axis: 1)

        let coordinates = generateCoordinates(
            hidden: lastHidden.expandedDimensions(axis: 0).expandedDimensions(axis: 1),
            firstToken: TokenConstants.coord,
            startPos: pos,
            cache: &cache,
            regionModel: regionModel,
            decodeOne: { [self] emb, cachePos, cache in
                self.decodeOne(embedding: emb, cachePos: cachePos, cache: &cache)
            },
            includeSize: false,
            maxObjects: 50
        )

        return formatCoordinates(coordinates, includeSize: false)
    }

    public func detect(
        pixels: MLXArray,
        object: String,
        tokenizer: any Tokenizer,
        maxTokens: Int = 256,
        temperature: Float = 0.0
    ) -> String {
        let promptTokens = TokenConstants.Detect.prefix + tokenizer.encode(text: " " + object) + TokenConstants.Detect.suffix

        let imgEmb = visionEncoder(pixels)
        let bosTokens = MLXArray([Int32(TokenConstants.bos)]).expandedDimensions(axis: 0)
        let bosEmb = textModel.embed(bosTokens)
        let inputsEmbeds = concatenated([bosEmb, imgEmb], axis: 1)

        var cache = KVCacheManager.allocateCache(
            layers: config.text.nLayers,
            kvHeads: config.text.nKvHeads,
            headDim: config.text.dim / config.text.nHeads
        )

        _ = prefill(embeddings: inputsEmbeds, cachePos: 0, cache: &cache)
        var pos = inputsEmbeds.dim(1)

        let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let promptEmb = textModel.embed(promptArray)
        let hidden = prefill(embeddings: promptEmb, cachePos: pos, cache: &cache)
        let logits = textModel.generateLogits(hidden)
        pos += promptEmb.dim(1)

        let firstToken = sampleToken(logits: logits, temperature: 0)

        if firstToken != TokenConstants.coord {
            return ""
        }

        let lastHidden = hidden[0..., (-1)..., 0...].squeezed(axis: 1)

        let coordinates = generateCoordinates(
            hidden: lastHidden.expandedDimensions(axis: 0).expandedDimensions(axis: 1),
            firstToken: TokenConstants.coord,
            startPos: pos,
            cache: &cache,
            regionModel: regionModel,
            decodeOne: { [self] emb, cachePos, cache in
                self.decodeOne(embedding: emb, cachePos: cachePos, cache: &cache)
            },
            includeSize: true,
            maxObjects: 150
        )

        return formatCoordinates(coordinates, includeSize: true)
    }

    // MARK: - LanguageModel Conformance

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let seqLen = inputIds.dim(1)
        let positions = MLXArray((0..<seqLen).map { Int32($0) })
        let maxCtx = config.text.maxContext
        let mask = tril(MLXArray.ones([1, 1, maxCtx, maxCtx]).asType(.bool))

        if let image = input.image {
            let imageFeatures = visionEncoder(image.pixels)
            let textEmbeddings = textModel.embed(inputIds)
            let bosTokens = MLXArray([Int32(TokenConstants.bos)]).expandedDimensions(axis: 0)
            let bosEmb = textModel.embed(bosTokens)
            let combined = concatenated([bosEmb, imageFeatures, textEmbeddings], axis: 1)
            let combinedLen = combined.dim(1)
            let combinedPositions = MLXArray((0..<combinedLen).map { Int32($0) })
            let (hidden, _) = textModel(combined, positions: combinedPositions, mask: mask, cache: nil, cachePos: 0)
            let logits = textModel.generateLogits(hidden)
            return .logits(LMOutput(logits: logits))
        } else {
            let embeddings = textModel.embed(inputIds)
            let (hidden, _) = textModel(embeddings, positions: positions, mask: mask, cache: nil, cachePos: 0)
            let logits = textModel.generateLogits(hidden)
            return .logits(LMOutput(logits: logits))
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let seqLen = inputs.dim(1)
        let positions = MLXArray((0..<seqLen).map { Int32($0) })
        let embeddings = textModel.embed(inputs)
        let (hidden, _) = textModel(embeddings, positions: positions, mask: nil, cache: nil, cachePos: 0)
        return textModel.generateLogits(hidden)
    }

    // MARK: - Weight Sanitization

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            var newKey = key

            if key.contains("position_id") || key.contains("rotary_emb") {
                continue
            }

            if newKey.contains(".mlp.fc1_bias") || newKey.contains(".mlp.fc2_bias") {
                let components = newKey.split(separator: ".")
                if let blocksIndex = components.firstIndex(of: "blocks"),
                   blocksIndex + 1 < components.count,
                   let layerNum = Int(components[blocksIndex + 1]),
                   layerNum < 4 {
                    newKey = newKey.replacingOccurrences(of: ".fc1_bias", with: ".fc1.bias")
                    newKey = newKey.replacingOccurrences(of: ".fc2_bias", with: ".fc2.bias")
                }
            }

            sanitized[newKey] = value
        }

        return sanitized
    }

    // MARK: - KV Cache

    public func allocateKVCache(batchSize: Int = 1, maxSeqLen: Int = 1024) -> [[(MLXArray, MLXArray)]] {
        KVCacheManager.allocateNestedCache(
            layers: config.text.nLayers,
            kvHeads: config.text.nKvHeads,
            headDim: config.text.dim / config.text.nHeads,
            maxSeqLen: maxSeqLen,
            batchSize: batchSize
        )
    }
}
