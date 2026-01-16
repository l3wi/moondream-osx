import XCTest
@testable import MoondreamKit

/// Tests for model types: Skill, CaptionLength, MoondreamResult, ModelInfo
final class ModelTypesTests: XCTestCase {

    // MARK: - Skill Tests

    func testSkillAllCases() {
        let allCases = Skill.allCases
        XCTAssertEqual(allCases.count, 4)
        XCTAssertTrue(allCases.contains(.caption))
        XCTAssertTrue(allCases.contains(.query))
        XCTAssertTrue(allCases.contains(.point))
        XCTAssertTrue(allCases.contains(.detect))
    }

    func testSkillRawValues() {
        XCTAssertEqual(Skill.caption.rawValue, "caption")
        XCTAssertEqual(Skill.query.rawValue, "query")
        XCTAssertEqual(Skill.point.rawValue, "point")
        XCTAssertEqual(Skill.detect.rawValue, "detect")
    }

    func testSkillIdentifiable() {
        XCTAssertEqual(Skill.caption.id, "caption")
        XCTAssertEqual(Skill.query.id, "query")
        XCTAssertEqual(Skill.point.id, "point")
        XCTAssertEqual(Skill.detect.id, "detect")
    }

    func testSkillDisplayNames() {
        XCTAssertEqual(Skill.caption.displayName, "Caption")
        XCTAssertEqual(Skill.query.displayName, "Query")
        XCTAssertEqual(Skill.point.displayName, "Point")
        XCTAssertEqual(Skill.detect.displayName, "Detect")
    }

    func testSkillIcons() {
        XCTAssertEqual(Skill.caption.icon, "text.alignleft")
        XCTAssertEqual(Skill.query.icon, "bubble.left.and.bubble.right")
        XCTAssertEqual(Skill.point.icon, "scope")
        XCTAssertEqual(Skill.detect.icon, "square.dashed")
    }

    func testSkillDescriptions() {
        XCTAssertFalse(Skill.caption.description.isEmpty)
        XCTAssertFalse(Skill.query.description.isEmpty)
        XCTAssertFalse(Skill.point.description.isEmpty)
        XCTAssertFalse(Skill.detect.description.isEmpty)

        XCTAssertTrue(Skill.caption.description.contains("description"))
        XCTAssertTrue(Skill.query.description.contains("question"))
        XCTAssertTrue(Skill.point.description.contains("locate"))
        XCTAssertTrue(Skill.detect.description.contains("object"))
    }

    func testSkillRequiresObjectInput() {
        XCTAssertFalse(Skill.caption.requiresObjectInput)
        XCTAssertFalse(Skill.query.requiresObjectInput)
        XCTAssertTrue(Skill.point.requiresObjectInput)
        XCTAssertTrue(Skill.detect.requiresObjectInput)
    }

    func testSkillRequiresInput() {
        XCTAssertFalse(Skill.caption.requiresInput)
        XCTAssertTrue(Skill.query.requiresInput)
        XCTAssertTrue(Skill.point.requiresInput)
        XCTAssertTrue(Skill.detect.requiresInput)
    }

    func testSkillCodable() throws {
        for skill in Skill.allCases {
            let encoded = try JSONEncoder().encode(skill)
            let decoded = try JSONDecoder().decode(Skill.self, from: encoded)
            XCTAssertEqual(decoded, skill)
        }
    }

    // MARK: - CaptionLength Tests

    func testCaptionLengthAllCases() {
        let allCases = CaptionLength.allCases
        XCTAssertEqual(allCases.count, 3)
        XCTAssertTrue(allCases.contains(.short))
        XCTAssertTrue(allCases.contains(.normal))
        XCTAssertTrue(allCases.contains(.long))
    }

    func testCaptionLengthRawValues() {
        XCTAssertEqual(CaptionLength.short.rawValue, "short")
        XCTAssertEqual(CaptionLength.normal.rawValue, "normal")
        XCTAssertEqual(CaptionLength.long.rawValue, "long")
    }

    func testCaptionLengthIdentifiable() {
        XCTAssertEqual(CaptionLength.short.id, "short")
        XCTAssertEqual(CaptionLength.normal.id, "normal")
        XCTAssertEqual(CaptionLength.long.id, "long")
    }

    func testCaptionLengthDisplayNames() {
        XCTAssertEqual(CaptionLength.short.displayName, "Short")
        XCTAssertEqual(CaptionLength.normal.displayName, "Normal")
        XCTAssertEqual(CaptionLength.long.displayName, "Long")
    }

    func testCaptionLengthCodable() throws {
        for length in CaptionLength.allCases {
            let encoded = try JSONEncoder().encode(length)
            let decoded = try JSONDecoder().decode(CaptionLength.self, from: encoded)
            XCTAssertEqual(decoded, length)
        }
    }

    // MARK: - QueryResult Tests

    func testQueryResultInit() {
        let result = QueryResult(answer: "The Mona Lisa", reasoning: "Based on the style", rawOutput: "raw")
        XCTAssertEqual(result.answer, "The Mona Lisa")
        XCTAssertEqual(result.reasoning, "Based on the style")
        XCTAssertEqual(result.rawOutput, "raw")
    }

    func testQueryResultDefaultValues() {
        let result = QueryResult(answer: "Answer")
        XCTAssertEqual(result.answer, "Answer")
        XCTAssertNil(result.reasoning)
        XCTAssertEqual(result.rawOutput, "")
    }

    func testQueryResultEquatable() {
        let result1 = QueryResult(answer: "A", reasoning: "B", rawOutput: "C")
        let result2 = QueryResult(answer: "A", reasoning: "B", rawOutput: "C")
        let result3 = QueryResult(answer: "Different")
        XCTAssertEqual(result1, result2)
        XCTAssertNotEqual(result1, result3)
    }

    // MARK: - CaptionResult Tests

    func testCaptionResultInit() {
        let result = CaptionResult(caption: "A beautiful painting", rawOutput: "raw")
        XCTAssertEqual(result.caption, "A beautiful painting")
        XCTAssertEqual(result.rawOutput, "raw")
    }

    func testCaptionResultDefaultValues() {
        let result = CaptionResult(caption: "Caption")
        XCTAssertEqual(result.caption, "Caption")
        XCTAssertEqual(result.rawOutput, "")
    }

    func testCaptionResultEquatable() {
        let result1 = CaptionResult(caption: "A", rawOutput: "B")
        let result2 = CaptionResult(caption: "A", rawOutput: "B")
        let result3 = CaptionResult(caption: "Different")
        XCTAssertEqual(result1, result2)
        XCTAssertNotEqual(result1, result3)
    }

    // MARK: - PointResult Tests

    func testPointResultInit() {
        let points = [NormalizedPoint(x: 0.5, y: 0.5)]
        let result = PointResult(points: points, rawOutput: "raw")
        XCTAssertEqual(result.points.count, 1)
        XCTAssertEqual(result.rawOutput, "raw")
    }

    func testPointResultDefaultValues() {
        let result = PointResult(points: [])
        XCTAssertEqual(result.points.count, 0)
        XCTAssertEqual(result.rawOutput, "")
    }

    func testPointResultEquatable() {
        let point = NormalizedPoint(x: 0.5, y: 0.5)
        let result1 = PointResult(points: [point])
        let result2 = PointResult(points: [point])
        XCTAssertEqual(result1, result2)
    }

    // MARK: - DetectResult Tests

    func testDetectResultInit() {
        let boxes = [NormalizedBox(xMin: 0.1, yMin: 0.1, xMax: 0.9, yMax: 0.9)]
        let result = DetectResult(boxes: boxes, rawOutput: "raw")
        XCTAssertEqual(result.boxes.count, 1)
        XCTAssertEqual(result.rawOutput, "raw")
    }

    func testDetectResultDefaultValues() {
        let result = DetectResult(boxes: [])
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertEqual(result.rawOutput, "")
    }

    func testDetectResultEquatable() {
        let box = NormalizedBox(xMin: 0.1, yMin: 0.1, xMax: 0.9, yMax: 0.9)
        let result1 = DetectResult(boxes: [box])
        let result2 = DetectResult(boxes: [box])
        XCTAssertEqual(result1, result2)
    }

    // MARK: - MoondreamResult Tests

    func testMoondreamResultQueryCase() {
        let queryResult = QueryResult(answer: "It's the Mona Lisa")
        let result = MoondreamResult.query(queryResult)

        XCTAssertEqual(result.displayText, "It's the Mona Lisa")
        XCTAssertNil(result.points)
        XCTAssertNil(result.boxes)
    }

    func testMoondreamResultCaptionCase() {
        let captionResult = CaptionResult(caption: "A famous painting")
        let result = MoondreamResult.caption(captionResult)

        XCTAssertEqual(result.displayText, "A famous painting")
        XCTAssertNil(result.points)
        XCTAssertNil(result.boxes)
    }

    func testMoondreamResultPointCase() {
        let points = [
            NormalizedPoint(x: 0.5, y: 0.5),
            NormalizedPoint(x: 0.3, y: 0.7)
        ]
        let pointResult = PointResult(points: points)
        let result = MoondreamResult.point(pointResult)

        XCTAssertEqual(result.displayText, "Found 2 point(s)")
        XCTAssertNotNil(result.points)
        XCTAssertEqual(result.points?.count, 2)
        XCTAssertNil(result.boxes)
    }

    func testMoondreamResultDetectCase() {
        let boxes = [
            NormalizedBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5),
            NormalizedBox(xMin: 0.5, yMin: 0.5, xMax: 0.9, yMax: 0.9),
            NormalizedBox(xMin: 0.2, yMin: 0.2, xMax: 0.4, yMax: 0.4)
        ]
        let detectResult = DetectResult(boxes: boxes)
        let result = MoondreamResult.detect(detectResult)

        XCTAssertEqual(result.displayText, "Detected 3 object(s)")
        XCTAssertNil(result.points)
        XCTAssertNotNil(result.boxes)
        XCTAssertEqual(result.boxes?.count, 3)
    }

    func testMoondreamResultEquatable() {
        let result1 = MoondreamResult.query(QueryResult(answer: "A"))
        let result2 = MoondreamResult.query(QueryResult(answer: "A"))
        let result3 = MoondreamResult.query(QueryResult(answer: "B"))
        let result4 = MoondreamResult.caption(CaptionResult(caption: "A"))

        XCTAssertEqual(result1, result2)
        XCTAssertNotEqual(result1, result3)
        XCTAssertNotEqual(result1, result4)
    }

    // MARK: - ModelInfo Tests

    func testModelInfoInit() {
        let info = ModelInfo(
            id: "test/model",
            displayName: "Test Model",
            description: "A test model",
            quantization: "int4",
            sizeBytes: 1_000_000_000
        )

        XCTAssertEqual(info.id, "test/model")
        XCTAssertEqual(info.displayName, "Test Model")
        XCTAssertEqual(info.description, "A test model")
        XCTAssertEqual(info.quantization, "int4")
        XCTAssertEqual(info.sizeBytes, 1_000_000_000)
    }

    func testModelInfoSizeDisplay() {
        let smallModel = ModelInfo(
            id: "test/small",
            displayName: "Small",
            description: "Small model",
            quantization: "int4",
            sizeBytes: 500_000_000  // 500 MB
        )
        XCTAssertFalse(smallModel.sizeDisplay.isEmpty)
        XCTAssertTrue(smallModel.sizeDisplay.contains("MB") || smallModel.sizeDisplay.contains("GB"))

        let largeModel = ModelInfo(
            id: "test/large",
            displayName: "Large",
            description: "Large model",
            quantization: "int4",
            sizeBytes: 6_000_000_000  // 6 GB
        )
        XCTAssertFalse(largeModel.sizeDisplay.isEmpty)
        XCTAssertTrue(largeModel.sizeDisplay.contains("GB"))
    }

    func testModelInfoIsCompatibleWithiOS() {
        // Under iOS limit
        let smallModel = ModelInfo(
            id: "test/small",
            displayName: "Small",
            description: "Small model",
            quantization: "int4",
            sizeBytes: 5_000_000_000  // 5 GB - under 5.5 GB limit
        )
        XCTAssertTrue(smallModel.isCompatibleWithiOS)

        // Over iOS limit
        let largeModel = ModelInfo(
            id: "test/large",
            displayName: "Large",
            description: "Large model",
            quantization: "bf16",
            sizeBytes: 7_000_000_000  // 7 GB - over 5.5 GB limit
        )
        XCTAssertFalse(largeModel.isCompatibleWithiOS)

        // At exact limit
        let limitModel = ModelInfo(
            id: "test/limit",
            displayName: "Limit",
            description: "At limit",
            quantization: "int4",
            sizeBytes: ModelInfo.iOSMaxSizeBytes
        )
        XCTAssertTrue(limitModel.isCompatibleWithiOS)
    }

    func testModelInfoRequiresHighMemoryMac() {
        // Under macOS 16GB limit
        let normalModel = ModelInfo(
            id: "test/normal",
            displayName: "Normal",
            description: "Normal model",
            quantization: "int4",
            sizeBytes: 6_000_000_000
        )
        XCTAssertFalse(normalModel.requiresHighMemoryMac)

        // Over macOS 16GB limit
        let hugeModel = ModelInfo(
            id: "test/huge",
            displayName: "Huge",
            description: "Huge model",
            quantization: "bf16",
            sizeBytes: 15_000_000_000
        )
        XCTAssertTrue(hugeModel.requiresHighMemoryMac)
    }

    func testModelInfoEquatable() {
        let info1 = ModelInfo(
            id: "test/model",
            displayName: "Test",
            description: "Desc",
            quantization: "int4",
            sizeBytes: 1000
        )
        let info2 = ModelInfo(
            id: "test/model",
            displayName: "Test",
            description: "Desc",
            quantization: "int4",
            sizeBytes: 1000
        )
        let info3 = ModelInfo(
            id: "test/other",
            displayName: "Test",
            description: "Desc",
            quantization: "int4",
            sizeBytes: 1000
        )

        XCTAssertEqual(info1, info2)
        XCTAssertNotEqual(info1, info3)
    }

    func testModelInfoHashable() {
        let info1 = ModelInfo(
            id: "test/model",
            displayName: "Test",
            description: "Desc",
            quantization: "int4",
            sizeBytes: 1000
        )
        let info2 = ModelInfo(
            id: "test/model",
            displayName: "Test",
            description: "Desc",
            quantization: "int4",
            sizeBytes: 1000
        )

        var set = Set<ModelInfo>()
        set.insert(info1)
        set.insert(info2)
        XCTAssertEqual(set.count, 1)
    }

    // MARK: - AvailableModels Tests

    func testAvailableModelsAll() {
        let models = AvailableModels.all
        XCTAssertGreaterThanOrEqual(models.count, 2)

        // Check that we have the expected models
        let modelIds = models.map { $0.id }
        XCTAssertTrue(modelIds.contains("moondream/md3p-int4"))
        XCTAssertTrue(modelIds.contains("lewi/md3p-int4-smol"))
    }

    func testAvailableModelsDefaultId() {
        let defaultId = AvailableModels.defaultId
        XCTAssertFalse(defaultId.isEmpty)

        // Default should be one of the available models
        let modelIds = AvailableModels.all.map { $0.id }
        XCTAssertTrue(modelIds.contains(defaultId))
    }

    func testAvailableModelsModelForId() {
        // Valid ID
        let standardModel = AvailableModels.model(for: "moondream/md3p-int4")
        XCTAssertNotNil(standardModel)
        XCTAssertEqual(standardModel?.id, "moondream/md3p-int4")

        let compactModel = AvailableModels.model(for: "lewi/md3p-int4-smol")
        XCTAssertNotNil(compactModel)
        XCTAssertEqual(compactModel?.id, "lewi/md3p-int4-smol")

        // Invalid ID
        let invalidModel = AvailableModels.model(for: "invalid/model")
        XCTAssertNil(invalidModel)
    }

    func testAvailableModelsDefaultModel() {
        let defaultModel = AvailableModels.defaultModel
        XCTAssertFalse(defaultModel.id.isEmpty)
        XCTAssertFalse(defaultModel.displayName.isEmpty)
        XCTAssertGreaterThan(defaultModel.sizeBytes, 0)
    }

    func testAvailableModelsProperties() {
        for model in AvailableModels.all {
            // All models should have valid properties
            XCTAssertFalse(model.id.isEmpty, "Model ID should not be empty")
            XCTAssertFalse(model.displayName.isEmpty, "Display name should not be empty")
            XCTAssertFalse(model.description.isEmpty, "Description should not be empty")
            XCTAssertFalse(model.quantization.isEmpty, "Quantization should not be empty")
            XCTAssertGreaterThan(model.sizeBytes, 0, "Size should be positive")
            XCTAssertFalse(model.sizeDisplay.isEmpty, "Size display should not be empty")
        }
    }
}
