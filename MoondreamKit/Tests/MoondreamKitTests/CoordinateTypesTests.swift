// Copyright 2025 Moondream AI. All rights reserved.
// SPDX-License-Identifier: MIT

import XCTest
import CoreGraphics
@testable import MoondreamKit

/// Tests for coordinate types: NormalizedPoint, NormalizedBox
final class CoordinateTypesTests: XCTestCase {

    // MARK: - NormalizedPoint Tests

    func testNormalizedPointInit() {
        let point = NormalizedPoint(x: 0.5, y: 0.75)
        XCTAssertEqual(point.x, 0.5)
        XCTAssertEqual(point.y, 0.75)
        XCTAssertNil(point.label)
    }

    func testNormalizedPointWithLabel() {
        let point = NormalizedPoint(x: 0.3, y: 0.4, label: "face")
        XCTAssertEqual(point.x, 0.3)
        XCTAssertEqual(point.y, 0.4)
        XCTAssertEqual(point.label, "face")
    }

    func testNormalizedPointWithCustomId() {
        let customId = UUID()
        let point = NormalizedPoint(id: customId, x: 0.1, y: 0.2)
        XCTAssertEqual(point.id, customId)
    }

    func testNormalizedPointIdentifiable() {
        let point1 = NormalizedPoint(x: 0.5, y: 0.5)
        let point2 = NormalizedPoint(x: 0.5, y: 0.5)
        // Each point should have a unique ID
        XCTAssertNotEqual(point1.id, point2.id)
    }

    func testNormalizedPointPositionInSize() {
        let point = NormalizedPoint(x: 0.5, y: 0.25)
        let size = CGSize(width: 100, height: 200)
        let position = point.position(in: size)

        XCTAssertEqual(position.x, 50.0, accuracy: 0.001)
        XCTAssertEqual(position.y, 50.0, accuracy: 0.001)
    }

    func testNormalizedPointPositionAtOrigin() {
        let point = NormalizedPoint(x: 0.0, y: 0.0)
        let size = CGSize(width: 640, height: 480)
        let position = point.position(in: size)

        XCTAssertEqual(position.x, 0.0, accuracy: 0.001)
        XCTAssertEqual(position.y, 0.0, accuracy: 0.001)
    }

    func testNormalizedPointPositionAtMax() {
        let point = NormalizedPoint(x: 1.0, y: 1.0)
        let size = CGSize(width: 1920, height: 1080)
        let position = point.position(in: size)

        XCTAssertEqual(position.x, 1920.0, accuracy: 0.001)
        XCTAssertEqual(position.y, 1080.0, accuracy: 0.001)
    }

    func testNormalizedPointPositionWithZeroSize() {
        let point = NormalizedPoint(x: 0.5, y: 0.5)
        let size = CGSize(width: 0, height: 0)
        let position = point.position(in: size)

        XCTAssertEqual(position.x, 0.0, accuracy: 0.001)
        XCTAssertEqual(position.y, 0.0, accuracy: 0.001)
    }

    func testNormalizedPointEquatable() {
        let id = UUID()
        let point1 = NormalizedPoint(id: id, x: 0.5, y: 0.5, label: "test")
        let point2 = NormalizedPoint(id: id, x: 0.5, y: 0.5, label: "test")
        let point3 = NormalizedPoint(x: 0.5, y: 0.5)  // Different ID

        XCTAssertEqual(point1, point2)
        XCTAssertNotEqual(point1, point3)
    }

    func testNormalizedPointCodable() throws {
        let original = NormalizedPoint(x: 0.123, y: 0.456, label: "test point")
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(NormalizedPoint.self, from: encoded)

        XCTAssertEqual(decoded.x, original.x, accuracy: 0.0001)
        XCTAssertEqual(decoded.y, original.y, accuracy: 0.0001)
        XCTAssertEqual(decoded.label, original.label)
    }

    func testNormalizedPointCodableRoundTrip() throws {
        // Test with label
        let withLabel = NormalizedPoint(x: 0.1, y: 0.9, label: "corner")
        let encodedWithLabel = try JSONEncoder().encode(withLabel)
        let decodedWithLabel = try JSONDecoder().decode(NormalizedPoint.self, from: encodedWithLabel)
        XCTAssertEqual(decodedWithLabel.label, "corner")

        // Test without label
        let withoutLabel = NormalizedPoint(x: 0.5, y: 0.5)
        let encodedWithoutLabel = try JSONEncoder().encode(withoutLabel)
        let decodedWithoutLabel = try JSONDecoder().decode(NormalizedPoint.self, from: encodedWithoutLabel)
        XCTAssertNil(decodedWithoutLabel.label)
    }

    // MARK: - NormalizedBox Tests

    func testNormalizedBoxInit() {
        let box = NormalizedBox(xMin: 0.1, yMin: 0.2, xMax: 0.8, yMax: 0.9)
        XCTAssertEqual(box.xMin, 0.1)
        XCTAssertEqual(box.yMin, 0.2)
        XCTAssertEqual(box.xMax, 0.8)
        XCTAssertEqual(box.yMax, 0.9)
        XCTAssertNil(box.label)
        XCTAssertNil(box.confidence)
    }

    func testNormalizedBoxWithLabelAndConfidence() {
        let box = NormalizedBox(
            xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5,
            label: "person",
            confidence: 0.95
        )
        XCTAssertEqual(box.label, "person")
        XCTAssertEqual(Double(box.confidence!), 0.95, accuracy: 0.001)
    }

    func testNormalizedBoxWithCustomId() {
        let customId = UUID()
        let box = NormalizedBox(id: customId, xMin: 0, yMin: 0, xMax: 1, yMax: 1)
        XCTAssertEqual(box.id, customId)
    }

    func testNormalizedBoxIdentifiable() {
        let box1 = NormalizedBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5)
        let box2 = NormalizedBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5)
        XCTAssertNotEqual(box1.id, box2.id)
    }

    func testNormalizedBoxWidth() {
        let box = NormalizedBox(xMin: 0.2, yMin: 0.0, xMax: 0.7, yMax: 1.0)
        XCTAssertEqual(box.width, 0.5, accuracy: 0.001)
    }

    func testNormalizedBoxHeight() {
        let box = NormalizedBox(xMin: 0.0, yMin: 0.1, xMax: 1.0, yMax: 0.6)
        XCTAssertEqual(box.height, 0.5, accuracy: 0.001)
    }

    func testNormalizedBoxCenterX() {
        let box = NormalizedBox(xMin: 0.2, yMin: 0.0, xMax: 0.8, yMax: 1.0)
        XCTAssertEqual(box.centerX, 0.5, accuracy: 0.001)
    }

    func testNormalizedBoxCenterY() {
        let box = NormalizedBox(xMin: 0.0, yMin: 0.3, xMax: 1.0, yMax: 0.7)
        XCTAssertEqual(box.centerY, 0.5, accuracy: 0.001)
    }

    func testNormalizedBoxFrameInSize() {
        let box = NormalizedBox(xMin: 0.25, yMin: 0.25, xMax: 0.75, yMax: 0.75)
        let size = CGSize(width: 200, height: 100)
        let frame = box.frame(in: size)

        XCTAssertEqual(frame.origin.x, 50.0, accuracy: 0.001)
        XCTAssertEqual(frame.origin.y, 25.0, accuracy: 0.001)
        XCTAssertEqual(frame.size.width, 100.0, accuracy: 0.001)
        XCTAssertEqual(frame.size.height, 50.0, accuracy: 0.001)
    }

    func testNormalizedBoxFrameFullSize() {
        let box = NormalizedBox(xMin: 0.0, yMin: 0.0, xMax: 1.0, yMax: 1.0)
        let size = CGSize(width: 640, height: 480)
        let frame = box.frame(in: size)

        XCTAssertEqual(frame.origin.x, 0.0, accuracy: 0.001)
        XCTAssertEqual(frame.origin.y, 0.0, accuracy: 0.001)
        XCTAssertEqual(frame.size.width, 640.0, accuracy: 0.001)
        XCTAssertEqual(frame.size.height, 480.0, accuracy: 0.001)
    }

    func testNormalizedBoxCenterInSize() {
        let box = NormalizedBox(xMin: 0.2, yMin: 0.3, xMax: 0.6, yMax: 0.7)
        let size = CGSize(width: 100, height: 200)
        let center = box.center(in: size)

        XCTAssertEqual(center.x, 40.0, accuracy: 0.001)  // (0.2 + 0.6) / 2 * 100 = 40
        XCTAssertEqual(center.y, 100.0, accuracy: 0.001)  // (0.3 + 0.7) / 2 * 200 = 100
    }

    func testNormalizedBoxEquatable() {
        let id = UUID()
        let box1 = NormalizedBox(id: id, xMin: 0.1, yMin: 0.2, xMax: 0.8, yMax: 0.9, label: "test")
        let box2 = NormalizedBox(id: id, xMin: 0.1, yMin: 0.2, xMax: 0.8, yMax: 0.9, label: "test")
        let box3 = NormalizedBox(xMin: 0.1, yMin: 0.2, xMax: 0.8, yMax: 0.9)

        XCTAssertEqual(box1, box2)
        XCTAssertNotEqual(box1, box3)
    }

    func testNormalizedBoxCodable() throws {
        let original = NormalizedBox(
            xMin: 0.1, yMin: 0.2, xMax: 0.8, yMax: 0.9,
            label: "object",
            confidence: 0.87
        )
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(NormalizedBox.self, from: encoded)

        XCTAssertEqual(decoded.xMin, original.xMin, accuracy: 0.0001)
        XCTAssertEqual(decoded.yMin, original.yMin, accuracy: 0.0001)
        XCTAssertEqual(decoded.xMax, original.xMax, accuracy: 0.0001)
        XCTAssertEqual(decoded.yMax, original.yMax, accuracy: 0.0001)
        XCTAssertEqual(decoded.label, original.label)
        XCTAssertEqual(Double(decoded.confidence!), Double(original.confidence!), accuracy: 0.0001)
    }

    func testNormalizedBoxCodableWithoutOptionals() throws {
        let original = NormalizedBox(xMin: 0.0, yMin: 0.0, xMax: 0.5, yMax: 0.5)
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(NormalizedBox.self, from: encoded)

        XCTAssertNil(decoded.label)
        XCTAssertNil(decoded.confidence)
    }

    // MARK: - Edge Cases

    func testNormalizedPointOutOfBounds() {
        // Values outside 0-1 range should still work (no clamping)
        let point = NormalizedPoint(x: -0.1, y: 1.5)
        let size = CGSize(width: 100, height: 100)
        let position = point.position(in: size)

        XCTAssertEqual(position.x, -10.0, accuracy: 0.001)
        XCTAssertEqual(position.y, 150.0, accuracy: 0.001)
    }

    func testNormalizedBoxInvertedCoordinates() {
        // Min > Max should still compute (negative dimensions)
        let box = NormalizedBox(xMin: 0.8, yMin: 0.9, xMax: 0.2, yMax: 0.1)
        XCTAssertEqual(box.width, -0.6, accuracy: 0.001)
        XCTAssertEqual(box.height, -0.8, accuracy: 0.001)
    }

    func testNormalizedBoxZeroSize() {
        let box = NormalizedBox(xMin: 0.5, yMin: 0.5, xMax: 0.5, yMax: 0.5)
        XCTAssertEqual(box.width, 0.0, accuracy: 0.001)
        XCTAssertEqual(box.height, 0.0, accuracy: 0.001)
        XCTAssertEqual(box.centerX, 0.5, accuracy: 0.001)
        XCTAssertEqual(box.centerY, 0.5, accuracy: 0.001)
    }

    func testNormalizedBoxFrameInZeroSize() {
        let box = NormalizedBox(xMin: 0.1, yMin: 0.2, xMax: 0.8, yMax: 0.9)
        let size = CGSize(width: 0, height: 0)
        let frame = box.frame(in: size)

        XCTAssertEqual(frame.origin.x, 0.0, accuracy: 0.001)
        XCTAssertEqual(frame.origin.y, 0.0, accuracy: 0.001)
        XCTAssertEqual(frame.size.width, 0.0, accuracy: 0.001)
        XCTAssertEqual(frame.size.height, 0.0, accuracy: 0.001)
    }

    // MARK: - Practical Scenarios

    func testPointConversionForHDImage() {
        // Typical HD image scenario
        let point = NormalizedPoint(x: 0.333, y: 0.666, label: "face")
        let hdSize = CGSize(width: 1920, height: 1080)
        let position = point.position(in: hdSize)

        XCTAssertEqual(position.x, 639.36, accuracy: 0.01)  // 0.333 * 1920
        XCTAssertEqual(position.y, 719.28, accuracy: 0.01)  // 0.666 * 1080
    }

    func testBoxConversionForSquareImage() {
        // Square image scenario
        let box = NormalizedBox(xMin: 0.1, yMin: 0.1, xMax: 0.9, yMax: 0.9, label: "object")
        let squareSize = CGSize(width: 512, height: 512)
        let frame = box.frame(in: squareSize)

        XCTAssertEqual(frame.origin.x, 51.2, accuracy: 0.01)
        XCTAssertEqual(frame.origin.y, 51.2, accuracy: 0.01)
        XCTAssertEqual(frame.size.width, 409.6, accuracy: 0.01)
        XCTAssertEqual(frame.size.height, 409.6, accuracy: 0.01)
    }

    func testMultipleBoxesNonOverlapping() {
        let box1 = NormalizedBox(xMin: 0.0, yMin: 0.0, xMax: 0.4, yMax: 0.4)
        let box2 = NormalizedBox(xMin: 0.6, yMin: 0.6, xMax: 1.0, yMax: 1.0)

        // Verify they don't overlap by checking centers are far apart
        XCTAssertLessThan(box1.centerX, 0.5)
        XCTAssertGreaterThan(box2.centerX, 0.5)
        XCTAssertLessThan(box1.centerY, 0.5)
        XCTAssertGreaterThan(box2.centerY, 0.5)
    }
}
