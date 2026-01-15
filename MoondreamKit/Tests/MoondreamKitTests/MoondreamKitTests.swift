import XCTest
@testable import MoondreamKit

final class MoondreamKitTests: XCTestCase {
    func testSkillEnumeration() throws {
        XCTAssertEqual(Skill.allCases.count, 4)
        XCTAssertEqual(Skill.caption.displayName, "Caption")
        XCTAssertEqual(Skill.query.requiresObjectInput, false)
        XCTAssertEqual(Skill.point.requiresObjectInput, true)
    }

    func testCaptionLength() throws {
        XCTAssertEqual(CaptionLength.allCases.count, 3)
        XCTAssertEqual(CaptionLength.normal.displayName, "Normal")
    }

    func testNormalizedPoint() throws {
        let point = NormalizedPoint(x: 0.5, y: 0.5)
        XCTAssertEqual(point.x, 0.5)
        XCTAssertEqual(point.y, 0.5)
    }

    func testNormalizedBox() throws {
        let box = NormalizedBox(xMin: 0.1, yMin: 0.2, xMax: 0.8, yMax: 0.9)
        XCTAssertEqual(box.width, 0.7, accuracy: 0.001)
        XCTAssertEqual(box.height, 0.7, accuracy: 0.001)
    }
}
