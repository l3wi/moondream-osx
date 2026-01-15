import SwiftUI
import AppKit
import CoreImage
import MoondreamKit

/// Two-column layout for when an image is loaded
struct ImageLoadedView: View {
    @EnvironmentObject var service: MoondreamService
    @Binding var droppedImage: NSImage?
    @State private var selectedSkill: Skill = .caption
    @State private var queryText: String = "What is in this image?"
    @State private var objectText: String = ""
    @State private var captionLength: CaptionLength = .normal
    @State private var result: String = ""
    @State private var isProcessing: Bool = false

    // Overlay data for point/detect results
    @State private var overlayPoints: [NormalizedPoint] = []
    @State private var overlayBoxes: [NormalizedBox] = []
    @State private var overlayObjectName: String = ""

    var body: some View {
        HStack(spacing: 0) {
            // Left: Image panel with overlays (2/3)
            ImagePanel(
                image: $droppedImage,
                points: overlayPoints,
                boxes: overlayBoxes,
                objectName: overlayObjectName
            )
            .frame(maxWidth: .infinity)
            .background(.regularMaterial)

            // Semi-transparent white divider
            Rectangle()
                .fill(Color.white.opacity(0.3))
                .frame(width: 1)

            // Right: Toolbar panel (1/3)
            ToolbarPanel(
                selectedSkill: $selectedSkill,
                queryText: $queryText,
                objectText: $objectText,
                captionLength: $captionLength,
                result: $result,
                isProcessing: $isProcessing,
                onRun: runInference
            )
            .frame(width: 320)
        }
        // Clear overlays when skill changes
        .onChange(of: selectedSkill) { _, _ in
            clearOverlays()
        }
    }

    private func clearOverlays() {
        overlayPoints = []
        overlayBoxes = []
        overlayObjectName = ""
    }

    private func runInference() {
        guard let nsImage = droppedImage else { return }

        // Convert NSImage to CIImage
        guard let tiffData = nsImage.tiffRepresentation,
              let ciImage = CIImage(data: tiffData) else {
            result = "Error: Could not convert image"
            return
        }

        isProcessing = true
        result = ""
        clearOverlays()

        Task {
            do {
                let output: String
                var newPoints: [NormalizedPoint] = []
                var newBoxes: [NormalizedBox] = []
                var newObjectName: String = ""

                switch selectedSkill {
                case .query:
                    let queryResult = try await service.query(image: ciImage, question: queryText)
                    output = queryResult.answer
                case .caption:
                    let captionResult = try await service.caption(image: ciImage, length: captionLength)
                    output = captionResult.caption
                case .point:
                    let pointResult = try await service.point(image: ciImage, object: objectText)
                    output = formatPointResult(pointResult)
                    // Store structured data for overlay
                    newPoints = pointResult.points
                    newObjectName = objectText
                case .detect:
                    let detectResult = try await service.detect(image: ciImage, object: objectText)
                    output = formatDetectResult(detectResult)
                    // Store structured data for overlay
                    newBoxes = detectResult.boxes
                    newObjectName = objectText
                }

                await MainActor.run {
                    result = output
                    overlayPoints = newPoints
                    overlayBoxes = newBoxes
                    overlayObjectName = newObjectName
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    result = "Error: \(error.localizedDescription)"
                    clearOverlays()
                    isProcessing = false
                }
            }
        }
    }

    private func formatPointResult(_ pointResult: PointResult) -> String {
        if pointResult.points.isEmpty {
            return "No points found"
        }
        var output = "Found \(pointResult.points.count) point(s):\n"
        for (index, point) in pointResult.points.enumerated() {
            output += "\(index + 1). (\(String(format: "%.2f", point.x)), \(String(format: "%.2f", point.y)))\n"
        }
        return output
    }

    private func formatDetectResult(_ detectResult: DetectResult) -> String {
        if detectResult.boxes.isEmpty {
            return "No objects detected"
        }
        var output = "Detected \(detectResult.boxes.count) object(s):\n"
        for (index, box) in detectResult.boxes.enumerated() {
            output += "\(index + 1). Box: [\(String(format: "%.2f", box.xMin)), \(String(format: "%.2f", box.yMin)), \(String(format: "%.2f", box.xMax)), \(String(format: "%.2f", box.yMax))]\n"
        }
        return output
    }
}
