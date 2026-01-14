import SwiftUI

#if os(iOS)
/// Detail sheet for tapped overlay items
struct OverlayDetailSheet: View {
    let item: OverlayItem

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                switch item {
                case .point(let point):
                    PointDetailView(point: point)

                case .box(let box):
                    BoxDetailView(box: box)
                }

                Spacer()
            }
            .padding()
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var title: String {
        switch item {
        case .point: "Point Details"
        case .box: "Detection Details"
        }
    }
}

/// Detailed view for a point
struct PointDetailView: View {
    let point: NormalizedPoint

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let label = point.label {
                DetailRow(label: "Label", value: label)
            }

            DetailRow(
                label: "Position",
                value: "(\(String(format: "%.3f", point.x)), \(String(format: "%.3f", point.y)))"
            )

            DetailRow(
                label: "X Coordinate",
                value: "\(String(format: "%.1f", point.x * 100))% from left"
            )

            DetailRow(
                label: "Y Coordinate",
                value: "\(String(format: "%.1f", point.y * 100))% from top"
            )
        }
    }
}

/// Detailed view for a bounding box
struct BoxDetailView: View {
    let box: NormalizedBox

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let label = box.label {
                DetailRow(label: "Label", value: label)
            }

            if let confidence = box.confidence {
                DetailRow(label: "Confidence", value: "\(Int(confidence * 100))%")
            }

            DetailRow(
                label: "Top-Left",
                value: "(\(String(format: "%.3f", box.xMin)), \(String(format: "%.3f", box.yMin)))"
            )

            DetailRow(
                label: "Bottom-Right",
                value: "(\(String(format: "%.3f", box.xMax)), \(String(format: "%.3f", box.yMax)))"
            )

            DetailRow(
                label: "Size",
                value: "\(String(format: "%.1f", box.width * 100))% × \(String(format: "%.1f", box.height * 100))%"
            )
        }
    }
}

/// Row for displaying a label-value pair
struct DetailRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)

            Spacer()

            Text(value)
                .fontWeight(.medium)
                .monospaced()
        }
        .font(.subheadline)
    }
}

#Preview {
    OverlayDetailSheet(item: .point(NormalizedPoint(x: 0.456, y: 0.789, label: "Person")))
}
#endif // os(iOS)
