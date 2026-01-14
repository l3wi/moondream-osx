import SwiftUI

/// Overlay for displaying points and bounding boxes on the frozen image
struct OverlayView: View {
    let points: [NormalizedPoint]
    let boxes: [NormalizedBox]
    let onPointTap: (NormalizedPoint) -> Void
    let onBoxTap: (NormalizedBox) -> Void

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Render bounding boxes first (behind points)
                ForEach(boxes) { box in
                    BoundingBoxView(box: box)
                        .frame(
                            width: box.width * geometry.size.width,
                            height: box.height * geometry.size.height
                        )
                        .position(
                            x: box.centerX * geometry.size.width,
                            y: box.centerY * geometry.size.height
                        )
                        .onTapGesture {
                            onBoxTap(box)
                        }
                }

                // Render points on top
                ForEach(points) { point in
                    PointMarkerView(point: point)
                        .position(
                            x: point.x * geometry.size.width,
                            y: point.y * geometry.size.height
                        )
                        .onTapGesture {
                            onPointTap(point)
                        }
                }
            }
        }
    }
}

#Preview {
    ZStack {
        Color.gray
        OverlayView(
            points: [
                NormalizedPoint(x: 0.3, y: 0.4),
                NormalizedPoint(x: 0.7, y: 0.6)
            ],
            boxes: [
                NormalizedBox(xMin: 0.1, yMin: 0.2, xMax: 0.5, yMax: 0.6)
            ],
            onPointTap: { _ in },
            onBoxTap: { _ in }
        )
    }
}
