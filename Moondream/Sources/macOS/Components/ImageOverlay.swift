import SwiftUI
import Foundation
import MoondreamKit

// MARK: - Point Overlay

/// Semi-transparent dot overlay for point results
struct PointOverlay: View {
    let points: [NormalizedPoint]
    let imageSize: CGSize
    let objectName: String

    var body: some View {
        ForEach(points) { point in
            PointMarker(point: point, imageSize: imageSize, objectName: objectName)
        }
    }
}

/// Individual point marker with hover effect
struct PointMarker: View {
    let point: NormalizedPoint
    let imageSize: CGSize
    let objectName: String

    @State private var isHovering = false

    private let dotSize: CGFloat = 14
    private let borderWidth: CGFloat = 2.5
    private let hoverScale: CGFloat = 1.3

    var body: some View {
        let position = point.position(in: imageSize)

        ZStack {
            // Outer glow ring (visible on hover)
            Circle()
                .fill(Color.white.opacity(0.25))
                .frame(width: dotSize * 2.2, height: dotSize * 2.2)
                .opacity(isHovering ? 1 : 0)

            // Main dot - white fill with dark border
            Circle()
                .fill(Color.white)
                .frame(width: dotSize, height: dotSize)

            // Border stroke
            Circle()
                .stroke(Color.black.opacity(0.6), lineWidth: borderWidth)
                .frame(width: dotSize, height: dotSize)
        }
        .frame(width: dotSize * 2.2, height: dotSize * 2.2)
        .contentShape(Circle())
        .shadow(color: .black.opacity(0.4), radius: 3, x: 0, y: 2)
        .scaleEffect(isHovering ? hoverScale : 1.0)
        .position(x: position.x, y: position.y)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovering = hovering
            }
        }
        .help(point.label ?? objectName)
    }
}

// MARK: - Detect Overlay

/// Bounding box overlay for detect results
struct DetectOverlay: View {
    let boxes: [NormalizedBox]
    let imageSize: CGSize
    let objectName: String

    var body: some View {
        ForEach(boxes) { box in
            BoxMarker(box: box, imageSize: imageSize, objectName: objectName)
        }
    }
}

/// Individual bounding box marker with hover effect
struct BoxMarker: View {
    let box: NormalizedBox
    let imageSize: CGSize
    let objectName: String

    @State private var isHovering = false

    private let borderWidth: CGFloat = 2.5
    private let cornerRadius: CGFloat = 8
    private let hoverScale: CGFloat = 1.02

    var body: some View {
        let frame = box.frame(in: imageSize)

        ZStack {
            // Semi-transparent fill (visible on hover)
            RoundedRectangle(cornerRadius: cornerRadius)
                .fill(Color.white.opacity(isHovering ? 0.15 : 0.05))

            // Border
            RoundedRectangle(cornerRadius: cornerRadius)
                .stroke(Color.white, lineWidth: borderWidth)
                .shadow(color: .black.opacity(0.5), radius: 2, x: 0, y: 1)

            // Label badge (visible on hover) - centered
            if isHovering {
                Text(box.label ?? objectName)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        Capsule()
                            .fill(Color.black.opacity(0.7))
                    )
            }
        }
        .frame(width: frame.width, height: frame.height)
        .contentShape(RoundedRectangle(cornerRadius: cornerRadius))
        .scaleEffect(isHovering ? hoverScale : 1.0)
        .position(x: frame.midX, y: frame.midY)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovering = hovering
            }
        }
        .help(box.label ?? objectName)
    }
}

// MARK: - Combined Overlay

/// Combined overlay that can show either points or boxes
struct ImageResultOverlay: View {
    let points: [NormalizedPoint]
    let boxes: [NormalizedBox]
    let imageSize: CGSize
    let objectName: String

    var body: some View {
        ZStack {
            // Point markers
            if !points.isEmpty {
                PointOverlay(points: points, imageSize: imageSize, objectName: objectName)
            }

            // Box markers
            if !boxes.isEmpty {
                DetectOverlay(boxes: boxes, imageSize: imageSize, objectName: objectName)
            }
        }
    }
}
