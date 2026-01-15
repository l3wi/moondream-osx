import SwiftUI
import AppKit
import MoondreamKit

/// Panel displaying the loaded image with optional point/box overlays
struct ImagePanel: View {
    @Binding var image: NSImage?
    let points: [NormalizedPoint]
    let boxes: [NormalizedBox]
    let objectName: String

    @State private var isHoveringClear = false
    @State private var imageFrame: CGRect = .zero

    init(image: Binding<NSImage?>, points: [NormalizedPoint] = [], boxes: [NormalizedBox] = [], objectName: String = "") {
        self._image = image
        self.points = points
        self.boxes = boxes
        self.objectName = objectName
    }

    var body: some View {
        GeometryReader { geo in
            if let img = image {
                ZStack(alignment: .topTrailing) {
                    // Container for image + overlays
                    GeometryReader { imageGeo in
                        let imageSize = calculateImageSize(image: img, in: imageGeo.size)

                        ZStack {
                            // Image with rounded corners
                            Image(nsImage: img)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .shadow(color: .black.opacity(0.15), radius: 10, x: 0, y: 5)

                            // Overlay container - positioned and sized to match image
                            ZStack {
                                ImageResultOverlay(
                                    points: points,
                                    boxes: boxes,
                                    imageSize: imageSize,
                                    objectName: objectName
                                )
                            }
                            .frame(width: imageSize.width, height: imageSize.height)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }

                    // Floating clear button
                    clearButton
                        .padding(12)
                }
                .padding(20)
            }
        }
    }

    /// Calculate the displayed image size maintaining aspect ratio
    private func calculateImageSize(image: NSImage, in containerSize: CGSize) -> CGSize {
        let imageAspect = image.size.width / image.size.height
        let containerAspect = containerSize.width / containerSize.height

        if imageAspect > containerAspect {
            // Image is wider - fit to width
            let width = containerSize.width
            let height = width / imageAspect
            return CGSize(width: width, height: height)
        } else {
            // Image is taller - fit to height
            let height = containerSize.height
            let width = height * imageAspect
            return CGSize(width: width, height: height)
        }
    }

    private var clearButton: some View {
        Button(action: {
            withAnimation(.easeInOut(duration: 0.2)) {
                image = nil
            }
        }) {
            ZStack {
                Circle()
                    .fill(.ultraThinMaterial)
                    .frame(width: 32, height: 32)

                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(.secondary)
            }
        }
        .buttonStyle(.plain)
        .scaleEffect(isHoveringClear ? 1.1 : 1.0)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHoveringClear = hovering
            }
        }
        .help("Clear image")
    }
}
