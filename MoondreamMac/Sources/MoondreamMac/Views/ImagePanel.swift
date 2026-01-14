import SwiftUI
import AppKit

/// Panel displaying the loaded image with a clear button
struct ImagePanel: View {
    @Binding var image: NSImage?
    @State private var isHoveringClear = false

    var body: some View {
        GeometryReader { geo in
            if let img = image {
                ZStack(alignment: .topTrailing) {
                    // Image with rounded corners
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .shadow(color: .black.opacity(0.15), radius: 10, x: 0, y: 5)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    // Floating clear button
                    clearButton
                        .padding(12)
                }
                .padding(20)
            }
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
