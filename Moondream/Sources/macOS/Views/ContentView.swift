import SwiftUI
import AppKit
import CoreImage

/// View modifier for Liquid Glass effect with fallback
extension View {
    @ViewBuilder
    func liquidGlass() -> some View {
        if #available(macOS 26.0, *) {
            self.glassEffect()
        } else {
            self.background(.ultraThinMaterial)
        }
    }
}

/// Main content view that switches between empty, webcam, and loaded states
struct ContentView: View {
    @EnvironmentObject var service: MoondreamService
    @State private var droppedImage: NSImage?
    @State private var isWebcamMode: Bool = false

    var body: some View {
        Group {
            if isWebcamMode {
                WebcamCaptureView(
                    droppedImage: $droppedImage,
                    isWebcamMode: $isWebcamMode
                )
            } else if droppedImage != nil {
                ImageLoadedView(droppedImage: $droppedImage)
            } else {
                EmptyStateView(droppedImage: $droppedImage, isWebcamMode: $isWebcamMode)
            }
        }
        .frame(minWidth: 800, minHeight: 600)
        .background(.clear)
        .animation(.easeInOut(duration: 0.3), value: droppedImage != nil)
        .task {
            // Auto-load model on launch
            if !service.isLoaded && !service.isLoading {
                try? await service.loadModel()
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(MoondreamService.shared)
}
