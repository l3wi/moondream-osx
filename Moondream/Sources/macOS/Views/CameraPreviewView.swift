import SwiftUI
import AVFoundation
import AppKit

/// NSViewRepresentable wrapper for AVCaptureVideoPreviewLayer on macOS
struct CameraPreviewView: NSViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer?

    func makeNSView(context: Context) -> CameraPreviewNSView {
        let view = CameraPreviewNSView()
        view.wantsLayer = true
        return view
    }

    func updateNSView(_ nsView: CameraPreviewNSView, context: Context) {
        nsView.updatePreviewLayer(previewLayer)
    }
}

/// Custom NSView that hosts the camera preview layer
class CameraPreviewNSView: NSView {
    private var currentPreviewLayer: AVCaptureVideoPreviewLayer?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        wantsLayer = true
    }

    func updatePreviewLayer(_ newLayer: AVCaptureVideoPreviewLayer?) {
        // Remove old layer if different
        if currentPreviewLayer !== newLayer {
            currentPreviewLayer?.removeFromSuperlayer()
            currentPreviewLayer = newLayer

            if let layer = newLayer {
                layer.frame = bounds
                layer.videoGravity = .resizeAspectFill
                self.layer?.addSublayer(layer)
            }
        }
    }

    override func layout() {
        super.layout()
        // Update layer frame when view resizes
        currentPreviewLayer?.frame = bounds
    }
}
