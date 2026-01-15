import SwiftUI
@preconcurrency import AVFoundation
#if canImport(UIKit)
import UIKit
#endif

/// UIViewRepresentable wrapper for AVCaptureVideoPreviewLayer
#if os(iOS)
struct CameraPreviewView: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer?

    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        uiView.previewLayer = previewLayer
    }
}

/// UIView that hosts the preview layer
class CameraPreviewUIView: UIView {
    var previewLayer: AVCaptureVideoPreviewLayer? {
        didSet {
            // Remove old layer
            oldValue?.removeFromSuperlayer()

            // Add new layer
            if let previewLayer {
                previewLayer.frame = bounds
                layer.addSublayer(previewLayer)
            }
        }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
}

#Preview {
    CameraPreviewView(previewLayer: nil)
}

#endif
