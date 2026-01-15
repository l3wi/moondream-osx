import SwiftUI

/// Floating capture button - shared between iOS and macOS
struct CaptureButton: View {
    let isProcessing: Bool
    let isFrozen: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                // Outer ring
                Circle()
                    .stroke(.white, lineWidth: 4)
                    .frame(width: 72, height: 72)

                // Inner circle
                Circle()
                    .fill(isFrozen ? .clear : .white)
                    .frame(width: 58, height: 58)

                if isProcessing {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .tint(.white)
                }
            }
        }
        .buttonStyle(.plain)
        .disabled(isProcessing || isFrozen)
        .opacity(isFrozen ? 0.5 : 1.0)
    }
}

#Preview {
    ZStack {
        Color.black
        VStack(spacing: 20) {
            CaptureButton(isProcessing: false, isFrozen: false) {}
            CaptureButton(isProcessing: true, isFrozen: false) {}
            CaptureButton(isProcessing: false, isFrozen: true) {}
        }
    }
}
