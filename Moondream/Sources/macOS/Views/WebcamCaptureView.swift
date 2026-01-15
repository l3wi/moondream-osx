import SwiftUI
import CoreImage
import AppKit

/// Full-screen webcam capture view
struct WebcamCaptureView: View {
    @Binding var droppedImage: NSImage?
    @Binding var isWebcamMode: Bool
    @StateObject private var cameraService = MacCameraService()
    @State private var showError = false

    var body: some View {
        ZStack {
            // Camera preview (full bleed)
            if cameraService.isReady {
                CameraPreviewView(previewLayer: cameraService.previewLayer)
                    .ignoresSafeArea()
            } else {
                // Loading state
                ZStack {
                    Color.black
                    if cameraService.error == nil {
                        ProgressView("Starting camera...")
                            .foregroundStyle(.white)
                    }
                }
            }

            // Controls overlay
            VStack {
                // Top bar with close button
                HStack {
                    Spacer()
                    closeButton
                        .padding(20)
                }

                Spacer()

                // Bottom capture button
                CaptureButton(isProcessing: false, isFrozen: false) {
                    captureFrame()
                }
                .padding(.bottom, 60)
            }
        }
        .background(Color.black)
        .onAppear {
            cameraService.start()
        }
        .onDisappear {
            cameraService.stop()
        }
        .alert("Camera Error", isPresented: $showError) {
            Button("OK") {
                isWebcamMode = false
            }
        } message: {
            Text(cameraService.error?.errorDescription ?? "Unknown error")
        }
        .onChange(of: cameraService.error) { _, newError in
            showError = newError != nil
        }
    }

    private var closeButton: some View {
        Button {
            isWebcamMode = false
        } label: {
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 32))
                .foregroundStyle(.white.opacity(0.8))
                .shadow(radius: 4)
        }
        .buttonStyle(.plain)
    }

    private func captureFrame() {
        guard let ciImage = cameraService.captureCurrentFrame() else { return }

        // Convert CIImage to NSImage
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))

        // Set the dropped image and exit webcam mode
        droppedImage = nsImage
        isWebcamMode = false
    }
}

#Preview {
    WebcamCaptureView(
        droppedImage: .constant(nil),
        isWebcamMode: .constant(true)
    )
    .frame(width: 800, height: 600)
}
