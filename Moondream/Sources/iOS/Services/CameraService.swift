@preconcurrency import AVFoundation
import CoreImage
import SwiftUI

/// Service for managing camera capture
final class CameraService: NSObject, ObservableObject, @unchecked Sendable {
    @MainActor @Published var previewLayer: AVCaptureVideoPreviewLayer?
    @MainActor @Published var isReady = false
    @MainActor @Published var error: CameraError?
    @MainActor private(set) var lastFrame: CIImage?

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session", qos: .userInitiated)
    private var outputDelegate: CameraOutputDelegate?

    override init() {
        super.init()
    }

    /// Request camera permissions and setup session
    @MainActor
    func start() {
        checkPermissions()
    }

    /// Stop the camera session
    func stop() {
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }

    /// Resume the camera session
    func resume() {
        sessionQueue.async { [weak self] in
            guard let self, !self.session.isRunning else { return }
            self.session.startRunning()
        }
    }

    /// Capture the current frame
    @MainActor
    func captureCurrentFrame() -> CIImage? {
        return lastFrame
    }

    // MARK: - Private Methods

    @MainActor
    private func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    if granted {
                        self?.setupSession()
                    } else {
                        self?.error = .permissionDenied
                    }
                }
            }
        case .denied, .restricted:
            error = .permissionDenied
        @unknown default:
            error = .permissionDenied
        }
    }

    @MainActor
    private func setupSession() {
        // Create a delegate wrapper
        let delegate = CameraOutputDelegate { [weak self] image in
            Task { @MainActor in
                self?.lastFrame = image
            }
        }
        self.outputDelegate = delegate

        sessionQueue.async { [weak self] in
            guard let self else { return }

            self.session.beginConfiguration()
            self.session.sessionPreset = .photo

            // Add camera input
            guard let camera = AVCaptureDevice.default(
                .builtInWideAngleCamera,
                for: .video,
                position: .back
            ) else {
                Task { @MainActor in
                    self.error = .cameraUnavailable
                }
                return
            }

            do {
                let input = try AVCaptureDeviceInput(device: camera)
                if self.session.canAddInput(input) {
                    self.session.addInput(input)
                }
            } catch {
                Task { @MainActor in
                    self.error = .setupFailed(error.localizedDescription)
                }
                return
            }

            // Add video output for frame capture
            self.videoOutput.setSampleBufferDelegate(delegate, queue: self.sessionQueue)
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]

            if self.session.canAddOutput(self.videoOutput) {
                self.session.addOutput(self.videoOutput)
            }

            // Set video orientation
            if let connection = self.videoOutput.connection(with: .video) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
            }

            self.session.commitConfiguration()

            // Create preview layer
            let layer = AVCaptureVideoPreviewLayer(session: self.session)
            layer.videoGravity = .resizeAspectFill

            Task { @MainActor in
                self.previewLayer = layer
                self.isReady = true
            }

            self.session.startRunning()
        }
    }
}

// MARK: - Camera Output Delegate

/// Separate delegate class to handle sample buffer callbacks
private final class CameraOutputDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let onFrame: @Sendable (CIImage) -> Void

    init(onFrame: @escaping @Sendable (CIImage) -> Void) {
        self.onFrame = onFrame
        super.init()
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        onFrame(ciImage)
    }
}

/// Camera errors
enum CameraError: LocalizedError, Identifiable {
    case permissionDenied
    case cameraUnavailable
    case setupFailed(String)

    var id: String {
        switch self {
        case .permissionDenied: "permissionDenied"
        case .cameraUnavailable: "cameraUnavailable"
        case .setupFailed: "setupFailed"
        }
    }

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            "Camera permission denied. Please enable camera access in Settings."
        case .cameraUnavailable:
            "Camera is not available on this device."
        case .setupFailed(let reason):
            "Failed to setup camera: \(reason)"
        }
    }
}
