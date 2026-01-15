import SwiftUI
import CoreImage
import MoondreamKit

#if os(iOS)
/// Main camera view with capture and settings
struct CameraView: View {
    // No model mode - shows greyed out buttons that redirect to download
    var noModelMode: Bool = false
    var onRequestDownload: (() -> Void)? = nil

    @Environment(AppState.self) private var appState
    @StateObject private var cameraService = CameraService()
    @StateObject private var moondreamService = MoondreamService.shared

    var body: some View {
        ZStack {
            // Camera preview or frozen image
            if appState.isFrozen, let ciImage = appState.capturedImage {
                FrozenImageView(image: ciImage)
                    .ignoresSafeArea()

                // Overlay for point/detect results
                if let result = appState.currentResult {
                    OverlayView(
                        points: result.points ?? [],
                        boxes: result.boxes ?? [],
                        onPointTap: { point in
                            appState.selectedOverlayItem = .point(point)
                        },
                        onBoxTap: { box in
                            appState.selectedOverlayItem = .box(box)
                        }
                    )
                    .ignoresSafeArea()
                }
            } else {
                CameraPreviewView(previewLayer: cameraService.previewLayer)
                    .ignoresSafeArea()
            }

            // Controls overlay
            VStack {
                Spacer()

                // Bottom controls
                HStack(alignment: .bottom) {
                    // Skill selector (expands upward and rightward, bottom-left anchored)
                    // Fixed 56x56 footprint - expanded menu overlays without shifting layout
                    SkillSelectorButton(selectedSkill: Binding(
                        get: { appState.selectedSkill },
                        set: { appState.selectedSkill = $0 }
                    ))

                    Spacer()

                    // Capture button or close button
                    GlassEffectContainer {
                        if appState.isFrozen && appState.currentResult != nil && !appState.isProcessing {
                            // Show close button only after we have results
                            CloseButton {
                                appState.resetCapture()
                                cameraService.resume()
                            }
                        } else {
                            // Show capture button (with spinner when processing)
                            CaptureButton(
                                isProcessing: appState.isProcessing,
                                isFrozen: appState.isFrozen
                            ) {
                                if noModelMode {
                                    // Redirect to download screen
                                    onRequestDownload?()
                                } else {
                                    captureAndProcess()
                                }
                            }
                            .opacity(noModelMode ? 0.5 : 1.0)
                        }
                    }

                    Spacer()

                    // Settings button
                    SettingsButton {
                        appState.showSettings = true
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 40)
            }

            // Input modal overlay (centered floating modal)
            if appState.showInputModal {
                InputModal(
                    skill: appState.selectedSkill,
                    onSubmit: {
                        // Run inference with captured image
                        if let image = appState.capturedImage {
                            Task {
                                await runInference(image: image)
                            }
                        }
                    },
                    onDismiss: {
                        // User dismissed without running - reset capture
                        appState.showInputModal = false
                        appState.resetCapture()
                        cameraService.resume()
                    }
                )
                .transition(.opacity.combined(with: .scale(scale: 0.95)))
                .animation(.easeOut(duration: 0.2), value: appState.showInputModal)
            }
        }
        .onAppear {
            cameraService.start()
        }
        .onDisappear {
            cameraService.stop()
        }
        .sheet(isPresented: Binding(
            get: { appState.showSettings },
            set: { appState.showSettings = $0 }
        )) {
            SettingsSheet()
        }
        .sheet(isPresented: Binding(
            get: { appState.showResults },
            set: { appState.showResults = $0 }
        )) {
            ResultsSheet()
        }
        .alert(item: Binding(
            get: { cameraService.error },
            set: { _ in }
        )) { error in
            Alert(
                title: Text("Camera Error"),
                message: Text(error.localizedDescription),
                dismissButton: .default(Text("OK"))
            )
        }
        // Overlay item detail popup
        .sheet(item: Binding(
            get: { appState.selectedOverlayItem },
            set: { appState.selectedOverlayItem = $0 }
        )) { item in
            OverlayDetailSheet(item: item)
                .presentationDetents([.height(200)])
        }
    }

    private func captureAndProcess() {
        guard !appState.isFrozen else { return }

        guard let frame = cameraService.captureCurrentFrame() else {
            return
        }

        appState.prepareCapture(image: frame)

        // Show input modal for all skills (caption for length picker, others for text input)
        appState.showInputModal = true
    }

    private func runInference(image: CIImage) async {
        appState.isProcessing = true

        do {
            let result: MoondreamResult

            switch appState.selectedSkill {
            case .query:
                let queryResult = try await moondreamService.query(
                    image: image,
                    question: appState.queryText
                )
                result = .query(queryResult)

            case .caption:
                let captionResult = try await moondreamService.caption(
                    image: image,
                    length: appState.captionLength
                )
                result = .caption(captionResult)

            case .point:
                let pointResult = try await moondreamService.point(
                    image: image,
                    object: appState.targetObject
                )
                result = .point(pointResult)

            case .detect:
                let detectResult = try await moondreamService.detect(
                    image: image,
                    object: appState.targetObject
                )
                result = .detect(detectResult)
            }

            appState.currentResult = result

            // Only show results sheet for caption and query
            // Point and detect render interactive overlays on the image
            if appState.selectedSkill == .caption || appState.selectedSkill == .query {
                appState.showResults = true
            }
        } catch {
            appState.modelError = error.localizedDescription
        }

        appState.isProcessing = false
    }
}

/// Processing indicator overlay with Liquid Glass effect
struct ProcessingOverlay: View {
    var body: some View {
        GlassEffectContainer {
            VStack(spacing: 16) {
                ProgressView()
                    .progressViewStyle(.circular)
                    .tint(.white)
                    .scaleEffect(1.5)

                Text("Processing...")
                    .font(.headline)
                    .foregroundStyle(.white)
            }
            .padding(32)
            .glassEffect(.regular, in: .rect(cornerRadius: 20))
        }
    }
}

#Preview {
    CameraView()
        .environment(AppState())
}
#endif // os(iOS)
