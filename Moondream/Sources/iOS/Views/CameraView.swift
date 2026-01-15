import SwiftUI
import CoreImage
import MoondreamKit

#if os(iOS)
/// Main camera view with capture and settings
struct CameraView: View {
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

                // Bottom controls with Liquid Glass container
                GlassEffectContainer {
                    HStack(alignment: .bottom) {
                        // Skill indicator
                        SkillIndicator(skill: appState.selectedSkill)
                            .frame(width: 80)

                        Spacer()

                        // Capture button or close button when frozen
                        if appState.isFrozen {
                            CloseButton {
                                appState.resetCapture()
                                cameraService.resume()
                            }
                        } else {
                            CaptureButton(
                                isProcessing: appState.isProcessing,
                                isFrozen: appState.isFrozen
                            ) {
                                captureAndProcess()
                            }
                        }

                        Spacer()

                        // Settings button
                        SettingsButton {
                            appState.showSettings = true
                        }
                        .frame(width: 80)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 40)
            }

            // Processing indicator
            if appState.isProcessing {
                ProcessingOverlay()
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

        if appState.selectedSkill == .query {
            // For query, show results sheet first to get question
            appState.showResults = true
        } else {
            // For other skills, run inference immediately
            Task {
                await runInference(image: frame)
            }
        }
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
            appState.showResults = true
        } catch {
            appState.modelError = error.localizedDescription
        }

        appState.isProcessing = false
    }
}

/// Shows the current skill as an indicator with Liquid Glass effect
struct SkillIndicator: View {
    let skill: Skill

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: skill.icon)
                .font(.title3)
            Text(skill.displayName)
                .font(.caption2)
        }
        .foregroundStyle(.white)
        .padding(8)
        .glassEffect(.regular, in: .rect(cornerRadius: 12))
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
