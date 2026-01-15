import SwiftUI
import MoondreamKit

#if os(iOS)
/// Root view that manages the app flow:
/// 1. No models downloaded → ModelDownloadView
/// 2. Model not loaded → DownloadProgressView (loading)
/// 3. Model loaded → CameraView
struct ContentView: View {
    @Environment(AppState.self) private var appState
    @StateObject private var moondreamService = MoondreamService.shared
    @State private var hasSkippedDownload = false

    var body: some View {
        Group {
            if !appState.hasDownloadedModels && !hasSkippedDownload {
                // No models downloaded - show download screen
                ModelDownloadView(
                    showSkipButton: true,
                    onSkip: { hasSkippedDownload = true },
                    onModelDownloaded: {
                        // Model downloaded, will auto-load
                        Task {
                            await loadModel()
                        }
                    }
                )
            } else if !appState.hasDownloadedModels && hasSkippedDownload {
                // Skipped but no model - show minimal prompt
                NoModelView(onDownload: { hasSkippedDownload = false })
            } else if !appState.isModelLoaded {
                // Has model but not loaded - show loading progress
                DownloadProgressView()
                    .task {
                        await loadModel()
                    }
            } else {
                // Ready - show camera
                CameraView()
            }
        }
        .preferredColorScheme(.dark)
        .task {
            // Refresh downloaded models on launch
            appState.refreshDownloadedModels()
        }
    }

    private func loadModel() async {
        guard !appState.isModelLoading else { return }
        appState.isModelLoading = true
        appState.modelError = nil

        do {
            try await moondreamService.loadModel(modelId: appState.selectedModelId)
            appState.isModelLoaded = true
        } catch {
            appState.modelError = error.localizedDescription
        }

        appState.isModelLoading = false
    }
}

/// Minimal view shown after skipping download - prompts to return
struct NoModelView: View {
    let onDownload: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 20) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 48))
                    .foregroundStyle(.orange)

                Text("Model Required")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundStyle(.primary)

                Text("Download a model to use Moondream")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                Button("Download Model") {
                    onDownload()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
            .padding()
        }
    }
}

#Preview("Content View") {
    ContentView()
        .environment(AppState())
}

#Preview("No Model View") {
    NoModelView(onDownload: {})
}
#endif
