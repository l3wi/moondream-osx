import SwiftUI
import MoondreamKit

#if os(iOS)
/// Root view that manages the app flow:
/// 1. No models downloaded → ModelDownloadView
/// 2. Model downloaded → CameraView (model loads on-demand when running inference)
struct ContentView: View {
    @Environment(AppState.self) private var appState
    @State private var hasSkippedDownload = false

    var body: some View {
        Group {
            if !appState.hasDownloadedModels && !hasSkippedDownload {
                // No models downloaded - show download screen
                ModelDownloadView(
                    showSkipButton: true,
                    onSkip: { hasSkippedDownload = true },
                    onModelDownloaded: {
                        // Model downloaded - camera view will load model on first run
                    }
                )
            } else if !appState.hasDownloadedModels && hasSkippedDownload {
                // Skipped - show camera but with disabled buttons that redirect to download
                CameraView(noModelMode: true, onRequestDownload: { hasSkippedDownload = false })
            } else {
                // Has model downloaded - show camera (model loads on-demand when running)
                CameraView()
            }
        }
        .preferredColorScheme(.dark)
        .task {
            // Refresh downloaded models on launch
            appState.refreshDownloadedModels()
        }
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
