import SwiftUI
import UIKit

#if os(iOS)
/// View showing model download progress
struct DownloadProgressView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            // App icon/logo
            Image(systemName: "moon.fill")
                .font(.system(size: 80))
                .foregroundStyle(.white)

            Text("Moondream Camera")
                .font(.title)
                .fontWeight(.bold)
                .foregroundStyle(.white)

            VStack(spacing: 16) {
                if let error = appState.modelError {
                    // Error state
                    VStack(spacing: 12) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.largeTitle)
                            .foregroundStyle(.red)

                        Text("Download Failed")
                            .font(.headline)
                            .foregroundStyle(.white)

                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)

                        Button("Retry") {
                            appState.modelError = nil
                        }
                        .buttonStyle(.borderedProminent)
                        .padding(.top)
                    }
                } else {
                    // Progress state
                    VStack(spacing: 12) {
                        ProgressView(value: appState.modelDownloadProgress)
                            .progressViewStyle(.linear)
                            .frame(width: 250)
                            .tint(.white)

                        Text(progressText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.horizontal, 40)

            Spacer()

            // Requirements notice
            VStack(spacing: 8) {
                Text("Requires iPhone 15 Pro or newer")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                Text("Model size: ~2GB")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.bottom, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
        .onAppear {
            // Prevent screen from sleeping during download
            UIApplication.shared.isIdleTimerDisabled = true
        }
        .onDisappear {
            // Re-enable screen sleep
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    private var progressText: String {
        let progress = appState.modelDownloadProgress
        if progress < 0.8 {
            let downloadPercent = Int(progress / 0.8 * 100)
            return "Downloading model... \(downloadPercent)%"
        } else {
            let loadPercent = Int((progress - 0.8) / 0.2 * 100)
            return "Loading model... \(loadPercent)%"
        }
    }
}

#Preview {
    let state = AppState()
    state.modelDownloadProgress = 0.45
    return DownloadProgressView()
        .environment(state)
}
#endif // os(iOS)
