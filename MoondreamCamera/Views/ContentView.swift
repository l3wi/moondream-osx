import SwiftUI

#if os(iOS)
/// Root view that shows download progress or camera view
struct ContentView: View {
    @Environment(AppState.self) private var appState
    @StateObject private var moondreamService = MoondreamService.shared

    var body: some View {
        Group {
            if !appState.isModelLoaded {
                DownloadProgressView()
                    .task {
                        await loadModel()
                    }
            } else {
                CameraView()
            }
        }
        .preferredColorScheme(.dark)
    }

    private func loadModel() async {
        guard !appState.isModelLoading else { return }
        appState.isModelLoading = true
        appState.modelError = nil

        do {
            try await moondreamService.loadModel { progress in
                Task { @MainActor in
                    appState.modelDownloadProgress = progress
                }
            }
            appState.isModelLoaded = true
        } catch {
            appState.modelError = error.localizedDescription
        }

        appState.isModelLoading = false
    }
}

#Preview {
    ContentView()
        .environment(AppState())
}
#endif
