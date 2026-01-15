import SwiftUI
import MoondreamKit

/// Initial model download screen shown when no models are downloaded
/// Styled with iOS 26 Liquid Glass design
struct ModelDownloadView: View {
    @Environment(AppState.self) private var appState
    let showSkipButton: Bool
    let onSkip: () -> Void
    let onModelDownloaded: () -> Void

    init(
        showSkipButton: Bool = true,
        onSkip: @escaping () -> Void = {},
        onModelDownloaded: @escaping () -> Void = {}
    ) {
        self.showSkipButton = showSkipButton
        self.onSkip = onSkip
        self.onModelDownloaded = onModelDownloaded
    }

    var body: some View {
        ZStack {
            // Background
            Color.black.ignoresSafeArea()

            VStack(spacing: 24) {
                Spacer()

                // Moon icon
                Image(systemName: "moon.fill")
                    .font(.system(size: 64, weight: .thin))
                    .foregroundStyle(.secondary)

                // Title
                Text("Moondream")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .foregroundStyle(.primary)

                // Subtitle
                VStack(spacing: 8) {
                    Text("No models found")
                        .font(.title3)
                        .foregroundStyle(.primary)
                    Text("Download a model to get started:")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                // Model list with Liquid Glass container
                GlassEffectContainer {
                    VStack(spacing: 12) {
                        ForEach(AvailableModels.all) { model in
                            ModelDownloadRow(
                                model: model,
                                onDownloadComplete: {
                                    onModelDownloaded()
                                }
                            )
                        }
                    }
                }
                .padding(.horizontal, 20)

                Spacer()

                // Footer with skip button
                VStack(spacing: 16) {
                    Text("Models are downloaded from HuggingFace")
                        .font(.caption)
                        .foregroundStyle(.tertiary)

                    if showSkipButton {
                        Button("Skip for now") {
                            onSkip()
                        }
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 20)
                        .glassEffect(.clear, in: .capsule)
                    }
                }
                .padding(.bottom, 40)
            }
        }
        .preferredColorScheme(.dark)
    }
}

/// A row displaying model info with download capability and Liquid Glass styling
struct ModelDownloadRow: View {
    let model: ModelInfo
    let onDownloadComplete: () -> Void
    @Environment(AppState.self) private var appState
    @State private var downloadTask: Task<Void, Never>?

    private var isDownloading: Bool {
        appState.downloadingModelId == model.id
    }

    private var isDownloaded: Bool {
        appState.downloadedModelIds.contains(model.id)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                // Model info
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.displayName)
                        .font(.headline)
                        .foregroundStyle(.primary)

                    Text(model.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)

                    HStack(spacing: 8) {
                        Text(model.quantization)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)

                        Text("•")
                            .foregroundStyle(.tertiary)

                        Text(model.sizeDisplay)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }

                    // Compatibility warning for iOS
                    if !model.isCompatibleWithiOS {
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.caption2)
                            Text("Not compatible with this device")
                                .font(.caption2)
                        }
                        .foregroundStyle(.orange)
                    }
                }

                Spacer()

                // Action button / status
                if isDownloading {
                    Button {
                        cancelDownload()
                    } label: {
                        Text("Cancel")
                            .font(.subheadline)
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.plain)
                } else if isDownloaded {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.green)
                } else if !model.isCompatibleWithiOS {
                    // Incompatible - show disabled state
                    Image(systemName: "xmark.circle")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                } else {
                    Button("Download") {
                        startDownload()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }
            }

            // Progress bar (shown when downloading)
            if isDownloading {
                VStack(spacing: 6) {
                    // Progress bar
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            // Background track
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.secondary.opacity(0.2))

                            // Progress fill
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.blue)
                                .frame(width: max(0, geo.size.width * appState.modelDownloadProgress))
                        }
                    }
                    .frame(height: 6)

                    // Progress text
                    HStack {
                        Text("Downloading...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(Int(appState.modelDownloadProgress * 100))%")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundStyle(.primary)
                    }
                }
                .padding(.top, 12)
            }
        }
        .padding(16)
        .glassEffect(.regular, in: .rect(cornerRadius: 12))
        .animation(.easeInOut(duration: 0.2), value: isDownloading)
    }

    private func startDownload() {
        downloadTask = Task {
            appState.downloadingModelId = model.id
            appState.modelDownloadProgress = 0

            do {
                let configuration = Moondream3Loader.configuration(for: model.id)
                _ = try await Moondream3Loader.loadContainer(
                    configuration: configuration
                ) { progress in
                    Task { @MainActor in
                        // Check if cancelled
                        guard appState.downloadingModelId == model.id else { return }
                        appState.modelDownloadProgress = progress.fractionCompleted
                    }
                }

                // Check if cancelled before completing
                guard !Task.isCancelled else {
                    await MainActor.run {
                        appState.downloadingModelId = nil
                    }
                    return
                }

                await MainActor.run {
                    appState.downloadedModelIds.insert(model.id)
                    appState.selectedModelId = model.id
                    appState.downloadingModelId = nil
                    onDownloadComplete()
                }
            } catch {
                if !Task.isCancelled {
                    await MainActor.run {
                        appState.modelError = error.localizedDescription
                        appState.downloadingModelId = nil
                    }
                }
            }
        }
    }

    private func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil
        appState.downloadingModelId = nil
        appState.modelDownloadProgress = 0
    }
}

#Preview {
    ModelDownloadView(showSkipButton: true, onSkip: {}, onModelDownloaded: {})
        .environment(AppState())
}
