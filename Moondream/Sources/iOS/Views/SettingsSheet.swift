import SwiftUI
import MoondreamKit
import Darwin

#if os(iOS)

/// Get available memory in bytes using os_proc_available_memory
private func getAvailableMemory() -> UInt64 {
    return UInt64(os_proc_available_memory())
}

/// Format bytes as human readable string
private func formatBytes(_ bytes: UInt64) -> String {
    let gb = Double(bytes) / (1024 * 1024 * 1024)
    if gb >= 1.0 {
        return String(format: "%.2f GB", gb)
    } else {
        let mb = Double(bytes) / (1024 * 1024)
        return String(format: "%.0f MB", mb)
    }
}
/// Settings sheet for skill selection and options
struct SettingsSheet: View {
    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss
    @StateObject private var moondreamService = MoondreamService.shared

    // Computed model lists
    private var downloadedModels: [ModelInfo] {
        AvailableModels.all.filter { appState.downloadedModelIds.contains($0.id) }
    }

    private var availableForDownload: [ModelInfo] {
        AvailableModels.all.filter { !appState.downloadedModelIds.contains($0.id) }
    }

    var body: some View {
        @Bindable var state = appState

        NavigationStack {
            Form {
                // Models section
                Section {
                    // Downloaded models
                    ForEach(downloadedModels) { model in
                        ModelRow(
                            model: model,
                            isSelected: appState.selectedModelId == model.id,
                            isDownloaded: true,
                            isDownloading: false,
                            downloadProgress: 0,
                            onSelect: { selectModel(model) },
                            onDownload: {},
                            onDelete: { deleteModel(model) }
                        )
                        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                            Button(role: .destructive) {
                                deleteModel(model)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }

                    // Available for download
                    ForEach(availableForDownload) { model in
                        ModelRow(
                            model: model,
                            isSelected: false,
                            isDownloaded: false,
                            isDownloading: appState.downloadingModelId == model.id,
                            downloadProgress: appState.modelDownloadProgress,
                            onSelect: {},
                            onDownload: { Task { await downloadModel(model) } },
                            onDelete: {}
                        )
                    }
                } header: {
                    Text("Models")
                } footer: {
                    if downloadedModels.isEmpty {
                        Text("No models downloaded. Download a model to use Moondream.")
                    } else {
                        Text("Tap to select. Swipe left to delete.")
                    }
                }

                // Skill selection
                Section {
                    ForEach(Skill.allCases) { skill in
                        SkillRow(
                            skill: skill,
                            isSelected: appState.selectedSkill == skill
                        ) {
                            state.selectedSkill = skill
                        }
                    }
                } header: {
                    Text("Skill")
                }

                // Skill-specific options
                switch appState.selectedSkill {
                case .caption:
                    Section {
                        Picker("Length", selection: $state.captionLength) {
                            ForEach(CaptionLength.allCases) { length in
                                Text(length.displayName).tag(length)
                            }
                        }
                        .pickerStyle(.segmented)
                    } header: {
                        Text("Caption Length")
                    }

                case .point, .detect:
                    Section {
                        TextField("e.g., person, car, dog", text: $state.targetObject)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                    } header: {
                        Text("Object to Find")
                    } footer: {
                        Text("Enter what you want to \(appState.selectedSkill == .point ? "locate" : "detect") in the image")
                    }

                case .query:
                    Section {
                        Text("You'll be prompted to enter your question after capturing an image")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } header: {
                        Text("Query Mode")
                    }
                }

                // Debug options
                Section {
                    Toggle("Debug Mode", isOn: $state.debugMode)
                } header: {
                    Text("Developer")
                } footer: {
                    Text("Shows raw model output alongside results")
                }

                // Memory info (always visible for debugging memory issues)
                Section {
                    MemoryInfoView()
                } header: {
                    Text("Memory")
                } footer: {
                    Text("Available memory for the app. Model requires ~2GB+ free.")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
        .presentationDetents([.medium, .large])
        .onAppear {
            appState.refreshDownloadedModels()
        }
    }

    // MARK: - Model Actions

    private func selectModel(_ model: ModelInfo) {
        guard appState.selectedModelId != model.id else { return }

        // Change selection - will need to reload model
        appState.selectedModelId = model.id
        appState.isModelLoaded = false
    }

    private func deleteModel(_ model: ModelInfo) {
        do {
            try ModelCache.deleteModel(model.id)
            appState.refreshDownloadedModels()

            // If we deleted the selected model, auto-select another
            if appState.selectedModelId == model.id {
                if let nextModel = appState.downloadedModelIds.first {
                    appState.selectedModelId = nextModel
                }
                // Mark as not loaded since we deleted the current model
                appState.isModelLoaded = false
            }
        } catch {
            appState.modelError = "Failed to delete model: \(error.localizedDescription)"
        }
    }

    private func downloadModel(_ model: ModelInfo) async {
        appState.downloadingModelId = model.id
        appState.modelDownloadProgress = 0

        do {
            let configuration = Moondream3Loader.configuration(for: model.id)
            _ = try await Moondream3Loader.loadContainer(
                configuration: configuration
            ) { progress in
                Task { @MainActor in
                    appState.modelDownloadProgress = progress.fractionCompleted
                }
            }

            appState.downloadedModelIds.insert(model.id)

            // If no model was selected, select this one
            if !appState.downloadedModelIds.contains(appState.selectedModelId) {
                appState.selectedModelId = model.id
            }
        } catch {
            appState.modelError = "Download failed: \(error.localizedDescription)"
        }

        appState.downloadingModelId = nil
    }
}

/// Row for skill selection
struct SkillRow: View {
    let skill: Skill
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: skill.icon)
                    .font(.title2)
                    .foregroundStyle(isSelected ? .blue : .secondary)
                    .frame(width: 32)

                VStack(alignment: .leading, spacing: 2) {
                    Text(skill.displayName)
                        .font(.body)
                        .foregroundStyle(.primary)

                    Text(skill.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark")
                        .foregroundStyle(.blue)
                        .fontWeight(.semibold)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

/// View that displays current memory info with auto-refresh
struct MemoryInfoView: View {
    @State private var availableMemory: UInt64 = getAvailableMemory()

    // Timer to refresh memory info
    let timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Available")
                Spacer()
                Text(formatBytes(availableMemory))
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(memoryColor)
            }

            // Memory bar
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.secondary.opacity(0.2))

                    RoundedRectangle(cornerRadius: 4)
                        .fill(memoryColor)
                        .frame(width: memoryBarWidth(totalWidth: geo.size.width))
                }
            }
            .frame(height: 8)
        }
        .onReceive(timer) { _ in
            availableMemory = getAvailableMemory()
        }
        .onAppear {
            availableMemory = getAvailableMemory()
        }
    }

    private var memoryColor: Color {
        let gb = Double(availableMemory) / (1024 * 1024 * 1024)
        if gb >= 3.0 {
            return .green
        } else if gb >= 2.0 {
            return .yellow
        } else {
            return .red
        }
    }

    private func memoryBarWidth(totalWidth: CGFloat) -> CGFloat {
        // Assume 8GB total as reference for bar
        let maxMemory: Double = 8 * 1024 * 1024 * 1024
        let fraction = min(Double(availableMemory) / maxMemory, 1.0)
        return totalWidth * fraction
    }
}

#Preview {
    SettingsSheet()
        .environment(AppState())
}
#endif // os(iOS)
