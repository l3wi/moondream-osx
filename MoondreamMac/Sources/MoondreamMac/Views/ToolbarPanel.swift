import SwiftUI

/// Right-side toolbar panel with skill selection and controls
struct ToolbarPanel: View {
    @EnvironmentObject var service: MoondreamService
    @Binding var selectedSkill: Skill
    @Binding var queryText: String
    @Binding var objectText: String
    @Binding var captionLength: CaptionLength
    @Binding var result: String
    @Binding var isProcessing: Bool
    var onRun: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Model loading status
            if !service.isLoaded {
                modelStatusView
                    .padding()
            }

            // Skill tabs
            SkillTabBar(selection: $selectedSkill)
                .padding(.horizontal, 16)
                .padding(.top, 16)
                .padding(.bottom, 12)

            Divider()
                .padding(.horizontal, 16)

            // Input section
            InputField(
                skill: selectedSkill,
                queryText: $queryText,
                objectText: $objectText,
                captionLength: $captionLength,
                onSubmit: onRun
            )
            .padding(16)

            Divider()
                .padding(.horizontal, 16)

            // Results section
            ResultsView(result: result, isProcessing: isProcessing)
                .padding(16)
                .frame(maxHeight: .infinity)

            Divider()
                .padding(.horizontal, 16)

            // Run button
            RunButton(
                isEnabled: service.isLoaded,
                isProcessing: isProcessing,
                action: onRun
            )
            .padding(16)
        }
        .background(.ultraThinMaterial)
    }

    private var modelStatusView: some View {
        VStack(spacing: 8) {
            if service.isLoading {
                ProgressView(value: service.loadingProgress)
                    .progressViewStyle(.linear)
                Text("Loading model: \(Int(service.loadingProgress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if let error = service.loadError {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Button("Retry") {
                    Task { try? await service.loadModel() }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Button("Load Model") {
                    Task { try? await service.loadModel() }
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.secondary.opacity(0.1))
        )
    }
}
