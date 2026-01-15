import SwiftUI
import MoondreamKit

#if os(iOS)
/// Floating Liquid Glass input modal for skills requiring text input
/// Displayed as a centered overlay over the frozen camera image
struct InputModal: View {
    @Environment(AppState.self) private var appState
    let skill: Skill
    let onSubmit: () -> Void
    let onDismiss: () -> Void

    @FocusState private var isInputFocused: Bool
    @State private var inputText: String = ""

    var body: some View {
        ZStack {
            // Semi-transparent background - tap to dismiss
            Color.black.opacity(0.4)
                .ignoresSafeArea()
                .onTapGesture {
                    isInputFocused = false
                    onDismiss()
                }

            // Centered modal card
            VStack(spacing: 20) {
                // Title and subtitle
                VStack(spacing: 6) {
                    Text(modalTitle)
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundStyle(.primary)

                    Text(modalSubtitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                // Text input - single line that expands
                TextField(placeholderText, text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.body)
                    .foregroundStyle(.primary)
                    .lineLimit(1...4)
                    .focused($isInputFocused)
                    .submitLabel(.done)
                    .onSubmit {
                        submit()
                    }
                    .padding(14)
                    .background {
                        RoundedRectangle(cornerRadius: 10)
                            .fill(.ultraThinMaterial)
                    }

                // Run button
                Button {
                    submit()
                } label: {
                    Text("Run")
                        .font(.headline)
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                }
                .glassEffect(.regular.interactive(), in: .capsule)
                .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty)
                .opacity(inputText.trimmingCharacters(in: .whitespaces).isEmpty ? 0.5 : 1.0)
            }
            .padding(20)
            .frame(maxWidth: 320)
            .glassEffect(.regular, in: .rect(cornerRadius: 20))
            .padding(.horizontal, 24)
        }
        .onAppear {
            // Pre-fill with existing value if available
            if skill == .query {
                inputText = appState.queryText
            } else {
                inputText = appState.targetObject
            }
            // Auto-focus immediately to launch keyboard
            isInputFocused = true
        }
    }

    // MARK: - Skill-Specific Content

    private var modalTitle: String {
        switch skill {
        case .query: "Ask a Question"
        case .point: "Find Object"
        case .detect: "Detect Objects"
        case .caption: "Caption"
        }
    }

    private var modalSubtitle: String {
        switch skill {
        case .query: "What would you like to know about this image?"
        case .point: "Enter the object you want to locate"
        case .detect: "Enter the type of objects to detect"
        case .caption: ""
        }
    }

    private var placeholderText: String {
        switch skill {
        case .query: "What is in this image?"
        case .point: "e.g., cat, red button"
        case .detect: "e.g., cars, people"
        case .caption: ""
        }
    }

    // MARK: - Actions

    private func submit() {
        let trimmed = inputText.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }

        isInputFocused = false

        // Write to correct appState property
        if skill == .query {
            appState.queryText = trimmed
        } else {
            appState.targetObject = trimmed
        }

        // Set processing BEFORE hiding modal to prevent reset
        appState.isProcessing = true
        appState.showInputModal = false
        onSubmit()
    }
}

#Preview {
    ZStack {
        Color.black
        InputModal(skill: .query, onSubmit: {}, onDismiss: {})
            .environment(AppState())
    }
}
#endif
