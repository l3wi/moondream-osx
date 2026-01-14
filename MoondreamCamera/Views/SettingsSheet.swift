import SwiftUI

#if os(iOS)
/// Settings sheet for skill selection and options
struct SettingsSheet: View {
    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        @Bindable var state = appState

        NavigationStack {
            Form {
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

#Preview {
    SettingsSheet()
        .environment(AppState())
}
#endif // os(iOS)
