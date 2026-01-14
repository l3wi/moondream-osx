import SwiftUI

/// Input field for query text or object name
struct InputField: View {
    let skill: Skill
    @Binding var queryText: String
    @Binding var objectText: String
    @Binding var captionLength: CaptionLength
    var onSubmit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            switch skill {
            case .query:
                Text("Question")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)

                TextField("What would you like to know?", text: $queryText)
                    .textFieldStyle(.plain)
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.background)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1)
                            )
                    )
                    .onSubmit(onSubmit)

            case .caption:
                Text("Caption Length")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)

                Picker("Length", selection: $captionLength) {
                    ForEach(CaptionLength.allCases) { length in
                        Text(length.displayName).tag(length)
                    }
                }
                .pickerStyle(.segmented)

            case .point:
                Text("Object to Point At")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)

                TextField("e.g., the red button, person's face", text: $objectText)
                    .textFieldStyle(.plain)
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.background)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1)
                            )
                    )
                    .onSubmit(onSubmit)

            case .detect:
                Text("Object to Detect")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)

                TextField("e.g., cars, people, dogs", text: $objectText)
                    .textFieldStyle(.plain)
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.background)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1)
                            )
                    )
                    .onSubmit(onSubmit)
            }
        }
    }
}
