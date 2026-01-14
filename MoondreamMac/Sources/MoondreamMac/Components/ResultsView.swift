import SwiftUI

/// View displaying inference results
struct ResultsView: View {
    let result: String
    let isProcessing: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Results")
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundStyle(.secondary)

            ScrollView {
                if isProcessing {
                    HStack(spacing: 12) {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text("Processing...")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                } else if result.isEmpty {
                    Text("Results will appear here")
                        .foregroundStyle(.tertiary)
                        .italic()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                } else {
                    Text(result)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(.background)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1)
                    )
            )
        }
    }
}

#Preview {
    VStack {
        ResultsView(result: "", isProcessing: false)
        ResultsView(result: "", isProcessing: true)
        ResultsView(result: "The image shows a beautiful sunset over the ocean.", isProcessing: false)
    }
    .padding()
    .frame(width: 300)
}
