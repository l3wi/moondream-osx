import SwiftUI

/// Primary action button for running inference
struct RunButton: View {
    let isEnabled: Bool
    let isProcessing: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isProcessing {
                    ProgressView()
                        .scaleEffect(0.7)
                        .frame(width: 16, height: 16)
                } else {
                    Image(systemName: "play.fill")
                        .font(.system(size: 12))
                }

                Text(isProcessing ? "Processing..." : "Run")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .disabled(!isEnabled || isProcessing)
    }
}

#Preview {
    VStack(spacing: 20) {
        RunButton(isEnabled: true, isProcessing: false, action: {})
        RunButton(isEnabled: true, isProcessing: true, action: {})
        RunButton(isEnabled: false, isProcessing: false, action: {})
    }
    .padding()
    .frame(width: 300)
}
