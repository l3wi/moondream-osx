import SwiftUI

/// Primary action button for running inference
struct RunButton: View {
    let isEnabled: Bool
    let isLoading: Bool    // Model loading state
    let isProcessing: Bool // Inference processing state
    let action: () -> Void

    private var buttonText: String {
        if isLoading { return "Loading..." }
        if isProcessing { return "Running..." }
        return "Run"
    }

    private var isWorking: Bool {
        isLoading || isProcessing
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isWorking {
                    ProgressView()
                        .scaleEffect(0.7)
                        .frame(width: 16, height: 16)
                } else {
                    Image(systemName: "play.fill")
                        .font(.system(size: 12))
                }

                Text(buttonText)
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .disabled(!isEnabled || isWorking)
    }
}

#Preview {
    VStack(spacing: 20) {
        RunButton(isEnabled: true, isLoading: false, isProcessing: false, action: {})
        RunButton(isEnabled: true, isLoading: true, isProcessing: false, action: {})
        RunButton(isEnabled: true, isLoading: false, isProcessing: true, action: {})
        RunButton(isEnabled: false, isLoading: false, isProcessing: false, action: {})
    }
    .padding()
    .frame(width: 300)
}
