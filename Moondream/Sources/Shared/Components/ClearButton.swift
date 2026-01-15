import SwiftUI

/// Clear button that replaces capture button when image is frozen - shared between iOS and macOS
struct ClearButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                // Outer ring
                Circle()
                    .stroke(.white, lineWidth: 4)
                    .frame(width: 72, height: 72)

                // Inner circle with X
                Circle()
                    .fill(.white.opacity(0.2))
                    .frame(width: 58, height: 58)

                Image(systemName: "xmark")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundStyle(.white)
            }
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    ZStack {
        Color.black
        ClearButton {}
    }
}
