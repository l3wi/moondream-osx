import SwiftUI

/// Visual marker for a detected point
struct PointMarkerView: View {
    let point: NormalizedPoint

    @State private var isAnimating = false

    var body: some View {
        ZStack {
            // Pulse animation ring
            Circle()
                .stroke(Color.red.opacity(0.5), lineWidth: 2)
                .frame(width: 44, height: 44)
                .scaleEffect(isAnimating ? 1.3 : 1.0)
                .opacity(isAnimating ? 0 : 0.8)

            // Outer ring
            Circle()
                .stroke(Color.red, lineWidth: 3)
                .frame(width: 32, height: 32)

            // Inner fill
            Circle()
                .fill(Color.red.opacity(0.3))
                .frame(width: 24, height: 24)

            // Center dot
            Circle()
                .fill(Color.white)
                .frame(width: 8, height: 8)
                .shadow(color: .black.opacity(0.3), radius: 2)

            // Label if present
            if let label = point.label {
                Text(label)
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.red.opacity(0.9), in: Capsule())
                    .offset(y: 28)
            }
        }
        .onAppear {
            withAnimation(.easeOut(duration: 1.0).repeatForever(autoreverses: false)) {
                isAnimating = true
            }
        }
    }
}

#Preview {
    VStack(spacing: 40) {
        PointMarkerView(point: NormalizedPoint(x: 0.5, y: 0.5))

        PointMarkerView(point: NormalizedPoint(x: 0.5, y: 0.5, label: "Person"))
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(Color.gray)
}
