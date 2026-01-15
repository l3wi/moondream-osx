import SwiftUI
import MoondreamKit

/// Visual representation of a detected bounding box
struct BoundingBoxView: View {
    let box: NormalizedBox

    var body: some View {
        ZStack(alignment: .topLeading) {
            // Box outline
            Rectangle()
                .stroke(Color.green, lineWidth: 3)

            // Semi-transparent fill
            Rectangle()
                .fill(Color.green.opacity(0.1))

            // Corner markers
            VStack {
                HStack {
                    CornerMarker()
                    Spacer()
                    CornerMarker()
                        .rotationEffect(.degrees(90))
                }
                Spacer()
                HStack {
                    CornerMarker()
                        .rotationEffect(.degrees(-90))
                    Spacer()
                    CornerMarker()
                        .rotationEffect(.degrees(180))
                }
            }

            // Label if present
            if let label = box.label {
                HStack(spacing: 4) {
                    Text(label)
                        .font(.caption2)
                        .fontWeight(.semibold)

                    if let confidence = box.confidence {
                        Text("\(Int(confidence * 100))%")
                            .font(.caption2)
                            .opacity(0.8)
                    }
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(Color.green.opacity(0.9), in: RoundedRectangle(cornerRadius: 4))
                .offset(x: 4, y: -24)
            }
        }
    }
}

/// Corner marker for bounding box
struct CornerMarker: View {
    var body: some View {
        Path { path in
            path.move(to: CGPoint(x: 0, y: 12))
            path.addLine(to: CGPoint(x: 0, y: 0))
            path.addLine(to: CGPoint(x: 12, y: 0))
        }
        .stroke(Color.green, lineWidth: 4)
        .frame(width: 12, height: 12)
    }
}

#Preview {
    VStack(spacing: 40) {
        BoundingBoxView(box: NormalizedBox(
            xMin: 0, yMin: 0, xMax: 1, yMax: 1
        ))
        .frame(width: 200, height: 150)

        BoundingBoxView(box: NormalizedBox(
            xMin: 0, yMin: 0, xMax: 1, yMax: 1,
            label: "Person",
            confidence: 0.95
        ))
        .frame(width: 200, height: 150)
    }
    .padding(40)
    .background(Color.gray)
}
