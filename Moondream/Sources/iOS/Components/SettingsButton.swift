import SwiftUI

/// Settings cog button with Liquid Glass effect
struct SettingsButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: "gearshape.fill")
                .font(.title2)
                .foregroundStyle(.white)
                .padding(14)
        }
        .glassEffect(.regular, in: .circle)
    }
}

#Preview {
    ZStack {
        Color.black
        SettingsButton {}
    }
}
