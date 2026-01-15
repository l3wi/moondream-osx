import SwiftUI

/// Settings cog button
struct SettingsButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: "gearshape.fill")
                .font(.title2)
                .foregroundStyle(.white)
                .padding(14)
                .background(.ultraThinMaterial, in: Circle())
        }
    }
}

#Preview {
    ZStack {
        Color.black
        SettingsButton {}
    }
}
