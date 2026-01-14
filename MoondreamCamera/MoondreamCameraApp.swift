import SwiftUI
import UIKit

#if os(iOS)
@main
struct MoondreamCameraApp: App {
    @State private var appState = AppState()

    init() {
        // Keep screen on in DEBUG builds for easier testing
        #if DEBUG
        UIApplication.shared.isIdleTimerDisabled = true
        #endif
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appState)
        }
    }
}
#endif
