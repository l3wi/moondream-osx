import SwiftUI
import AppKit
import CoreImage
import MoondreamKit

/// Helper to ensure app becomes frontmost and stays interactive
class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Set activation policy to regular app (not accessory)
        NSApp.setActivationPolicy(.regular)

        // Make sure we're the active app
        NSApp.activate(ignoringOtherApps: true)

        // Make the main window key
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            if let window = NSApp.windows.first {
                window.makeKeyAndOrderFront(nil)
                window.acceptsMouseMovedEvents = true
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        // Ensure window is key when app becomes active
        NSApp.windows.first?.makeKeyAndOrderFront(nil)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            NSApp.windows.first?.makeKeyAndOrderFront(nil)
        }
        return true
    }
}

// CLI runner as a separate class to avoid escaping closure issues
@MainActor
class CLIRunner {
    static func run(imagePath: String, skill: String, query: String, modelId: String?) async {
        let service = MoondreamService.shared

        print("MoondreamMac CLI")
        print("================")
        print("Image: \(imagePath)")
        print("Skill: \(skill)")
        if skill == "query" || skill == "point" || skill == "detect" {
            print("Query: \(query)")
        }
        if let modelId = modelId {
            print("Model: \(modelId)")
        }
        print("")

        // Load image using ImageConverter
        let ciImage: CIImage
        do {
            ciImage = try ImageConverter.loadImage(from: URL(fileURLWithPath: imagePath))
        } catch {
            print("ERROR: \(error.localizedDescription)")
            return
        }

        let imageSize = ciImage.extent.size
        print("Image loaded: \(Int(imageSize.width))x\(Int(imageSize.height))")
        print("")

        // Load model
        print("Loading model...")
        do {
            try await service.loadModel(modelId: modelId)
            print("Model loaded successfully!")
            print("")
        } catch {
            print("ERROR loading model: \(error)")
            return
        }

        // Run inference
        print("Running inference...")
        do {
            let result: String
            switch skill.lowercased() {
            case "caption", "caption:short":
                let captionResult = try await service.caption(image: ciImage, length: .short)
                result = captionResult.caption
            case "caption:normal":
                let captionResult = try await service.caption(image: ciImage, length: .normal)
                result = captionResult.caption
            case "caption:long":
                let captionResult = try await service.caption(image: ciImage, length: .long)
                result = captionResult.caption
            case "query":
                let queryResult = try await service.query(image: ciImage, question: query)
                result = queryResult.answer
            case "point":
                let pointResult = try await service.point(image: ciImage, object: query)
                result = "Points: \(pointResult.points.count)\nRaw: \(pointResult.rawOutput)"
            case "detect":
                let detectResult = try await service.detect(image: ciImage, object: query)
                result = "Boxes: \(detectResult.boxes.count)\nRaw: \(detectResult.rawOutput)"
            default:
                let captionResult = try await service.caption(image: ciImage, length: .normal)
                result = captionResult.caption
            }

            print("RESULT:")
            print("-------")
            print(result)
            print("")
        } catch {
            print("ERROR during inference: \(error)")
        }
    }
}

/// Configures window for clean appearance with transparent background
struct GlassWindowConfig: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            if let window = view.window {
                window.titlebarAppearsTransparent = true
                window.titleVisibility = .hidden
                window.styleMask.insert(.fullSizeContentView)
                window.isMovableByWindowBackground = true
                window.backgroundColor = .clear
                window.isOpaque = false
            }
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

@main
struct MoondreamMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var service = MoondreamService.shared

    init() {
        // Check for CLI arguments (filter out system debug flags like -NSDocumentRevisionsDebugMode)
        let args = CommandLine.arguments.filter { arg in
            !arg.hasPrefix("-NS") && !arg.hasPrefix("-Apple") && arg != "YES" && arg != "NO"
        }

        if args.count > 1 {
            // CLI mode: run inference and exit
            // Parse --model argument
            var modelId: String? = nil
            var positionalArgs: [String] = []

            var i = 1
            while i < args.count {
                if args[i] == "--model" && i + 1 < args.count {
                    modelId = args[i + 1]
                    i += 2
                } else {
                    positionalArgs.append(args[i])
                    i += 1
                }
            }

            guard !positionalArgs.isEmpty else {
                print("Usage: MoondreamMac <image> [skill] [query] [--model <model-id>]")
                print("Models: moondream/md3p-int4, lewi/md3p-int4-smol, moondream/moondream3-preview")
                exit(1)
            }

            let imagePath = positionalArgs[0]
            let skill = positionalArgs.count > 1 ? positionalArgs[1] : "caption"
            let query = positionalArgs.count > 2 ? positionalArgs[2] : "What is in this image?"

            Task { @MainActor in
                await CLIRunner.run(imagePath: imagePath, skill: skill, query: query, modelId: modelId)
                exit(0)
            }
            // Keep running until task completes
            RunLoop.main.run()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(service)
                .background(GlassWindowConfig())
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1000, height: 700)
    }
}
