import SwiftUI
import AppKit
import CoreImage

// CLI runner as a separate class to avoid escaping closure issues
@MainActor
class CLIRunner {
    static func run(imagePath: String, skill: String, query: String) async {
        let service = MoondreamService.shared

        print("MoondreamMac CLI")
        print("================")
        print("Image: \(imagePath)")
        print("Skill: \(skill)")
        if skill == "query" || skill == "point" || skill == "detect" {
            print("Query: \(query)")
        }
        print("")

        // Load image
        guard let nsImage = NSImage(contentsOfFile: imagePath) else {
            print("ERROR: Could not load image at \(imagePath)")
            return
        }

        guard let tiffData = nsImage.tiffRepresentation,
              let ciImage = CIImage(data: tiffData) else {
            print("ERROR: Could not convert image to CIImage")
            return
        }

        print("Image loaded: \(Int(nsImage.size.width))x\(Int(nsImage.size.height))")
        print("")

        // Load model
        print("Loading model...")
        do {
            try await service.loadModel()
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

@main
struct MoondreamMacApp: App {
    @StateObject private var service = MoondreamService.shared

    init() {
        // Check for CLI arguments
        let args = CommandLine.arguments
        if args.count > 1 {
            // CLI mode: run inference and exit
            let imagePath = args[1]
            let skill = args.count > 2 ? args[2] : "caption"
            let query = args.count > 3 ? args[3] : "What is in this image?"

            Task { @MainActor in
                await CLIRunner.run(imagePath: imagePath, skill: skill, query: query)
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
        }
    }
}
