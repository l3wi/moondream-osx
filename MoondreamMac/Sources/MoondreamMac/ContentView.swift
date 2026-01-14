import SwiftUI
import AppKit
import CoreImage
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var service: MoondreamService

    @State private var droppedImage: NSImage?
    @State private var selectedSkill: Skill = .caption
    @State private var queryText: String = "What is in this image?"
    @State private var objectText: String = "painting"
    @State private var captionLength: CaptionLength = .normal
    @State private var result: String = ""
    @State private var isProcessing: Bool = false
    @State private var isTargeted: Bool = false

    var body: some View {
        HSplitView {
            // Left panel - Image drop zone
            VStack(spacing: 16) {
                Text("Drop Image Here")
                    .font(.headline)

                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(
                            isTargeted ? Color.accentColor : Color.secondary,
                            style: StrokeStyle(lineWidth: 2, dash: [8])
                        )
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(isTargeted ? Color.accentColor.opacity(0.1) : Color.clear)
                        )

                    if let image = droppedImage {
                        Image(nsImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .padding(8)
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "photo.on.rectangle.angled")
                                .font(.system(size: 48))
                                .foregroundColor(.secondary)
                            Text("Drag & drop an image")
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .frame(minHeight: 300)
                .onDrop(of: [.image, .fileURL], isTargeted: $isTargeted) { providers in
                    handleDrop(providers: providers)
                }

                // Open file button
                Button("Open Image...") {
                    openFile()
                }
            }
            .padding()
            .frame(minWidth: 350)

            // Right panel - Controls and results
            VStack(alignment: .leading, spacing: 16) {
                // Model loading status
                if !service.isLoaded {
                    VStack(spacing: 8) {
                        if service.isLoading {
                            ProgressView(value: service.loadingProgress)
                            Text("Loading model: \(Int(service.loadingProgress * 100))%")
                                .font(.caption)
                        } else if let error = service.loadError {
                            Text("Error: \(error)")
                                .foregroundColor(.red)
                            Button("Retry") {
                                Task { try? await service.loadModel() }
                            }
                        } else {
                            Button("Load Model") {
                                Task { try? await service.loadModel() }
                            }
                        }
                    }
                    .padding()
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.1)))
                }

                // Skill picker
                Text("Skill")
                    .font(.headline)

                Picker("Skill", selection: $selectedSkill) {
                    ForEach(Skill.allCases) { skill in
                        Text(skill.displayName).tag(skill)
                    }
                }
                .pickerStyle(.segmented)

                // Skill-specific input
                Group {
                    switch selectedSkill {
                    case .query:
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Question")
                                .font(.subheadline)
                            TextField("Enter your question", text: $queryText)
                                .textFieldStyle(.roundedBorder)
                        }

                    case .caption:
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Length")
                                .font(.subheadline)
                            Picker("Length", selection: $captionLength) {
                                ForEach(CaptionLength.allCases) { length in
                                    Text(length.displayName).tag(length)
                                }
                            }
                            .pickerStyle(.segmented)
                        }

                    case .point, .detect:
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Object to find")
                                .font(.subheadline)
                            TextField("Enter object name", text: $objectText)
                                .textFieldStyle(.roundedBorder)
                        }
                    }
                }

                // Run button
                Button(action: runInference) {
                    HStack {
                        if isProcessing {
                            ProgressView()
                                .scaleEffect(0.7)
                        }
                        Text(isProcessing ? "Processing..." : "Run")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!service.isLoaded || droppedImage == nil || isProcessing)

                // Results
                Text("Result")
                    .font(.headline)

                ScrollView {
                    Text(result.isEmpty ? "Results will appear here" : result)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(maxHeight: .infinity)
                .padding(8)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.1)))

                Spacer()
            }
            .padding()
            .frame(minWidth: 300)
        }
        .frame(minWidth: 700, minHeight: 500)
        .task {
            // Auto-load model on launch
            if !service.isLoaded && !service.isLoading {
                try? await service.loadModel()
            }
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }

        // Try loading as file URL first (more reliable)
        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier) { data, _ in
                guard let data = data as? Data,
                      let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
                let loadedImage = NSImage(contentsOf: url)
                Task { @MainActor in
                    self.droppedImage = loadedImage
                }
            }
            return true
        }

        // Try loading as image data
        if provider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                guard let data = data, let loadedImage = NSImage(data: data) else { return }
                Task { @MainActor in
                    self.droppedImage = loadedImage
                }
            }
            return true
        }

        return false
    }

    private func openFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .png, .jpeg, .gif, .webP]
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            droppedImage = NSImage(contentsOf: url)
        }
    }

    private func runInference() {
        guard let nsImage = droppedImage else { return }

        // Convert NSImage to CIImage
        guard let tiffData = nsImage.tiffRepresentation,
              let ciImage = CIImage(data: tiffData) else {
            result = "Error: Could not convert image"
            return
        }

        isProcessing = true
        result = ""

        Task {
            do {
                let output: String
                switch selectedSkill {
                case .query:
                    let queryResult = try await service.query(image: ciImage, question: queryText)
                    output = queryResult.answer
                case .caption:
                    let captionResult = try await service.caption(image: ciImage, length: captionLength)
                    output = captionResult.caption
                case .point:
                    let pointResult = try await service.point(image: ciImage, object: objectText)
                    output = "Found \(pointResult.points.count) point(s)\nRaw: \(pointResult.rawOutput)"
                case .detect:
                    let detectResult = try await service.detect(image: ciImage, object: objectText)
                    output = "Detected \(detectResult.boxes.count) object(s)\nRaw: \(detectResult.rawOutput)"
                }
                await MainActor.run {
                    result = output
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    result = "Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(MoondreamService.shared)
}
