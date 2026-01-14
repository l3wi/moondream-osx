import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Empty state view with drop zone for loading images
struct EmptyStateView: View {
    @Binding var droppedImage: NSImage?
    @State private var isTargeted = false

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Centered 4:3 drop zone
                dropZone
                    .aspectRatio(4/3, contentMode: .fit)
                    .frame(maxWidth: min(geo.size.width * 0.8, 600))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(.regularMaterial)
        .onDrop(of: [.image, .fileURL], isTargeted: $isTargeted) { providers in
            handleDrop(providers: providers)
        }
    }

    private var dropZone: some View {
        ZStack {
            // Drop zone border only - no fill
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(
                    isTargeted ? Color.accentColor : Color.secondary.opacity(0.3),
                    style: StrokeStyle(lineWidth: 1.5, dash: isTargeted ? [] : [6])
                )

            // Content
            VStack(spacing: 20) {
                Image(systemName: "photo.on.rectangle.angled")
                    .font(.system(size: 64, weight: .thin))
                    .foregroundStyle(.secondary)

                VStack(spacing: 8) {
                    Text("Drop image here")
                        .font(.title2)
                        .fontWeight(.medium)

                    Text("or click to open file picker")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Button("Choose Image...") {
                    openFilePicker()
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }
            .padding(40)
        }
        .contentShape(Rectangle())
        .onTapGesture {
            openFilePicker()
        }
    }

    private func openFilePicker() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = ImageConverter.supportedContentTypes
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Select an image to analyze"

        if panel.runModal() == .OK, let url = panel.url {
            droppedImage = NSImage(contentsOf: url)
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }

        // Try loading as file URL first
        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier) { data, _ in
                guard let data = data as? Data,
                      let url = URL(dataRepresentation: data, relativeTo: nil),
                      let image = NSImage(contentsOf: url) else { return }
                Task { @MainActor in
                    self.droppedImage = image
                }
            }
            return true
        }

        // Try loading as image data
        if provider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                guard let data = data, let image = NSImage(data: data) else { return }
                Task { @MainActor in
                    self.droppedImage = image
                }
            }
            return true
        }

        return false
    }
}
