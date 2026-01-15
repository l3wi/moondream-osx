import SwiftUI
import MoondreamKit

/// Reusable model row component for Settings
/// Shows model info with select/download/delete capabilities
struct ModelRow: View {
    let model: ModelInfo
    let isSelected: Bool
    let isDownloaded: Bool
    let isDownloading: Bool
    let downloadProgress: Double
    let onSelect: () -> Void
    let onDownload: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            // Selection indicator
            if isDownloaded {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundStyle(isSelected ? .blue : .secondary)
            }

            // Model info
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName)
                    .font(.body)
                    .foregroundStyle(.primary)

                HStack(spacing: 6) {
                    Text(model.quantization)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text("•")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text(model.sizeDisplay)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                // Compatibility warning for iOS
                if !model.isCompatibleWithiOS {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.caption)
                        Text("Not compatible with this device")
                            .font(.caption)
                    }
                    .foregroundStyle(.orange)
                }
            }

            Spacer()

            // Action / Status
            if isDownloading {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.7)
                    Text("\(Int(downloadProgress * 100))%")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 36, alignment: .trailing)
                }
            } else if isDownloaded {
                // Delete button for downloaded models
                Button {
                    onDelete()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            } else {
                // Download button for undownloaded models
                if model.isCompatibleWithiOS {
                    Button {
                        onDownload()
                    } label: {
                        Image(systemName: "arrow.down.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.blue)
                    }
                    .buttonStyle(.plain)
                } else {
                    // Incompatible - show disabled state
                    Image(systemName: "xmark.circle")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if isDownloaded {
                onSelect()
            }
        }
    }
}

/// Simplified model row for download-only contexts
struct ModelDownloadOnlyRow: View {
    let model: ModelInfo
    let isDownloading: Bool
    let downloadProgress: Double
    let onDownload: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName)
                    .font(.body)
                    .foregroundStyle(.primary)

                HStack(spacing: 6) {
                    Text(model.quantization)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text("•")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text(model.sizeDisplay)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                // Compatibility warning for iOS
                if !model.isCompatibleWithiOS {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.caption)
                        Text("Not compatible with this device")
                            .font(.caption)
                    }
                    .foregroundStyle(.orange)
                }
            }

            Spacer()

            if isDownloading {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.7)
                    Text("\(Int(downloadProgress * 100))%")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 36, alignment: .trailing)
                }
            } else if model.isCompatibleWithiOS {
                Button {
                    onDownload()
                } label: {
                    Image(systemName: "arrow.down.circle")
                        .font(.title2)
                        .foregroundStyle(.blue)
                }
                .buttonStyle(.plain)
            } else {
                // Incompatible - show disabled state
                Image(systemName: "xmark.circle")
                    .font(.title2)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
