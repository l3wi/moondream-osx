import SwiftUI
import MoondreamKit

#if os(iOS)
import UIKit

/// Sheet displaying inference results
struct ResultsSheet: View {
    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss
    @StateObject private var moondreamService = MoondreamService.shared

    @FocusState private var isQueryFocused: Bool

    var body: some View {
        @Bindable var state = appState

        NavigationStack {
            VStack(spacing: 0) {
                // Query input (only for query skill before inference)
                if appState.selectedSkill == .query && appState.currentResult == nil {
                    QueryInputSection(
                        queryText: $state.queryText,
                        isQueryFocused: $isQueryFocused,
                        onSubmit: submitQuery
                    )
                }

                // Results content
                if let result = appState.currentResult {
                    ResultsContent(result: result, debugMode: appState.debugMode)
                } else if appState.selectedSkill != .query {
                    // Waiting for results
                    VStack(spacing: 16) {
                        ProgressView()
                        Text("Processing...")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .navigationTitle(navigationTitle)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Close") {
                        dismiss()
                    }
                }

                if appState.selectedSkill == .query && appState.currentResult == nil {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Ask") {
                            submitQuery()
                        }
                        .disabled(appState.queryText.trimmingCharacters(in: .whitespaces).isEmpty)
                        .fontWeight(.semibold)
                    }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        .interactiveDismissDisabled(appState.isProcessing)
        .onAppear {
            if appState.selectedSkill == .query && appState.currentResult == nil {
                isQueryFocused = true
            }
        }
    }

    private var navigationTitle: String {
        switch appState.selectedSkill {
        case .query: "Query"
        case .caption: "Caption"
        case .point: "Points"
        case .detect: "Detection"
        }
    }

    private func submitQuery() {
        guard !appState.queryText.trimmingCharacters(in: .whitespaces).isEmpty,
              let image = appState.capturedImage else {
            return
        }

        isQueryFocused = false

        Task {
            appState.isProcessing = true

            do {
                let result = try await moondreamService.query(
                    image: image,
                    question: appState.queryText
                )
                appState.currentResult = .query(result)
            } catch {
                appState.modelError = error.localizedDescription
            }

            appState.isProcessing = false
        }
    }
}

/// Query input section
struct QueryInputSection: View {
    @Binding var queryText: String
    var isQueryFocused: FocusState<Bool>.Binding
    let onSubmit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Ask a question about the image:")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            TextField("What is in this image?", text: $queryText, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(3...6)
                .focused(isQueryFocused)
                .submitLabel(.send)
                .onSubmit(onSubmit)
        }
        .padding()
        .background(Color(uiColor: .systemGroupedBackground))
    }
}

/// Results content based on result type
struct ResultsContent: View {
    let result: MoondreamResult
    let debugMode: Bool

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Main result
                switch result {
                case .query(let queryResult):
                    ResultSection(title: "Answer", content: queryResult.answer)

                    if let reasoning = queryResult.reasoning, !reasoning.isEmpty {
                        ResultSection(title: "Reasoning", content: reasoning)
                    }

                case .caption(let captionResult):
                    ResultSection(title: "Caption", content: captionResult.caption)

                case .point(let pointResult):
                    ResultSection(
                        title: "Points Found",
                        content: "\(pointResult.points.count) point(s) located"
                    )

                    ForEach(Array(pointResult.points.enumerated()), id: \.element.id) { index, point in
                        PointDetail(index: index + 1, point: point)
                    }

                case .detect(let detectResult):
                    ResultSection(
                        title: "Objects Detected",
                        content: "\(detectResult.boxes.count) object(s) found"
                    )

                    ForEach(Array(detectResult.boxes.enumerated()), id: \.element.id) { index, box in
                        BoxDetail(index: index + 1, box: box)
                    }
                }

                // Debug output
                if debugMode {
                    DebugOutputSection(result: result)
                }
            }
            .padding()
        }
    }
}

/// Section for displaying result text
struct ResultSection: View {
    let title: String
    let content: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .foregroundStyle(.secondary)

            Text(content)
                .font(.body)
                .textSelection(.enabled)
        }
    }
}

/// Detail for a single point
struct PointDetail: View {
    let index: Int
    let point: NormalizedPoint

    var body: some View {
        HStack {
            Text("Point \(index)")
                .font(.subheadline)
                .fontWeight(.medium)

            Spacer()

            Text("(\(String(format: "%.2f", point.x)), \(String(format: "%.2f", point.y)))")
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospaced()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .cornerRadius(8)
    }
}

/// Detail for a single bounding box
struct BoxDetail: View {
    let index: Int
    let box: NormalizedBox

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Object \(index)")
                    .font(.subheadline)
                    .fontWeight(.medium)

                Spacer()

                if let confidence = box.confidence {
                    Text("\(Int(confidence * 100))%")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Text("[\(String(format: "%.2f", box.xMin)), \(String(format: "%.2f", box.yMin))] → [\(String(format: "%.2f", box.xMax)), \(String(format: "%.2f", box.yMax))]")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .monospaced()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .cornerRadius(8)
    }
}

/// Debug output section
struct DebugOutputSection: View {
    let result: MoondreamResult

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "ladybug")
                Text("Debug Output")
            }
            .font(.headline)
            .foregroundStyle(.orange)

            ScrollView(.horizontal, showsIndicators: true) {
                Text(rawOutput)
                    .font(.caption)
                    .monospaced()
                    .textSelection(.enabled)
            }
            .padding(8)
            .background(Color(uiColor: .tertiarySystemGroupedBackground))
            .cornerRadius(8)
        }
    }

    private var rawOutput: String {
        switch result {
        case .query(let r): r.rawOutput
        case .caption(let r): r.rawOutput
        case .point(let r): r.rawOutput
        case .detect(let r): r.rawOutput
        }
    }
}

#Preview {
    let state = AppState()
    state.currentResult = .caption(CaptionResult(caption: "A beautiful sunset over the ocean with waves crashing on the shore.", rawOutput: "A beautiful sunset..."))
    return ResultsSheet()
        .environment(state)
}
#endif // os(iOS)
