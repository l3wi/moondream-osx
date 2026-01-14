import SwiftUI
import CoreImage

#if os(iOS)
import UIKit

/// Displays the frozen/captured image
struct FrozenImageView: View {
    let image: CIImage

    @State private var uiImage: UIImage?

    var body: some View {
        GeometryReader { geometry in
            if let uiImage {
                Image(uiImage: uiImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .clipped()
            } else {
                Color.black
                    .onAppear {
                        convertImage()
                    }
            }
        }
    }

    private func convertImage() {
        let context = CIContext()
        if let cgImage = context.createCGImage(image, from: image.extent) {
            uiImage = UIImage(cgImage: cgImage)
        }
    }
}
#endif
