// Copyright 2024 Moondream AI
// Token constants for Moondream3 model

import Foundation

/// Centralized token constants for Moondream3 model
/// These match the tokenizer configuration from moondream/starmie-v1
public enum TokenConstants {
    // MARK: - Special Tokens

    /// Beginning of sequence token (also used as EOS in this model)
    public static let bos: Int = 0

    /// End of sequence token (same as BOS in Moondream3)
    public static let eos: Int = 0

    /// Answer token - marks end of reasoning section
    public static let answer: Int = 3

    /// Thinking token - marks start of reasoning
    public static let thinking: Int = 4

    /// Coordinate token - used for point() and detect() skills
    public static let coord: Int = 5

    // MARK: - Caption Templates

    /// Token templates for caption generation
    public enum Caption {
        /// Short caption: concise 1-2 sentence description
        public static let short: [Int] = [1, 32708, 2, 12492, 3]

        /// Normal caption: standard description
        public static let normal: [Int] = [1, 32708, 2, 6382, 3]

        /// Long caption: detailed description
        public static let long: [Int] = [1, 32708, 2, 4059, 3]

        /// Get template for caption length
        public static func template(for length: String) -> [Int] {
            switch length {
            case "short": return short
            case "long": return long
            default: return normal
            }
        }
    }

    // MARK: - Query Templates

    /// Token templates for query (visual Q&A)
    public enum Query {
        /// Prefix tokens before question
        public static let prefix: [Int] = [1, 15381, 2]

        /// Suffix tokens after question
        public static let suffix: [Int] = [3]
    }

    // MARK: - Point Templates

    /// Token templates for point (locate object)
    public enum Point {
        /// Prefix tokens before object name
        public static let prefix: [Int] = [1, 2581, 2]

        /// Suffix tokens after object name
        public static let suffix: [Int] = [3]
    }

    // MARK: - Detect Templates

    /// Token templates for detect (bounding boxes)
    public enum Detect {
        /// Prefix tokens before object name
        public static let prefix: [Int] = [1, 7235, 476, 2]

        /// Suffix tokens after object name
        public static let suffix: [Int] = [3]
    }
}
