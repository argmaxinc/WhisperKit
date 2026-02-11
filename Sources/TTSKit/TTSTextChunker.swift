//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import Tokenizers

/// Strategy for splitting long text into chunks.
@frozen
public enum TTSChunkingStrategy: String, Codable, CaseIterable, Sendable {
    /// No chunking, generate the full text in a single pass.
    case none
    /// Split at sentence boundaries using TTSTextChunker.
    case sentence
}

/// Splits long text into sentence-bounded chunks suitable for chunked TTS generation.
///
/// Uses natural sentence boundaries (. ! ? and newlines) to avoid splitting mid-sentence,
/// with a fallback to clause boundaries (, ; :) and then word boundaries for very long sentences.
///
/// Algorithm: tokenize the full text, decode windows of `targetChunkSize` tokens,
/// find the last natural boundary in each decoded window, then re-tokenize the accepted
/// chunk to advance by the exact token count - no character estimation needed.
///
/// `defaultTargetChunkSize` / `defaultMinChunkSize` are the single source of truth for
/// these defaults - all other call sites (TTSGenerationOptions, SpeakAX) reference them.
public struct TTSTextChunker {
    /// Default target chunk size (tokens).
    public static let defaultTargetChunkSize: Int = 50

    /// Default minimum chunk size (tokens).
    public static let defaultMinChunkSize: Int = 10

    /// Target chunk size in tokens. Actual chunks may be shorter (sentence boundary)
    /// or slightly longer (no boundary found within the window).
    public let targetChunkSize: Int

    /// Minimum chunk size in tokens. Short trailing segments are merged into
    /// the previous chunk to avoid tiny segments with poor prosody.
    public let minChunkSize: Int

    private let encodeText: (String) -> [Int]
    private let decodeTokens: ([Int]) -> String

    public init(
        targetChunkSize: Int = TTSTextChunker.defaultTargetChunkSize,
        minChunkSize: Int = TTSTextChunker.defaultMinChunkSize,
        tokenizer: any Tokenizer
    ) {
        self.targetChunkSize = targetChunkSize
        self.minChunkSize = minChunkSize
        self.encodeText = { tokenizer.encode(text: $0) }
        self.decodeTokens = { tokenizer.decode(tokens: $0) }
    }

    /// Internal init for testing - accepts raw encode/decode functions so tests
    /// don't need a heavyweight `Tokenizer` conformance.
    init(
        targetChunkSize: Int = TTSTextChunker.defaultTargetChunkSize,
        minChunkSize: Int = TTSTextChunker.defaultMinChunkSize,
        encode: @escaping (String) -> [Int],
        decode: @escaping ([Int]) -> String
    ) {
        self.targetChunkSize = targetChunkSize
        self.minChunkSize = minChunkSize
        self.encodeText = encode
        self.decodeTokens = decode
    }

    /// Split text into chunks at natural boundaries, respecting the token budget.
    /// Returns an array of non-empty text chunks.
    public func chunk(_ text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var tokens = encodeText(trimmed)
        guard tokens.count > targetChunkSize else { return [trimmed] }

        var chunks: [String] = []

        while !tokens.isEmpty {
            if tokens.count <= targetChunkSize {
                let tail = decodeTokens(tokens)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !tail.isEmpty {
                    // Merge tiny trailing segment with previous chunk to avoid poor prosody
                    if encodeText(tail).count < minChunkSize, let last = chunks.last {
                        chunks[chunks.count - 1] = last + " " + tail
                    } else {
                        chunks.append(tail)
                    }
                }
                break
            }

            // Decode a window of targetChunkSize tokens, then find the best boundary within it
            let window = Array(tokens.prefix(targetChunkSize))
            let windowText = decodeTokens(window)
                .trimmingCharacters(in: .whitespacesAndNewlines)

            let accepted = lastBoundary(in: windowText) ?? windowText

            if !accepted.isEmpty {
                chunks.append(accepted)
            }

            // Re-tokenize the accepted text to advance by its exact token count,
            // avoiding drift from imperfect encode/decode round-trips
            let consumed = encodeText(accepted).count
            tokens.removeFirst(min(max(consumed, 1), tokens.count))
        }

        return chunks
    }

    // TODO: move to ArgmaxCore and share with WhisperKit
    /// Finds the last natural boundary (sentence → clause → word) in the string.
    /// Returns the text up to and including that boundary, or `nil` if none found.
    private func lastBoundary(in text: String) -> String? {
        let sentenceEnders: [Character] = [".", "!", "?", "\n"]
        let clauseEnders: [Character] = [",", ";", ":", "-", "–"]

        for enders in [sentenceEnders, clauseEnders] {
            if let idx = text.lastIndex(where: { enders.contains($0) }) {
                let candidate = String(text[...idx]).trimmingCharacters(in: .whitespacesAndNewlines)
                if encodeText(candidate).count >= minChunkSize {
                    return candidate
                }
            }
        }

        // Word boundary fallback
        if let idx = text.lastIndex(of: " ") {
            let candidate = String(text[..<idx]).trimmingCharacters(in: .whitespacesAndNewlines)
            if encodeText(candidate).count >= minChunkSize {
                return candidate
            }
        }

        return nil
    }
}
