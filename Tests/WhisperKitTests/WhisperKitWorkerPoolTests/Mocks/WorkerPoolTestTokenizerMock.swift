//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
@testable import WhisperKit

final class WorkerPoolTestTokenizerMock: WhisperTokenizer {
    let specialTokens = SpecialTokens(
        endToken: 1,
        englishToken: 2,
        noSpeechToken: 3,
        noTimestampsToken: 4,
        specialTokenBegin: 10_000,
        startOfPreviousToken: 6,
        startOfTranscriptToken: 7,
        timeTokenBegin: 8,
        transcribeToken: 9,
        translateToken: 10,
        whitespaceToken: 11
    )
    let allLanguageTokens: Set<Int> = []

    func encode(text: String) -> [Int] {
        []
    }

    func decode(tokens: [Int]) -> String {
        tokens.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Int(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(id)
    }

    func splitToWordTokens(tokenIds: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        let words = tokenIds.map(String.init)
        let wordTokens = tokenIds.map { [$0] }
        return (words, wordTokens)
    }
}
