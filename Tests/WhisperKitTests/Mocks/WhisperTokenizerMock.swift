//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

@testable import WhisperKit

final class WhisperTokenizerMock: WhisperTokenizer {
    let specialTokens: SpecialTokens
    let allLanguageTokens: Set<Int>

    init(specialTokenBegin: Int) {
        self.specialTokens = SpecialTokens(
            endToken: 1,
            englishToken: 2,
            noSpeechToken: 3,
            noTimestampsToken: 4,
            specialTokenBegin: specialTokenBegin,
            startOfPreviousToken: 5,
            startOfTranscriptToken: 6,
            timeTokenBegin: 7,
            transcribeToken: 8,
            translateToken: 9,
            whitespaceToken: 10
        )
        self.allLanguageTokens = []
    }

    func encode(text: String) -> [Int] {
        [1, 2, 3]
    }

    func decode(tokens: [Int]) -> String {
        "mock text"
    }

    func convertTokenToId(_ token: String) -> Int? {
        nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        "token_\(id)"
    }

    func splitToWordTokens(tokenIds: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        (["mock"], [[1, 2, 3]])
    }
}
