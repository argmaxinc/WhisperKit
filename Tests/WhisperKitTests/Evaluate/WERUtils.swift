//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// Return the operations needed to transform s1 into s2 using Wagner-Fischer algo.
/// "i" = insertion, "d" = deletion, "r" = replacement
enum EditOp: UInt8 {
    case blank
    case replace
    case delete
    case insert
}

enum WERUtils {
    static func wordsToChars(reference: [[String]], hypothesis: [[String]]) -> ([String], [String]) {
        // tokenize each word into an integer
        let vocabulary = Set((reference + hypothesis).flatMap { $0 })
        let word2char = Dictionary(uniqueKeysWithValues: vocabulary.enumerated().map { index, value in
            (value, index)
        })

        let referenceCharsEfficient = reference.map { sentence in
            String(sentence.lazy.compactMap { word in
                if let charCode = word2char[word], let unicodeScalar = UnicodeScalar(charCode) {
                    return Character(unicodeScalar)
                }
                return nil
            })
        }

        let hypothesisCharsEfficient = hypothesis.map { sentence in
            String(sentence.lazy.compactMap { word in
                if let charCode = word2char[word], let unicodeScalar = UnicodeScalar(charCode) {
                    return Character(unicodeScalar)
                }
                return nil
            })
        }

        return (referenceCharsEfficient, hypothesisCharsEfficient)
    }

    static func processWords(reference: [String], hypothesis: [String]) -> (Double, [[String?]]) {
        var refTransformed = NormalizationUtils.removeMultipleSpaces(sentences: reference)
        refTransformed = NormalizationUtils.strip(sentences: refTransformed)
        let refTransformedReduced = NormalizationUtils.reduceToListOfListOfWordsWithSpaces(sentences: refTransformed)

        var hypTransformed = NormalizationUtils.removeMultipleSpaces(sentences: hypothesis)
        hypTransformed = NormalizationUtils.strip(sentences: hypTransformed)
        let hypTransformedReduced = NormalizationUtils.reduceToListOfListOfWordsWithSpaces(sentences: hypTransformed)

        let (refAsChars, hypAsChars) = WERUtils.wordsToChars(reference: refTransformedReduced, hypothesis: hypTransformedReduced)

        let refArrays = refAsChars.map { Array($0.unicodeScalars) }
        let hypArrays = hypAsChars.map { Array($0.unicodeScalars) }

        var (numHits, numSubstitutions, numDeletions, numInsertions) = (0, 0, 0, 0)
        var (numRfWords, numHypWords) = (0, 0)
        var diffResult: [[String?]] = []

        for (referenceSentence, hypothesisSentence) in zip(refArrays, hypArrays) {
            let editOps = levenshtein(referenceSentence, hypothesisSentence)

            // count the number of edits of each type
            var substitutions = 0
            var deletions = 0
            var insertions = 0

            var referenceIndex = 0
            var hypothesisIndex = 0
            for op in editOps {
                switch op {
                    case .replace:
                        diffResult.append([String(refTransformedReduced[0][referenceIndex]), "-"])
                        diffResult.append([String(hypTransformedReduced[0][hypothesisIndex]), "+"])
                        substitutions += 1
                        referenceIndex += 1
                        hypothesisIndex += 1
                    case .delete:
                        diffResult.append([String(refTransformedReduced[0][referenceIndex]), "-"])
                        deletions += 1
                        referenceIndex += 1
                    case .insert:
                        diffResult.append([String(hypTransformedReduced[0][hypothesisIndex]), "+"])
                        insertions += 1
                        hypothesisIndex += 1
                    case .blank:
                        diffResult.append([String(refTransformedReduced[0][referenceIndex]), nil])
                        referenceIndex += 1
                        hypothesisIndex += 1
                }
            }

            let hits: Int = referenceSentence.count - (substitutions + deletions)

            numHits += hits
            numSubstitutions += substitutions
            numDeletions += deletions
            numInsertions += insertions
            numRfWords += referenceSentence.count
            numHypWords += hypothesisSentence.count
        }

        let wer = Double(numSubstitutions + numDeletions + numInsertions) / Double(numHits + numSubstitutions + numDeletions)

        return (wer, diffResult)
    }

    static func evaluate(originalTranscript: String, generatedTranscript: String, normalizeOriginal: Bool = true) -> (wer: Double, diff: [[String?]]) {
        let normalizer = EnglishTextNormalizer()
        let reference = normalizeOriginal ? normalizer.normalize(text: originalTranscript) : originalTranscript
        let hypothesis = normalizer.normalize(text: generatedTranscript)

        let (wer, diff) = WERUtils.processWords(
            reference: [reference],
            hypothesis: [hypothesis]
        )
        return (wer, diff)
    }

    static func processDiff(originalTranscript: String, generatedTranscript: String) -> [[String?]] {
        let (_, diff) = evaluate(originalTranscript: originalTranscript, generatedTranscript: generatedTranscript)
        return diff
    }

    static func diffString(from diff: [[String?]]) -> String {
        return diff.compactMap { entry -> String? in
            guard let word = entry[0], word != " " else { return nil }
            if let changeType = entry[1] {
                return "\(changeType)\(word)"
            }
            return word
        }.joined(separator: " ")
    }
}
