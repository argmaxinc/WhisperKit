import Foundation

// Return the operations needed to transform s1 into s2 using Wagner-Fischer algo.
// "i" = insertion, "d" = deletion, "r" = replacement
enum EditOp:UInt8{
    case blank
    case replace
    case delete
    case insert
}

class WERUtils{
    static func wordsToChars(reference: [[String]], hypothesis: [[String]]) -> ([String],[String]){
        //tokenize each word into an integer
        let vocabulary = Set((reference + hypothesis).flatMap{$0})
        let word2char = Dictionary(uniqueKeysWithValues: vocabulary.enumerated().map { index, value in
            return (value, index)
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
    
    static func processWords(reference: [String], hypothesis: [String]) -> Double{
        var refTransformed = NormalizationUtils.removeMultipleSpaces(sentences: reference)
        refTransformed = NormalizationUtils.strip(sentences: refTransformed)
        let refTransformedReduced = NormalizationUtils.reduceToListOfListOfWords(sentences: refTransformed)
        
        var hypTransformed = NormalizationUtils.removeMultipleSpaces(sentences: hypothesis)
        hypTransformed = NormalizationUtils.strip(sentences: hypTransformed)
        let hypTransformedReduced = NormalizationUtils.reduceToListOfListOfWords(sentences: hypTransformed)
        
        let (refAsChars, hypAsChars) = WERUtils.wordsToChars(reference: refTransformedReduced, hypothesis: hypTransformedReduced)
        
        let refArrays = refAsChars.map({Array($0.unicodeScalars)})
        let hypArrays = hypAsChars.map({Array($0.unicodeScalars)})
        
        var (numHits, numSubstitutions, numDeletions, numInsertions) = (0, 0, 0, 0)
        var (numRfWords, numHypWords) = (0, 0)
        
        for (reference_sentence, hypothesis_sentence) in zip(refArrays, hypArrays){
            // Get the required edit operations to transform reference into hypothesis
            let editOps = hirschberg(reference_sentence, hypothesis_sentence)
            
            // count the number of edits of each type
            var substitutions: Int = 0
            var deletions: Int = 0
            var insertions: Int = 0
            
            for op in editOps{
                switch op{
                case .replace:
                    substitutions += 1
                    continue
                case .delete:
                    deletions += 1
                    continue
                case .insert:
                    insertions += 1
                    continue
                case .blank:
                    continue
                }
            }
            
            let hits:Int = reference_sentence.count - (substitutions + deletions)
            
            // update state
            numHits += hits
            numSubstitutions += substitutions
            numDeletions += deletions
            numInsertions += insertions
            numRfWords += reference_sentence.count
            numHypWords += hypothesis_sentence.count
        }
        let (S, D, I, H) = (numSubstitutions, numDeletions, numInsertions, numHits)
        
        let wer = Double(S + D + I) / Double(H + S + D)
        
        return wer
    }
    
    static func evaluate(originalTranscript: String, generatedTranscript: String, normalizeOriginal: Bool = false) -> Double{
        var wer: Double = -Double.infinity
        let normalizer = EnglishTextNormalizer()
        let reference = normalizeOriginal ? normalizer.normalize(text: originalTranscript) : originalTranscript
        let hypothesis = normalizer.normalize(text: generatedTranscript)
        
        wer = WERUtils.processWords(
            reference: [reference],
            hypothesis: [hypothesis]
        )
        return wer
    }
}
