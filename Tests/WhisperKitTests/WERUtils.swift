import Foundation

// Return the operations needed to transform s1 into s2 using Wagner-Fischer algo.
// "i" = insertion, "d" = deletion, "r" = replacement
func wagnerFischerEditOperations(s1: String, s2: String) -> [Character] {
    let m = s1.count
    let n = s2.count
    var dp = Array(repeating: Array(repeating: (0, Character(" ")), count: n + 1), count: m + 1)
    
    // Initialize first row and column
    for i in 0...m {
        dp[i][0] = (i, i > 0 ? "d" : Character(" "))
    }
    for j in 0...n {
        dp[0][j] = (j, j > 0 ? "i" : Character(" "))
    }
    // Fill the matrix
    for i in 1...m {
        for j in 1...n {
            let cost = s1[s1.index(s1.startIndex, offsetBy: i - 1)] == s2[s2.index(s2.startIndex, offsetBy: j - 1)] ? 0 : 1
            let insertCost = dp[i][j - 1].0
            let deleteCost = dp[i - 1][j].0
            var replaceCost = dp[i - 1][j - 1].0
            var replaceOp = dp[i - 1][j - 1].1
            if cost == 1 {
                replaceOp = "r"
            }
            replaceCost += cost
            let minCost = min(insertCost + 1, deleteCost + 1, replaceCost)
            var operation: Character = Character(" ")
            if minCost == insertCost + 1 {
                operation = "i"
            } else if minCost == deleteCost + 1 {
                operation = "d"
            } else if cost == 1{
                operation = replaceOp
            }
            dp[i][j] = (minCost, operation)
        }
    }
    
    // Traceback to get the operations
    var i = m
    var j = n
    var operations = [Character]()
    while i > 0 || j > 0 {
        let (_, op) = dp[i][j]
        if op != Character(" ") {
            operations.append(op)
        }
        if op == "i" {
            j -= 1
        } else if op == "d" {
            i -= 1
        } else {
            i -= 1
            j -= 1
        }
    }
    operations.reverse()
    return operations
}

// MARK:- TRANSFORMS
// sentences = ["this is   an   example ", "  hello goodbye  ", "  "]
// ['this is an example ', " hello goodbye ", " "]
func removeMultipleSpaces(sentences: [String]) -> [String]{
    
    var replacedSentences = [String]()
    for sentence in sentences {
        // Define the pattern you want to replace
        let pattern = "\\s\\s+"

        do {
            let regex = try NSRegularExpression(pattern: pattern, options: [])
            let replacedString = regex.stringByReplacingMatches(in: sentence, options: [],
                                                                 range: NSRange(location: 0, length: sentence.utf16.count),
                                                                 withTemplate: " ")
            replacedSentences.append(replacedString)
        } catch {
            print("Error while creating regex: \(error)")
        }
    }
    return replacedSentences
}

//[" this is an example ", "  hello goodbye  ", "  "]
//['this is an example', "hello goodbye", ""]
func strip(sentences: [String]) -> [String]{
    var replacedSentences = [String]()
    
    for sentence in sentences {
        let replacedString = sentence.trimmingCharacters(in: .whitespaces)
        replacedSentences.append(replacedString)
    }
    return replacedSentences
}

//["hi", "this is an example"]
//[['hi'], ['this', 'is', 'an, 'example']]
func reduceToListOfListOfWords(sentences: [String], word_delimiter: String = " ") -> [[String]]{
    
    let sentence_collection = [[String]]()
    func process_string(sentence: String) -> [[String]]{
        return [sentence.components(separatedBy: word_delimiter).filter{ !$0.isEmpty }]
    }
    
    func process_list(sentences: [String]) -> [[String]]{
        var sentence_collection = [[String]]()
        
        for sentence in sentences{
            let list_of_words = process_string(sentence: sentence)[0]
            if !list_of_words.isEmpty {
                sentence_collection.append(list_of_words)
            }
        }
        
        return sentence_collection
    }
    
    return process_list(sentences: sentences)
}

func words2char(reference: [[String]], hypothesis: [[String]]) -> ([String],[String]){
    //tokenize each word into an integer
    let vocabulary = Set((reference + hypothesis).flatMap{$0})
    let word2char = Dictionary(uniqueKeysWithValues: vocabulary.enumerated().map { index, value in
        return (value, index)
    })
    
    let referenceChars = reference.map { sentence in
        String(sentence.map { word in
            Character(UnicodeScalar(word2char[word]!)!)
        })
    }

    let hypothesisChars = hypothesis.map { sentence in
        String(sentence.map { word in
            Character(UnicodeScalar(word2char[word]!)!)
        })
    }
    
    return (referenceChars, hypothesisChars)
}

func process_words(reference: [String], hypothesis: [String]) -> Double{
    var refTransformed = removeMultipleSpaces(sentences: reference)
    refTransformed = strip(sentences: refTransformed)
    let refTransformedReduced = reduceToListOfListOfWords(sentences: refTransformed)
    
    var hypTransformed = removeMultipleSpaces(sentences: hypothesis)
    hypTransformed = strip(sentences: hypTransformed)
    let hypTransformedReduced = reduceToListOfListOfWords(sentences: hypTransformed)
    
    let (refAsChars, hypAsChars) = words2char(reference: refTransformedReduced, hypothesis: hypTransformedReduced)
    
    var (numHits, numSubstitutions, numDeletions, numInsertions) = (0, 0, 0, 0)
    var (numRfWords, numHypWords) = (0, 0)
    
    for (reference_sentence, hypothesis_sentence) in zip(refAsChars, hypAsChars){
        // Get the required edit operations to transform reference into hypothesis
        let editOps: [Character] = wagnerFischerEditOperations(
            s1: reference_sentence, s2: hypothesis_sentence
        )
        
        // count the number of edits of each type
        let substitutions: Int = editOps.map { $0 == "r" ? 1 : 0 }.reduce(0, +)
        let deletions:Int = editOps.map { $0 == "d" ? 1 : 0 }.reduce(0, +)
        let insertions:Int = editOps.map { $0 == "i" ? 1 : 0 }.reduce(0, +)
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
