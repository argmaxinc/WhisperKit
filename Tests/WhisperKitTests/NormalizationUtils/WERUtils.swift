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
    func processString(sentence: String) -> [[String]]{
        return [sentence.components(separatedBy: word_delimiter).filter{ !$0.isEmpty }]
    }
    
    func processList(sentences: [String]) -> [[String]]{
        var sentence_collection = [[String]]()
        
        for sentence in sentences{
            let list_of_words = processString(sentence: sentence)[0]
            if !list_of_words.isEmpty {
                sentence_collection.append(list_of_words)
            }
        }
        
        return sentence_collection
    }
    
    return processList(sentences: sentences)
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


// MARK: Normalization

class EnglishNumberNormalizer{
    //    Convert any spelled-out numbers into arabic numbers, while handling:
    //
    //    - remove any commas
    //    - keep the suffixes such as: `1960s`, `274th`, `32nd`, etc.
    //    - spell out currency symbols after the number. e.g. `$20 million` -> `20000000 dollars`
    //    - spell out `one` and `ones`
    //    - interpret successive single-digit numbers as nominal: `one oh one` -> `101`
    let zeros: Set<String>
    let ones: [String:Int]
    let onesPlural: [String:(Int, String)]
    let onesOrdinal: [String:(Int, String)]
    let onesSuffixed: [String:(Int, String)]
    
    let tens: [String:Int]
    let tensPlural: [String:(Int, String)]
    let tensOrdinal: [String:(Int, String)]
    let tensSuffixed: [String:(Int, String)]
    
    let multipliers: [String:Int]
    let multipliersPlural: [String : (Int, String)]
    let multipliersOrdinal: [String : (Int, String)]
    let multipliersSuffixed: [String : (Int, String)]
    
    let decimals: Set<String>
    let precedingPrefixers: [String:String]
    let followingPrefixers: [String:String]
    
    let prefixes: Set<String>
    let suffixers: [String:Any]
    let specials: Set<String>
    let words: Set<String>
    let literalWords: Set<String>
    
    
    init(){
        let zeros: Set = ["o", "oh", "zero"]
        let ones = Dictionary(uniqueKeysWithValues:[
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen"].enumerated().map { ($0.element, $0.offset + 1)})

        let onesPlural = Dictionary(uniqueKeysWithValues:
            ones.map { name, value in
                return (name == "six" ? "sixes" : name + "s", (value, "s"))
            }
        )

        var onesOrdinal = {
            var onesDictionary: [String: (Int, String)] = [
                "zeroth": (0, "th"),
                "first": (1, "st"),
                "second": (2, "nd"),
                "third": (3, "rd"),
                "fifth": (5, "th"),
                "twelfth": (12, "th")
            ]

            let updatedOnes = ones.filter { name, value in
                value > 3 && value != 5 && value != 12
            }.map { name, value in
                return (name + (name.hasSuffix("t") ? "h" : "th"), (value, "th"))
            }

            for (key, value) in updatedOnes {
                onesDictionary[key] = value
            }

            return (onesDictionary)
        }()

        let onesSuffixed = onesPlural.merging(onesOrdinal) { $1 }

        let tens = [
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        ]

        let tensPlural = Dictionary(uniqueKeysWithValues: tens.map { name, value in
            return (name.replacingOccurrences(of: "y", with: "ies"), (value, "s"))
        })
        let tensOrdinal = Dictionary(uniqueKeysWithValues: tens.map { name, value in
            return (name.replacingOccurrences(of: "y", with: "ieth"), (value, "th"))
        })

        let tensSuffixed = tensPlural.merging(tensOrdinal) { $1 }

        let multipliers: [String: Int] = [
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000
        //    "sextillion": 1_000_000_000_000_000_000_000,
        //    "septillion": 1_000_000_000_000_000_000_000_000,
        //    "octillion": 1_000_000_000_000_000_000_000_000_000,
        //    "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
        //    "decillion": 1_000_000_000_000_000_000_000_000_000_000_000
        ]

        let multipliersPlural = Dictionary(uniqueKeysWithValues: multipliers.map { name, value in
            return (name + "s", (value, "s"))
        })

        let multipliersOrdinal = Dictionary(uniqueKeysWithValues: multipliers.map { name, value in
            return (name + "th", (value, "th"))
        })

        let multipliersSuffixed = multipliersPlural.merging(multipliersOrdinal) { $1 }

        let decimals: Set = Set(ones.keys).union(tens.keys).union(zeros)

        let precedingPrefixers: [String: String] = [
            "minus": "-",
            "negative": "-",
            "plus": "+",
            "positive": "+"
        ]

        let followingPrefixers: [String: String] = [
            "pound": "£",
            "pounds": "£",
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "cent": "¢",
            "cents": "¢"
        ]

        let prefixes = Set(precedingPrefixers.values)
                        .union(followingPrefixers.values)

        let suffixers: [String: Any] = [
            "per": ["cent": "%"],
            "percent": "%"
        ]

        let specials: Set = ["and", "double", "triple", "point"]

        let words = zeros.union(ones.keys)
                         .union(onesSuffixed.keys)
                         .union(tens.keys)
                         .union(tensSuffixed.keys)
                         .union(multipliers.keys)
                         .union(multipliersSuffixed.keys)
                         .union(precedingPrefixers.keys)
                         .union(followingPrefixers.keys)
                         .union(suffixers.keys)
                         .union(specials)

        let literalWords: Set = ["one", "ones"]
        
        self.zeros = zeros
        self.ones = ones
        self.onesPlural = onesPlural
        self.onesOrdinal = onesOrdinal
        self.onesSuffixed = onesSuffixed
        
        self.tens =  tens
        self.tensPlural = tensPlural
        self.tensOrdinal = tensOrdinal
        self.tensSuffixed = tensSuffixed
        
        self.multipliers = multipliers
        self.multipliersPlural = multipliersPlural
        self.multipliersOrdinal = multipliersOrdinal
        self.multipliersSuffixed = multipliersSuffixed
        
        self.decimals = decimals
        self.precedingPrefixers = precedingPrefixers
        self.followingPrefixers = followingPrefixers
        
        self.prefixes = prefixes
        self.suffixers = suffixers
        self.specials = specials
        self.words = words
        self.literalWords = literalWords
    }
    
    func processWords(_ words: [String]) -> [String] {
        var prefix: String? = nil
        var value: String? = nil
        var skip = false
        var results: [String] = []

        func output(_ result: String) -> String {
            var result = result
            if let prefix = prefix {
                result = prefix + result
            }
            value = nil
            prefix = nil
            return result
        }

        
        for idx in 0..<words.count{
//            guard let current = wordsIterator.next() else { return "" }
            let current = words[idx]
            let prev = idx != 0 ? words[idx - 1] : nil
            let next = words.indices.contains(idx + 1) ? words[idx + 1] : nil
//            iteratorIndex += 1
            
            if skip {
                skip = false
                continue
            }
            
            // should be checking for regex match here. But if just checking for number, Int(next) should be good
            let nextIsNumeric = next != nil && Int(next!) != nil
            let hasPrefix = current.first.map { self.prefixes.contains(String($0)) } ?? false
            let currentWithoutPrefix = hasPrefix ? String(current.dropFirst()) : current
            
            if currentWithoutPrefix.range(of: #"^\d+(\.\d+)?$"#, options: .regularExpression) != nil {
                // arabic numbers (potentially with signs and fractions)
                if let f = Fraction(currentWithoutPrefix) {
                    if var v = value, v.hasSuffix(".") {
                        v = v + current
                        value = v
                        continue
                    } else if var v = value{
                        results.append(output(v))
                    }
                    else {
                        prefix = hasPrefix ? String(current.first!) : prefix
                        value = f.denominator == 1 ? String(f.numerator) : currentWithoutPrefix
                    }
                } else {
                    fatalError("Converting the fraction failed")
                }
            } else if !self.words.contains(current) {
                // non-numeric words
                if let v = value {
                    value = v
                    results.append(output(v))
                }
                results.append(output(current))
            } else if self.zeros.contains(current) {
                value = (value ?? "") + "0"
            } else if let ones = self.ones[current] {
                if value == nil {
                    value = String(ones)
                } else if var v = value, let prev = prev, self.ones[prev] != nil {
                    if let tens = self.tens[prev], ones < 10 {
                        value = String(v.dropLast()) + String(ones)
                    } else {
                        value = v + String(ones)
                    }
                } else if ones < 10 {
                    if var v = value, Int(v) != nil, Int(v)! % 10 == 0 {
                        value = String(Int(v)! + ones)
                    } else {
                        value = value! + String(ones)
                    }
                } else {
                    if var v = value, Int(v)! % 100 == 0 {
                        value = String(Int(v)! + ones)
                    } else {
                        value = value! + String(ones)
                    }
                }
            } else if let (ones, suffix) = self.onesSuffixed[current] {
                let currentIsOrdinal = self.onesSuffixed[current] != nil
                if value == nil {
                    results.append(output("\(ones)\(suffix)"))
                } else if let v = value, let prev = prev, self.ones[prev] != nil {
                    if let tens = self.tens[prev], ones < 10 {
                        results.append(output("\(v.dropLast())\(ones)\(suffix)"))
                    } else {
                        results.append(output("\(v)\(ones)\(suffix)"))
                    }
                } else if ones < 10 {
                    if let v = value, Int(v)! % 10 == 0 {
                        results.append(output("\(Int(v)! + ones)\(suffix)"))
                    } else {
                        results.append(output("\(value!)\(ones)\(suffix)"))
                    }
                } else {
                    if let v = value, Int(v)! % 100 == 0 {
                        results.append(output("\(Int(v)! + ones)\(suffix)"))
                    } else {
                        results.append(output("\(value!)\(ones)\(suffix)"))
                    }
                }
            value = nil
            } else if let tens = self.tens[current] {
                if value == nil {
                    value = String(tens)
                } else if let v = value, !v.isEmpty {
                    value = v + String(tens)
                } else {
                    if let v = value, Int(v)! % 100 == 0 {
                        value = String(Int(v)! + tens)
                    } else {
                        value = value! + String(tens)
                    }
                }
            } else if let (tens, suffix) = self.tensSuffixed[current] {
                if value == nil {
                    results.append(output("\(tens)\(suffix)"))
                } else if let v = value, !v.isEmpty {
                    if Int(v)! % 100 == 0 {
                        results.append(output("\(Int(v)! + tens)\(suffix)"))
                    } else {
                        results.append(output("\(value!)\(tens)\(suffix)"))
                    }
                } else {
                    value = nil
                }
            } else if let multiplier = self.multipliers[current] {
                if value == nil {
                    value = String(multiplier)
                } else if var v = value, let f = Fraction(v), let p = f * multiplier{
                    if p.denominator == 1 {
                        value = String(p.numerator)
                    } else {
                        value = v
                        results.append(output(v))
                    }
                } else if var v = value{
                    let before = Int(v)! / 1000 * 1000
                    let residual = Int(v)! % 1000
                    value = "\(before + residual * multiplier)"
                }
            } else if let (multiplier, suffix) = self.multipliersSuffixed[current] {
                if value == nil {
                    results.append(output("\(multiplier)\(suffix)"))
                } else if let v = value, let f = Fraction(v), f * multiplier == Fraction(numerator: 1, denominator: 1) {
                    results.append(output("\(f.numerator)\(suffix)"))
                } else if var value{
                    let before = Int(value)! / 1000 * 1000
                    let residual = Int(value)! % 1000
                    value = "\(before + residual * multiplier)"
                    results.append(output("\(value)\(suffix)"))
                }
                value = nil
            } else if let prefixValue = self.precedingPrefixers[current] {
                if value != nil {
                    results.append(output(value!))
                }
                if self.words.contains(next!) || nextIsNumeric {
                    prefix = prefixValue
                } else {
                    results.append(output(current))
                }
            } else if let prefixValue = self.followingPrefixers[current] {
                if value != nil {
                    prefix = prefixValue
                    results.append(output(value!))
                } else {
                    results.append(output(current))
                }
            } else if let suffixValue = self.suffixers[current] {
                if value != nil {
                    if let dictSuffixValue = suffixValue as? [String: String] {
                        if let n = next, let nextSuffix = dictSuffixValue[n] {
                            results.append(output("\(value!)\(nextSuffix)"))
                            skip = true
                        } else {
                            results.append(output(value!))
                            results.append(output(current))
                        }
                    } else {
                        results.append(output("\(value!)\(suffixValue)"))
                    }
                } else {
                    results.append(output(current))
                }
            } else if self.specials.contains(current) {
                if !self.words.contains(next!) && !nextIsNumeric {
                    if let v = value {
                        results.append(output(v))
                    }
                    results.append(output(current))
                } else if current == "and" {
                    if !self.multipliers.keys.contains(prev!) {
                        if let v = value {
                            results.append(output(v))
                        }
                        results.append(output(current))
                    }
                } else if current == "double" || current == "triple" {
                    if let ones = self.ones[next!] {
                        let repeats = current == "double" ? 2 : 3
                        value = "\(value ?? "")\(ones)\(repeats)"
                        skip = true
                    } else {
                        if let v = value {
                            results.append(output(v))
                        }
                        results.append(output(current))
                    }
                } else if current == "point" {
                    if self.decimals.contains(next!) || nextIsNumeric {
                        value = "\(value ?? "")."
                    }
                } else {
                    fatalError("Unexpected token: \(current)")
                }
            } else {
                fatalError("Unexpected token: \(current)")
            }
        }
        if let v = value{
            results.append(output(v))
        }
        return results
    }
    
    func preprocess(_ s: String) -> String {
        var results = [String]()
        
        let segments = s.split(separator: "and a half", omittingEmptySubsequences: false)
        for (i, segment) in segments.enumerated() {
            let trimmedSegment = segment.trimmingCharacters(in: .whitespaces)
            if trimmedSegment.isEmpty {
                continue
            }
            
            if i == segments.count - 1 {
                results.append(String(trimmedSegment))
            } else {
                results.append(String(trimmedSegment))
                let lastWord = trimmedSegment.split(separator: " ").last ?? ""
                if decimals.contains(String(lastWord)) || multipliers.keys.contains(String(lastWord)) {
                    results.append("point five")
                } else {
                    results.append("and a half")
                }
            }
        }
        
        var processedString = results.joined(separator: " ")
        
        // Put a space at number/letter boundary
        processedString = processedString.replacingOccurrences(of: #"([a-z])([0-9])"#, with: "$1 $2", options: .regularExpression)
        processedString = processedString.replacingOccurrences(of: #"([0-9])([a-z])"#, with: "$1 $2", options: .regularExpression)
        
        // Remove spaces which could be a suffix
        processedString = processedString.replacingOccurrences(of: #"([0-9])\s+(st|nd|rd|th|s)\b"#, with: "$1$2", options: .regularExpression)
        
        return processedString
    }
    
    func postprocess(_ s: String) -> String {
        func combineCents(match: NSTextCheckingResult, in string: String) -> String {
            guard let currencyRange = Range(match.range(at: 1), in: string),
                  let integerRange = Range(match.range(at: 2), in: string),
                  let centsRange = Range(match.range(at: 3), in: string) else {
                return String(string)
            }
            let currency = String(string[currencyRange])
            let integer = String(string[integerRange])
            let cents = Int(String(string[centsRange])) ?? 0
            return "\(currency)\(integer).\(String(format: "%02d", cents))"
        }

        func extractCents(match: NSTextCheckingResult, in string: String) -> String {
            guard let centsRange = Range(match.range(at: 1), in: string) else {
                return String(string)
            }
            let cents = Int(String(string[centsRange])) ?? 0
            return "¢\(cents)"
        }

        var processedString = s
        
        
        // apply currency postprocessing; "$2 and ¢7" -> "$2.07"
        do {
            let regex1 = try NSRegularExpression(pattern: #"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b"#)
            let matches1 = regex1.matches(in: processedString, range: NSRange(processedString.startIndex..., in: processedString))
            for match in matches1.reversed() {
                let range = Range(match.range, in: processedString)!
                let replacement = combineCents(match: match, in: processedString)
                processedString.replaceSubrange(range, with: replacement)
            }
        } catch {
            print("Error in regex: \(error)")
        }

        do {
            let regex2 = try NSRegularExpression(pattern: #"[€£$]0\\.([0-9]{1,2})\b"#)
            let matches2 = regex2.matches(in: processedString, range: NSRange(processedString.startIndex..., in: processedString))
            for match in matches2.reversed() {
                let range = Range(match.range, in: processedString)!
                let replacement = extractCents(match: match, in: processedString)
                processedString.replaceSubrange(range, with: replacement)
            }
        } catch {
            print("Error in regex: \(error)")
        }

        // write "one(s)" instead of "1(s)", just for readability
        processedString = processedString.replacingOccurrences(of: #"\b1(s?)\b"#, with: "one$1", options: .regularExpression)

        return processedString
    }
    
    func normalize(_ text: String) -> String{
        var s = self.preprocess(text)
        let out = self.processWords(s.components(separatedBy: " ")).filter({ $0 != ""})
        s = out.joined(separator: " ")
        s = self.postprocess(s)
        return s
    }
    
}

class EnglishSpellingNormalizer{
    //
    //Applies British-American spelling mappings as listed in [1].

    //[1] https://www.tysto.com/uk-us-spelling-list.html
    
    var mapping: [String:String] = [:]
    
    init(englishSpellingMapping:[String:String]){
        self.mapping = englishSpellingMapping
    }
    
    func normalize(_ text: String) -> String{
        let out = text.components(separatedBy: " ").map( {self.mapping[$0] ?? $0} )
        return out.joined(separator: " ")
    }
}

class EnglishTextNormalizer{
    let numberNormalizer: EnglishNumberNormalizer
    let spellingNormalizer: EnglishSpellingNormalizer
    let ignorePatterns = #"\b(hmm|mm|mhm|mmm|uh|um)\b"#
    let replacers: KeyValuePairs = [
        // common contractions
        #"\bwon't\b"#: "will not",
        #"\bcan't\b"#: "can not",
        #"\blet's\b"#: "let us",
        #"\bain't\b"#: "aint",
        #"\by'all\b"#: "you all",
        #"\bwanna\b"#: "want to",
        #"\bgotta\b"#: "got to",
        #"\bgonna\b"#: "going to",
        #"\bi'ma\b"#: "i am going to",
        #"\bimma\b"#: "i am going to",
        #"\bwoulda\b"#: "would have",
        #"\bcoulda\b"#: "could have",
        #"\bshoulda\b"#: "should have",
        #"\bma'am\b"#: "madam",
        // contractions in titles/prefixes
        #"\bmr\b"#: "mister ",
        #"\bmrs\b"#: "missus ",
        #"\bst\b"#: "saint ",
        #"\bdr\b"#: "doctor ",
        #"\bprof\b"#: "professor ",
        #"\bcapt\b"#: "captain ",
        #"\bgov\b"#: "governor ",
        #"\bald\b"#: "alderman ",
        #"\bgen\b"#: "general ",
        #"\bsen\b"#: "senator ",
        #"\brep\b"#: "representative ",
        #"\bpres\b"#: "president ",
        #"\brev\b"#: "reverend ",
        #"\bhon\b"#: "honorable ",
        #"\basst\b"#: "assistant ",
        #"\bassoc\b"#: "associate ",
        #"\blt\b"#: "lieutenant ",
        #"\bcol\b"#: "colonel ",
        #"\bjr\b"#: "junior ",
        #"\bsr\b"#: "senior ",
        #"\besq\b"#: "esquire ",
        // prefect tenses, ideally it should be any past participles, but it's harder..
        #"'d been\b"#: " had been",
        #"'s been\b"#: " has been",
        #"'d gone\b"#: " had gone",
        #"'s gone\b"#: " has gone",
        #"'d done\b"#: " had done",  // "'s done" is ambiguous
        #"'s got\b"#: " has got",
        // general contractions
        #"n't\b"#: " not",
        #"'re\b"#: " are",
        #"'s\b"#: " is",
        #"'d\b"#: " would",
        #"'ll\b"#: " will",
        #"'t\b"#: " not",
        #"'ve\b"#: " have",
        #"'m\b"#: " am",
    ]
    // non-ASCII letters that are not separated by "NFKD" normalization
    let ADDITIONAL_DIACRITICS = [
        "œ": "oe",
        "Œ": "OE",
        "ø": "o",
        "Ø": "O",
        "æ": "ae",
        "Æ": "AE",
        "ß": "ss",
        "ẞ": "SS",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "th",
        "ł": "l",
        "Ł": "L",
    ]
    
    init(){
        self.numberNormalizer = EnglishNumberNormalizer()
        self.spellingNormalizer = EnglishSpellingNormalizer(englishSpellingMapping: englishSpellingMappingAbbr)
    }
    
    func normalize(text: String) -> String{
        var processedText = text
        processedText = processedText.lowercased()
        
        processedText.regReplace(pattern: #"[<\[][^>\]]*[>\]]"#, replaceWith: "") // remove words between brackets
        processedText.regReplace(pattern: #"\(([^)]+?)\)"#, replaceWith: "") // remove words between parenthesis
        processedText.regReplace(pattern: self.ignorePatterns, replaceWith: "")
        processedText.regReplace(pattern: #"\s+'"#, replaceWith: "'") // standardize when there's a space before an apostrophe

        for (pattern, replacement) in self.replacers{
            processedText.regReplace(pattern: pattern, replaceWith: replacement)
        }
        
        processedText.regReplace(pattern: #"(\d),(\d)"#, replaceWith: #"$1$2"#) // remove commas between digits
        processedText.regReplace(pattern: #"\.([^0-9]|$)"#, replaceWith: " $1") // remove periods not followed by numbers
        processedText = self.removeSymbolsAndDiacritics(text: processedText, keep: ".%$¢€£")  // keep some symbols for numerics

        processedText = self.numberNormalizer.normalize(processedText)
        processedText = self.spellingNormalizer.normalize(processedText)

        // now remove prefix/suffix symbols that are not preceded/followed by numbers
        processedText.regReplace(pattern: #"[.$¢€£]([^0-9])"#, replaceWith: #" $1"#)
        processedText.regReplace(pattern: #"([^0-9])%"#, replaceWith: #"$1 "#)
        processedText.regReplace(pattern: #"\s+"#, replaceWith: " ") // replace any successive whitespace characters with a space
        return processedText
    }
    
    func removeSymbolsAndDiacritics(text: String, keep:String="") -> String{
        //
        //Replace any other markers, symbols, and punctuations with a space, and drop any diacritics
        //(category 'Mn' and some manual mappings)
        //"""
        let keepSet = Set(keep)
        let categoriesToReplaceWithSpace: [Unicode.GeneralCategory] = [
            .nonspacingMark,
            .spacingMark,
            .enclosingMark,
            .mathSymbol,
            .otherSymbol,
            .currencySymbol,
            .modifierSymbol,
            .dashPunctuation,
            .openPunctuation,
            .closePunctuation,
            .finalPunctuation,
            .otherPunctuation,
            .initialPunctuation,
            .connectorPunctuation
        ]
        func replaceCharacter(char: Character) -> String{
            
            if keepSet.contains(char){
                return String(char)
            }
            else if self.ADDITIONAL_DIACRITICS.keys.contains(String(char)){
                return self.ADDITIONAL_DIACRITICS[String(char)]!
            }
            else if unicodeCategoryFor(char: char) == Unicode.GeneralCategory.nonspacingMark{
                return ""
            }
            else if let category = unicodeCategoryFor(char: char), categoriesToReplaceWithSpace.contains(category){
                return " "
            }
            return String(char)
        }
                                                             
        func unicodeCategoryFor(char: Character) -> Unicode.GeneralCategory?{
            guard let scalar = char.unicodeScalars.first else {return nil}
            return scalar.properties.generalCategory
        }
        
        if let normalizedString = text.applyingTransform(StringTransform(rawValue: "NFKD"), reverse: false) {
            var out = normalizedString.map({ replaceCharacter(char: $0)})
            return out.joined(separator: "")
        }
        return text
    }
}

private extension String {
    mutating func regReplace(pattern: String, replaceWith: String = "") {
        do {
            let regex = try NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .anchorsMatchLines])
            let range = NSRange(self.startIndex..., in: self)
            self = regex.stringByReplacingMatches(in: self, options: [], range: range, withTemplate: replaceWith)
        } catch { return }
    }
}
