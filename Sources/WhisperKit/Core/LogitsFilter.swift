//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
import Tokenizers

public protocol LogitsFiltering {
    func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class SuppressTokensFilter: LogitsFiltering {
    let suppressTokens: [Int]
    private let suppressTokenIndexes: [[NSNumber]]

    public init(suppressTokens: [Int]) {
        self.suppressTokens = suppressTokens
        self.suppressTokenIndexes = suppressTokens.map { [0, 0, $0 as NSNumber] }
    }

    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        logits.fill(indexes: suppressTokenIndexes, with: -FloatType.infinity)
        return logits
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class SuppressBlankFilter: LogitsFiltering {
    let suppressBlankTokens: [Int]
    let sampleBegin: Int
    private let suppressTokenIndexes: [[NSNumber]]

    public init(suppressBlankTokens: [Int], sampleBegin: Int) {
        self.suppressBlankTokens = suppressBlankTokens
        self.sampleBegin = sampleBegin
        self.suppressTokenIndexes = suppressBlankTokens.map { [0, 0, $0 as NSNumber] }
    }

    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        guard tokens.count == sampleBegin else {
            return logits
        }
        logits.fill(indexes: suppressTokenIndexes, with: -FloatType.infinity)
        return logits
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class TimestampRulesFilter: LogitsFiltering {
    let tokenizer: Tokenizer
    let sampleBegin: Int
    let maxInitialTimestamp: Int?

    public init(tokenizer: Tokenizer, sampleBegin: Int) {
        // TODO: implement
        fatalError("Not implemented: \(#function)")
    }

    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        // TODO: implement
        return logits
    }
}


@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class LanguageLogitsFilter: LogitsFiltering {
    let allLanguageTokens: [Int]
    let logitsDim: Int
    let sampleBegin: Int

    public init(allLanguageTokens: [Int], logitsDim: Int, sampleBegin: Int) {
        self.allLanguageTokens = allLanguageTokens
        self.logitsDim = logitsDim
        self.sampleBegin = sampleBegin
    }

    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        guard tokens.count == sampleBegin else{
            return logits
        }
        //get the non-language token indexes and suppress them
        let nonLanguageTokenIndexes = self.getNonLanguageTokenIndexes(logitsDim: self.logitsDim, allLanguageTokens: self.allLanguageTokens)
        logits.fill(indexes: nonLanguageTokenIndexes, with: -FloatType.infinity)
        return logits
    }
    
    private func getNonLanguageTokenIndexes(logitsDim: Int, allLanguageTokens: [Int]) -> [[NSNumber]]{
        var indexes: [[NSNumber]] = []
        for i in 0..<logitsDim{
            if !allLanguageTokens.contains(i){
                indexes.append([0, 0, i as NSNumber])
            }
        }
        return indexes
    }
}
