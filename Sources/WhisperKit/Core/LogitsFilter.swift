//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
import Tokenizers

public protocol LogitsFiltering {
    func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray
}

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
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

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
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

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
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
