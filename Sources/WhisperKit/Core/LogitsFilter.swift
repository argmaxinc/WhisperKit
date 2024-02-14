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
    private let tokenIndexes: [[NSNumber]]

    public init(suppressTokens: [Int]) {
        self.suppressTokens = suppressTokens
        self.tokenIndexes = suppressTokens.map { [0, 0, $0 as NSNumber] }
    }

    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        let pointer = UnsafeMutablePointer<FloatType>(OpaquePointer(logits.dataPointer))
        for index in tokenIndexes {
            let linearOffset = logits.linearOffset(for: index)
            pointer[linearOffset] = -FloatType.infinity
        }
        return logits
    }
}

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
public class SuppressBlankFilter: LogitsFiltering {
    let tokenizer: Tokenizer
    let sampleBegin: Int

    public init(tokenizer: Tokenizer, sampleBegin: Int) {
        self.tokenizer = tokenizer
        self.sampleBegin = sampleBegin
        // TODO: implement
        fatalError("Not implemented: \(#function)")
    }

    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        if tokens.count == sampleBegin {
            if let blankToken = tokenizer.convertTokenToId(" ") {
                Logging.debug(blankToken)
            }
            // TODO: implement
        }
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
