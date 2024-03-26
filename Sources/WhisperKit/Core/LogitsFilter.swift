//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import CoreML
import Foundation
import Tokenizers

public protocol LogitsFiltering {
    func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class SuppressTokensFilter: LogitsFiltering {
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
open class SuppressBlankFilter: LogitsFiltering {
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

/// Implementation based on https://github.com/openai/whisper/blob/master/whisper/decoding.py#L441
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class TimestampRulesFilter: LogitsFiltering {
    let transcribeToken: Int
    let translateToken: Int
    let noTimestampsToken: Int
    let timeTokenBegin: Int
    let endToken: Int
    let sampleBegin: Int
    let maxInitialTimestampIndex: Int?
    let isModelMultilingual: Bool

    public init(
        transcribeToken: Int,
        translateToken: Int,
        noTimestampsToken: Int,
        timeTokenBegin: Int,
        endToken: Int,
        sampleBegin: Int,
        maxInitialTimestampIndex: Int?,
        isModelMultilingual: Bool
    ) {
        self.transcribeToken = transcribeToken
        self.translateToken = translateToken
        self.noTimestampsToken = noTimestampsToken
        self.timeTokenBegin = timeTokenBegin
        self.endToken = endToken
        self.sampleBegin = sampleBegin
        self.maxInitialTimestampIndex = maxInitialTimestampIndex
        self.isModelMultilingual = isModelMultilingual
    }

    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        guard let sampleBegin = sampleBegin(for: tokens) else {
            return logits
        }
        // suppress <|notimestamps|> which is handled by `withoutTimestamps`
        logits.fill(indexes: [[0, 0, noTimestampsToken as NSNumber]], with: -FloatType.infinity)

        if tokens.count > sampleBegin {
            // timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
            let sampledTokens = tokens[sampleBegin...]
            let lastWasTimestamp = sampledTokens.count >= 1 && sampledTokens.last! >= timeTokenBegin
            let penultimateWasTimestamp = sampledTokens.count < 2 || sampledTokens.dropLast().last! >= timeTokenBegin
            if lastWasTimestamp {
                if penultimateWasTimestamp {
                    // has to be non-timestamp
                    logits.fillLastDimension(indexes: timeTokenBegin..<logits.count, with: -FloatType.infinity)
                } else {
                    // cannot be normal text tokens
                    logits.fillLastDimension(indexes: 0..<endToken, with: -FloatType.infinity)
                }
            }

            let timestamps = sampledTokens.filter { $0 >= timeTokenBegin }
            if let lastTimestamp = timestamps.last {
                // timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                // also force each segment to have a nonzero length, to prevent infinite looping
                let timestampLast =
                    if lastWasTimestamp && !penultimateWasTimestamp {
                        lastTimestamp
                    } else {
                        lastTimestamp + 1
                    }
                logits.fillLastDimension(indexes: timeTokenBegin..<timestampLast, with: -FloatType.infinity)
            }
        }

        if tokens.count == sampleBegin {
            // suppress generating non-timestamp tokens at the beginning
            logits.fillLastDimension(indexes: 0..<timeTokenBegin, with: -FloatType.infinity)
            if let maxInitialTimestampIndex {
                // apply the `maxInitialTimestamp` option
                let lastAllowed = timeTokenBegin + maxInitialTimestampIndex + 1
                logits.fillLastDimension(indexes: lastAllowed..<logits.count, with: -FloatType.infinity)
            }
        }

        // if sum of probability over timestamps is above any other token, sample timestamp
        if sumOfProbabilityOverTimestampsIsAboveAnyOtherToken(logits: logits, timeTokenBegin: timeTokenBegin) {
            logits.fillLastDimension(indexes: 0..<timeTokenBegin, with: -FloatType.infinity)
        }
        return logits
    }

    private func sampleBegin(for tokens: [Int]) -> Int? {
        if isModelMultilingual {
            // NOTE: for multilingual model we don't want to supress "<|transcribe|>" or "<|translate|>" tokens
            if let taskTokenIndex = tokens.prefix(3).firstIndex(where: { $0 == transcribeToken || $0 == translateToken }) {
                return max(taskTokenIndex + 1, sampleBegin)
            } else {
                return nil
            }
        } else {
            return sampleBegin
        }
    }

    private func sumOfProbabilityOverTimestampsIsAboveAnyOtherToken(logits: MLMultiArray, timeTokenBegin: Int) -> Bool {
        let timeTokenBeginOffset = logits.linearOffset(for: [0, 0, timeTokenBegin as NSNumber])

        let logprobsInputPointer = UnsafeMutableRawBufferPointer(
            start: logits.dataPointer,
            count: logits.count * MemoryLayout<FloatType>.stride
        )

        guard let logprobsInputDescriptor = BNNSNDArrayDescriptor(
            data: logprobsInputPointer,
            scalarType: FloatType.self,
            shape: .vector(logits.count, stride: 1)
        ) else {
            Logging.error("Cannot create `logprobsInputDescriptor`")
            return false
        }

        let logprobs = BNNSNDArrayDescriptor.allocateUninitialized(
            scalarType: FloatType.self,
            shape: .vector(logits.count, stride: 1)
        )
        defer { logprobs.deallocate() }

        do {
            try BNNS.applyActivation(
                activation: BNNS.ActivationFunction.logSoftmax,
                input: logprobsInputDescriptor,
                output: logprobs,
                batchSize: 1
            )

            let timeTokenCount = logits.count - timeTokenBeginOffset
            let noTimeTokenCount = timeTokenBeginOffset
            let logSumExpInputPointer = UnsafeMutableRawBufferPointer(
                start: logprobs.data!.advanced(by: timeTokenBeginOffset * MemoryLayout<FloatType>.stride),
                count: timeTokenCount * MemoryLayout<FloatType>.stride
            )

            guard let logSumExpInputDescriptor = BNNSNDArrayDescriptor(
                data: logSumExpInputPointer,
                scalarType: FloatType.self,
                shape: .vector(timeTokenCount, stride: 1)
            ) else {
                Logging.error("Cannot create `logSumExpInputDescriptor`")
                return false
            }

            let timestampLogProb = BNNSNDArrayDescriptor.allocateUninitialized(
                scalarType: FloatType.self,
                shape: .vector(1, stride: 1)
            )
            defer { timestampLogProb.deallocate() }

            try BNNS.applyReduction(
                .logSumExp,
                input: logSumExpInputDescriptor,
                output: timestampLogProb,
                weights: nil
            )

            let maxTextTokenLogProbInputPointer = UnsafeMutableRawBufferPointer(
                start: logprobs.data,
                count: noTimeTokenCount * MemoryLayout<FloatType>.stride
            )

            guard let maxTextTokenLogProbInputDescriptor = BNNSNDArrayDescriptor(
                data: maxTextTokenLogProbInputPointer,
                scalarType: FloatType.self,
                shape: .vector(noTimeTokenCount, stride: 1)
            ) else {
                Logging.error("Cannot create `maxTextTokenLogProbInputDescriptor`")
                return false
            }

            let maxTextTokenLogProb = BNNSNDArrayDescriptor.allocateUninitialized(
                scalarType: FloatType.self,
                shape: .vector(1, stride: 1)
            )
            defer { maxTextTokenLogProb.deallocate() }

            try BNNS.applyReduction(
                .max,
                input: maxTextTokenLogProbInputDescriptor,
                output: maxTextTokenLogProb,
                weights: nil
            )

            guard let timestampLogProbValue = timestampLogProb.makeArray(of: FloatType.self)?.first,
                  let maxTextTokenLogProbValue = maxTextTokenLogProb.makeArray(of: FloatType.self)?.first
            else {
                Logging.error("Cannot create logProb arrays")
                return false
            }
            return timestampLogProbValue > maxTextTokenLogProbValue
        } catch {
            Logging.error("TimestampRulesFilter error: \(error)")
            return false
        }
    }
}
