//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
import MLX
import MLXNN
import MLXRandom
import WhisperKit

open class MLXGreedyTokenSampler: TokenSampling {
    public var temperature: Float
    public var eotToken: Int
    public var decodingOptions: DecodingOptions

    public init(temperature: Float, eotToken: Int, decodingOptions: DecodingOptions) {
        self.temperature = temperature
        self.eotToken = eotToken
        self.decodingOptions = decodingOptions
    }

    public func update(tokens: [Int], logits: MLMultiArray, logProbs: [Float]) -> SamplingResult {
        return update(tokens: tokens, logits: logits.asMLXArray(FloatType.self), logProbs: logProbs)
    }

    public func update(tokens: [Int], logits: MLXArray, logProbs: [Float]) -> SamplingResult {
        let softmaxOutput: MLXArray
        var sampledToken: MLXArray

        // Scale logits by temperature if > 0
        let scaledLogits = temperature != 0.0 ? logits / MLXArray(temperature) : logits

        // Always apply softmax
        softmaxOutput = softmax(scaledLogits, axis: -1)

        if temperature != 0.0 {
            // Top-k multinomial sampling
            let sortedIndices = MLX.argSort(softmaxOutput, axis: -1)

            // Implement top-k selection (argSort is ascending)
            let topKIndices = MLXArray(-decodingOptions.topK..<0)
            let sortedProbs = take(softmaxOutput, sortedIndices, axis: -1)
            let bestValues = sortedProbs.take(topKIndices, axis: -1)
            let bestIndices = sortedIndices.take(topKIndices, axis: -1)

            // multinomial sample from top-k
            let sumOfbestIndicesResult = bestValues.sum()
            let rnd = MLXRandom.uniform(low: 0.0, high: sumOfbestIndicesResult)
            let cumulativeProbs = cumsum(bestValues, axis: -1)

            let chosenIndex = MLX.argMax(cumulativeProbs .>= rnd)

            sampledToken = bestIndices.take(chosenIndex)
        } else {
            // Argmax sampling
            sampledToken = MLX.argMax(softmaxOutput, axis: -1)
        }
        let nextToken = sampledToken.item(Int.self)

        // Log of softmax probability of chosen token
        let nextProb = softmaxOutput.take(sampledToken)
        let nextLogprob = MLX.log(nextProb).item(Float.self)

        let nextTokens = tokens + [nextToken]
        let nextLogprobs = logProbs + [nextLogprob]
        let completed = nextToken == eotToken

        return SamplingResult(tokens: nextTokens, logProbs: nextLogprobs, completed: completed)
    }

    public func finalize(tokens: [Int], logProbs: [Float]) -> SamplingResult {
        var finalTokens = tokens
        var finalLogProbs = logProbs
        if tokens.last != eotToken {
            finalTokens.append(eotToken)
            finalLogProbs.append(0)
        }

        return SamplingResult(tokens: finalTokens, logProbs: finalLogProbs, completed: true)
    }
}
