//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import MLX
import MLXNN
import MLXRandom
import WhisperKit
import Foundation

public protocol MLXTokenSampling {
    func update(tokens: [Int], logits: MLXArray, logProbs: [Float]) -> SamplingResult
    func finalize(tokens: [Int], logProbs: [Float]) -> SamplingResult
}

public struct SamplingResult {
    public var tokens: [Int]
    public var logProbs: [Float]
    public var completed: Bool

    public init(tokens: [Int], logProbs: [Float], completed: Bool) {
        self.tokens = tokens
        self.logProbs = logProbs
        self.completed = completed
    }
}

open class MLXGreedyTokenSampler: MLXTokenSampling {
    public var temperature: Float
    public var eotToken: Int
    public var decodingOptions: DecodingOptions

    public init(temperature: Float, eotToken: Int, decodingOptions: DecodingOptions) {
        self.temperature = temperature
        self.eotToken = eotToken
        self.decodingOptions = decodingOptions
    }

    public func update(tokens: [Int], logits: MLXArray, logProbs: [Float]) -> SamplingResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        print("Input shapes:")
        print("logits shape:", logits.shape)
        print("logits strides:", logits.strides)
        
//        let flattenStartTime = CFAbsoluteTimeGetCurrent()
//        let logitArray = logits.flattened()
//        let flattenEndTime = CFAbsoluteTimeGetCurrent()
//        print("Flattening time: \(flattenEndTime - flattenStartTime) seconds")
//        
//        print("Flattened logits shape:", logitArray.shape)
//        print("Flattened logits strides:", logitArray.strides)

        let scaleStartTime = CFAbsoluteTimeGetCurrent()
        // Scale logits by temperature if > 0

//        let scaledLogits = temperature != 0.0 ? logitArray / MLXArray(temperature) : logitArray
        let scaleEndTime = CFAbsoluteTimeGetCurrent()
        print("Scaling time: \(scaleEndTime - scaleStartTime) seconds")

        let softmaxStartTime = CFAbsoluteTimeGetCurrent()
        // Apply softmax
//        let probs = MLX.softmax(scaledLogits)
        let probs: MLXArray
        if temperature != 0.0 {
            probs = softmax(logits / temperature, axis: -1)
        } else {
            probs = logits
        }

//        let sortedIndices = argSort(probs, axis: -1)
//
//        let sortedProbs = take(probs, sortedIndices, axis: -1).squeezed(axis: 0)
//        ---- Transcription Timings ----
//        Audio Load:              0.00 ms /      1 runs (    0.00 ms/run)  0.00%
//        Audio Processing:        1.51 ms /      3 runs (    0.50 ms/run)  0.03%
//        Mels:                  100.73 ms /      3 runs (   33.58 ms/run)  2.19%
//        Encoding:              420.60 ms /      3 runs (  140.20 ms/run)  9.13%
//        Matrices Init:           1.01 ms /      1 runs (    1.01 ms/run)  0.02%
//        Prefill:                 0.05 ms /      1 runs (    0.05 ms/run)  0.00%
//        Decoding:             3790.05 ms /    248 runs (   15.28 ms/run) 82.29%
//        Non-inference:         231.55 ms /    248 runs (    0.93 ms/run)  5.03%
//        - Logit Filtering:       0.01 ms /    248 runs (    0.00 ms/run)  0.00%
//        - Sampling:            156.38 ms /    248 runs (    0.63 ms/run)  3.40%
//        - Kv Caching:           16.10 ms /    248 runs (    0.06 ms/run)  0.35%
//        - Word Timestamps:       0.00 ms /      0 runs (    0.00 ms/run)  0.00%
//        - Windowing:             1.31 ms /      3 runs (    0.44 ms/run)  0.03%
//        Fallbacks:               0.00 ms /      0 runs (    0.00 ms/run)  0.00%
//        Decoding Full Loop:   4604.25 ms /    248 runs (   18.57 ms/run) 99.97%
//        -------------------------------
//        Model Load Time:               0.54 seconds
//        Inference Duration (Global):   4.61 seconds
//        - Decoding Loop (Avg/window):  1.53 seconds
//        - Audio Windows:               3.00
//        Time to first token:           0.28 seconds
//        Total Tokens:                  247
//        Tokens per Second:             53.85 tok/s
//        Real Time Factor:              0.077
//        Fallbacks:                     0.0
        let softmaxEndTime = CFAbsoluteTimeGetCurrent()
        print("Softmax time: \(softmaxEndTime - softmaxStartTime) seconds")
        //        if temperature != 0.0 {
        //            // Top-k multinomial sampling
        //            let k = decodingOptions.topK
        //        let test = MLX.top([1, 2, 3], k: 2)
        //            let topKValues = MLX.argSort().top(probs, k: k)
        //
        //            // Multinomial sample from top-k
        //            let sumOfTopKValues = topKValues.sum().item()
        //            let rnd = MLXRandom.uniform(Float.self, low: 0, high: sumOfTopKValues)
        //            let cumulativeProbs = MLX.cumsum(topKValues)
        //            let chosenIndex = MLX.argMax(cumulativeProbs .>= rnd).item()
        //
        //            nextToken = topKIndices[chosenIndex].item()
        //            nextLogprob = MLX.log(topKValues[chosenIndex]).item()
        //        } else {
        // Argmax sampling
//        nextLogprob = probs.take(nextToken)
        //        }
        var nextToken: MLXArray
        var nextLogprob: MLXArray

//        nextToken = MLX.argMax(probs, axis: -1)


        let samplingStartTime = CFAbsoluteTimeGetCurrent()
        // Argmax sampling
//        nextToken = compiledArgmax(probs)
//        measure(noncompiledArgmax, probs)
//        measure(compiledArgmax, probs)
//        nextLogprob = probs.take(nextToken)
        let token: Int = compiledArgmax(probs).item()
        let logprob: Float = 0.05//nextLogprob.item()
        let samplingEndTime = CFAbsoluteTimeGetCurrent()
        print("Sampling time: \(samplingEndTime - samplingStartTime) seconds")

        let postProcessStartTime = CFAbsoluteTimeGetCurrent()
        let nextTokens = tokens + [token]
        let nextLogprobs: [Float] = logProbs + [logprob]
        let completed = token == eotToken
        let postProcessEndTime = CFAbsoluteTimeGetCurrent()
        print("Post-processing time: \(postProcessEndTime - postProcessStartTime) seconds")

        let endTime = CFAbsoluteTimeGetCurrent()
        print("Total update time: \(endTime - startTime) seconds")

        return SamplingResult(tokens: nextTokens, logProbs: nextLogprobs, completed: completed)
    }


    private let compiledArgmax: (MLXArray) -> MLXArray = compile { logits in
        MLX.argMax(logits, axis: -1)
    }

    private func noncompiledArgmax(_ logits: MLXArray) -> MLXArray {
        return MLX.argMax(logits, axis: -1)
    }

    func measure(_ f: (MLXArray) -> MLXArray, _ x: MLXArray) {
        // warm up
        for _ in 0..<10 {
            eval(f(x))
        }

        let start = Date.timeIntervalSinceReferenceDate
        let iterations = 100
        for _ in 0..<iterations {
            eval(f(x))
        }
        let end = Date.timeIntervalSinceReferenceDate

        let timePerIteration = 1000.0 * (end - start) / Double(iterations)

        print("Time per iteration \(timePerIteration.formatted()) ms")
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
