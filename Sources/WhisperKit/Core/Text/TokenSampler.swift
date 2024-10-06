//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import CoreML
import Foundation

public protocol TokenSampling {
    func update(tokens: [Int], logits: MLMultiArray, logProbs: [Float]) -> SamplingResult
    func finalize(tokens: [Int], logProbs: [Float]) -> SamplingResult
}

public struct SamplingResult: Sendable {
    public var tokens: [Int]
    public var logProbs: [Float]
    public var completed: Bool
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class GreedyTokenSampler: TokenSampling {
    public var temperature: FloatType
    public var eotToken: Int
    public var decodingOptions: DecodingOptions
    
    public init(temperature: FloatType, eotToken: Int, decodingOptions: DecodingOptions) {
        self.temperature = temperature
        self.eotToken = eotToken
        self.decodingOptions = decodingOptions
    }
    
    public func update(tokens: [Int], logits: MLMultiArray, logProbs: [Float]) -> SamplingResult {
        var softmaxOutput: BNNSNDArrayDescriptor?
        var argmaxOutput: BNNSNDArrayDescriptor?
        var softmaxInput: BNNSNDArrayDescriptor?
        var softmaxInputNeedsDeallocate = false
        
        var nextToken: Int?
        
        do {
            let logitsRawPointer = UnsafeMutableRawBufferPointer(
                start: logits.dataPointer,
                count: logits.count * MemoryLayout<FloatType>.stride
            )
            
            let logitsDescriptor = BNNSNDArrayDescriptor(
                data: logitsRawPointer,
                scalarType: FloatType.self,
                shape: .vector(logits.count, stride: 1)
            )!
            
            softmaxInput = logitsDescriptor
            
            // Scale logits by temperature if > 0
            if temperature != 0.0 {
                let scaledLogits = BNNSNDArrayDescriptor.allocateUninitialized(
                    scalarType: FloatType.self,
                    shape: .vector(logits.count, stride: 1)
                )
                
                try! BNNS.applyActivation(
                    activation: BNNS.ActivationFunction.linear(alpha: Float(1 / temperature)),
                    input: logitsDescriptor,
                    output: scaledLogits,
                    batchSize: 1
                )
                
                softmaxInput = scaledLogits
                softmaxInputNeedsDeallocate = true
            }
            
            // Always softmax once
            softmaxOutput = BNNSNDArrayDescriptor.allocateUninitialized(
                scalarType: Float.self,
                shape: .vector(logits.count, stride: 1)
            )
            
            try BNNS.applyActivation(
                activation: BNNS.ActivationFunction.softmax,
                input: softmaxInput!,
                output: softmaxOutput!,
                batchSize: 1
            )
            
            if temperature != 0.0 {
                // top-k multinomial sampling
                let k = decodingOptions.topK
                
                let bestValues = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float.self, shape: .vector(k, stride: 1))
                let bestIndices = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Int32.self, shape: .vector(k, stride: 1))
                
                try! BNNS.applyTopK(
                    k: k,
                    input: softmaxOutput!,
                    bestValues: bestValues,
                    bestIndices: bestIndices,
                    axis: 0,
                    batchSize: 1
                )
                
                let bestValuesResult = bestValues.makeArray(of: Float.self)!
                let bestIndicesResult = bestIndices.makeArray(of: Int32.self)!
                
                bestValues.deallocate()
                bestIndices.deallocate()
                
                // multinomial sample from top-k
                let sumOfbestIndicesResult = bestValuesResult.reduce(0, +)
                let rnd = Float.random(in: 0..<sumOfbestIndicesResult)
                var accumulator = Float(0.0)
                var chosenIndex = 0
                for i in 0..<bestValuesResult.count {
                    accumulator += bestValuesResult[i]
                    if rnd < accumulator {
                        chosenIndex = i
                        break
                    }
                }
                
                nextToken = Int(bestIndicesResult[chosenIndex])
            } else {
                // Argmax sampling
                argmaxOutput = BNNSNDArrayDescriptor.allocateUninitialized(
                    scalarType: Float.self,
                    shape: .vector(1, stride: 1)
                )
                
                try! BNNS.applyReduction(
                    BNNS.ReductionFunction.argMax,
                    input: logitsDescriptor,
                    output: argmaxOutput!,
                    weights: nil
                )
                
                let argmaxResult = argmaxOutput!.makeArray(of: Float.self)!
                
                nextToken = Int(argmaxResult[0])
            }
        } catch {
            Logging.error("Sampling error: \(error)")
        }
        
        // Log of softmax probability of chosen token
        let softmaxResult = softmaxOutput!.makeArray(of: Float.self)!
        let nextLogprob = log(Float(softmaxResult[nextToken!]))
        
        let nextTokens = tokens + [nextToken!]
        let nextLogprobs = logProbs + [nextLogprob]
        let completed = nextToken == eotToken
        
        // Deallocations
        softmaxOutput?.deallocate()
        argmaxOutput?.deallocate()
        if softmaxInputNeedsDeallocate {
            softmaxInput?.deallocate()
        }
        
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

open class BeamSearchTokenSampler: TokenSampling {
    public var beamSize: Int
    public var eotToken: Int
    public var patience: Float
    var maxCandidates: Int
    var finishedSequences: [Float]
    
    public init(
        beamSize: Int,
        eotToken: Int,
        patience: Float = 1
    ) {
        self.beamSize = beamSize
        self.eotToken = eotToken
        self.patience = patience
        self.maxCandidates = Int(Float(beamSize) * patience)
        self.finishedSequences = []
        if self.maxCandidates <= 0 {
            self.maxCandidates = 1
            fatalError("Invalid beam size \(beamSize) or patience \(patience)")
        }
    }
    
    public func reset() {
        finishedSequences = []
    }
    
    public func update(tokens: [Int], logits: MLMultiArray, logProbs: [Float]) -> SamplingResult {
        // TODO: Implement
        fatalError("Not implemented: \(#function)")
    }
    
    public func finalize(tokens: [Int], logProbs: [Float]) -> SamplingResult {
        // TODO: Implement
        fatalError("Not implemented: \(#function)")
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)
open class NTokenSampler: TokenSampling {
    public var temperature: Float
     public var eotToken: Int
     public var decodingOptions: DecodingOptions

     public init(temperature: Float, eotToken: Int, decodingOptions: DecodingOptions) {
         self.temperature = temperature
         self.eotToken = eotToken
         self.decodingOptions = decodingOptions
     }

     public func update(tokens: [Int], logits: MLMultiArray, logProbs: [Float]) -> SamplingResult {
         // MLMultiArrayがFloat32であることを確認
         guard logits.dataType == .float32 else {
             fatalError("Logits MLMultiArray must be of type Float32")
         }

         let logitsCount = logits.count

         // ロジットデータへのアクセス
         let logitsPointer = logits.dataPointer.bindMemory(to: Float.self, capacity: logitsCount)
         let logitsBuffer = UnsafeBufferPointer(start: logitsPointer, count: logitsCount)
         var logitsArray = [Float](logitsBuffer)

         // 温度が0より大きい場合はロジットをスケーリング
         if temperature != 0.0 {
             let tempReciprocal = 1.0 / temperature
             vDSP_vsmul(logitsArray, 1, [tempReciprocal], &logitsArray, 1, vDSP_Length(logitsCount))
         }

         // ソフトマックス計算
         var softmaxOutput = [Float](repeating: 0, count: logitsCount)
         computeSoftmax(logitsArray, result: &softmaxOutput)

         var nextToken: Int = 0
         var nextLogprob: Float = 0.0

         if temperature != 0.0 {
             // トップKのサンプリング
             let k = min(decodingOptions.topK, logitsCount)

             // 値とインデックスをペアにしてソート
             let indices = Array(0..<logitsCount)
             let sortedPairs = zip(softmaxOutput, indices).sorted { $0.0 > $1.0 }
             let topKPairs = sortedPairs.prefix(k)

             let topKValues = topKPairs.map { $0.0 }
             let topKIndices = topKPairs.map { $0.1 }

             // トップKの確率を正規化
             let sumTopK = topKValues.reduce(0, +)
             let normalizedTopKValues = topKValues.map { $0 / sumTopK }

             // トップKからサンプリング
             let randomValue = Float.random(in: 0..<1)
             var cumulativeProbability: Float = 0.0
             for (i, probability) in normalizedTopKValues.enumerated() {
                 cumulativeProbability += probability
                 if randomValue < cumulativeProbability {
                     nextToken = topKIndices[i]
                     nextLogprob = log(probability)
                     break
                 }
             }
         } else {
             // アーグマックスサンプリング
             var maxValue: Float = 0
             var maxIndex: vDSP_Length = 0
             vDSP_maxvi(softmaxOutput, 1, &maxValue, &maxIndex, vDSP_Length(logitsCount))
             nextToken = Int(maxIndex)
             nextLogprob = log(maxValue)
         }

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

     // ソフトマックスを効率的に計算するヘルパー関数
     func computeSoftmax(_ input: [Float], result: inout [Float]) {
         var input = input

         // オーバーフローを防ぐために最大値を引く
         var maxValue: Float = 0
         vDSP_maxv(input, 1, &maxValue, vDSP_Length(input.count))
         var negativeMax = -maxValue
         vDSP_vsadd(input, 1, &negativeMax, &input, 1, vDSP_Length(input.count))

         // 指数関数を適用
         vvexpf(&result, input, [Int32(input.count)])

         // 指数関数の合計を計算
         var sumOfExponents: Float = 0
         vDSP_sve(result, 1, &sumOfExponents, vDSP_Length(input.count))

         // 合計で割って確率を得る
         vDSP_vsdiv(result, 1, &sumOfExponents, &result, 1, vDSP_Length(input.count))
     }
 }
