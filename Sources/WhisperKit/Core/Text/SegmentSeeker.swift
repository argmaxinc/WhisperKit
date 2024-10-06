//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import CoreML
import Foundation
import Tokenizers

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public protocol SegmentSeeking {
    func findSeekPointAndSegments(
        decodingResult: DecodingResult,
        options: DecodingOptions,
        allSegmentsCount: Int,
        currentSeek seek: Int,
        segmentSize: Int,
        sampleRate: Int,
        timeToken: Int,
        specialToken: Int,
        tokenizer: WhisperTokenizer
    ) -> (Int, [TranscriptionSegment]?)
    
    func addWordTimestamps(
        segments: [TranscriptionSegment],
        alignmentWeights: MLMultiArray,
        tokenizer: WhisperTokenizer,
        seek: Int,
        segmentSize: Int,
        prependPunctuations: String,
        appendPunctuations: String,
        lastSpeechTimestamp: Float,
        options: DecodingOptions,
        timings: TranscriptionTimings
    ) throws -> [TranscriptionSegment]?
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public struct SegmentSeeker: SegmentSeeking {
    public init() {}
    
    // MARK: - Seek & Segments
    
    public func findSeekPointAndSegments(
        decodingResult: DecodingResult,
        options: DecodingOptions,
        allSegmentsCount: Int,
        currentSeek: Int,
        segmentSize: Int,
        sampleRate: Int,
        timeToken: Int,
        specialToken: Int,
        tokenizer: WhisperTokenizer
    ) -> (Int, [TranscriptionSegment]?) {
        // Check if we need to skip this segment entirely
        var seek = currentSeek
        let timeOffset = Float(seek) / Float(sampleRate)
        let secondsPerTimeToken = WhisperKit.secondsPerTimeToken
        
        if let threshold = options.noSpeechThreshold {
            var shouldSkip = decodingResult.noSpeechProb > threshold
            
            if let logProbThreshold = options.logProbThreshold,
               decodingResult.avgLogProb > logProbThreshold {
                // Confidence in overall segment overrides no speech threshold
                shouldSkip = false
            }
            
            if shouldSkip {
                // Skip one full segment, this one is silent
                seek += segmentSize
                return (seek, nil)
            }
        }
        
        var currentSegments: [TranscriptionSegment] = []
        
        // Process tokens to identify timestamps and create segments
        let currentTokens = decodingResult.tokens
        let currentLogProbs = decodingResult.tokenLogProbs
        let isTimestampToken = currentTokens.map { $0 >= timeToken }
        
        // Find all end indexes of time token pairs
        var sliceIndexes = [Int]()
        var previousTokenIsTimestamp = false
        for (currentIndex, currentTokenIsTimestamp) in isTimestampToken.enumerated() {
            if previousTokenIsTimestamp && currentTokenIsTimestamp {
                sliceIndexes.append(currentIndex)
            }
            previousTokenIsTimestamp = currentTokenIsTimestamp
        }
        
        // Optimize handling of timestamp endings
        if !sliceIndexes.isEmpty {
            let lastTimestampIndex = isTimestampToken.lastIndex(of: true) ?? currentTokens.count - 1
            sliceIndexes.append(lastTimestampIndex + 1)
            
            var lastSliceStart = 0
            for currentSliceEnd in sliceIndexes {
                let slicedTokens = Array(currentTokens[lastSliceStart..<currentSliceEnd])
                let slicedTokenLogProbs = Array(currentLogProbs[lastSliceStart..<currentSliceEnd])
                let timestampTokens = slicedTokens.filter { $0 >= timeToken }
                
                guard let firstTimestamp = timestampTokens.first,
                      let lastTimestamp = timestampTokens.last else { continue }
                
                let startTimestampSeconds = Float(firstTimestamp - timeToken) * secondsPerTimeToken
                let endTimestampSeconds = Float(lastTimestamp - timeToken) * secondsPerTimeToken
                
                // Decode segment text
                let wordTokens = slicedTokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                let slicedTextTokens = options.skipSpecialTokens ? wordTokens : slicedTokens
                let sliceText = tokenizer.decode(tokens: slicedTextTokens)
                
                let newSegment = TranscriptionSegment(
                    id: allSegmentsCount + currentSegments.count,
                    seek: seek,
                    start: timeOffset + startTimestampSeconds,
                    end: timeOffset + endTimestampSeconds,
                    text: sliceText,
                    tokens: slicedTokens,
                    tokenLogProbs: slicedTokenLogProbs,
                    temperature: decodingResult.temperature,
                    avgLogprob: decodingResult.avgLogProb,
                    compressionRatio: decodingResult.compressionRatio,
                    noSpeechProb: decodingResult.noSpeechProb
                )
                currentSegments.append(newSegment)
                lastSliceStart = currentSliceEnd
            }
            
            // Seek to the last timestamp in the segment
            if let lastTimestampToken = currentTokens[lastSliceStart - 1] - timeToken as Int? {
                let lastTimestampSeconds = Float(lastTimestampToken) * secondsPerTimeToken
                let lastTimestampSamples = Int(lastTimestampSeconds * Float(sampleRate))
                seek += lastTimestampSamples
            } else {
                seek += segmentSize
            }
        } else {
            // Handle case with no consecutive timestamps
            let durationSeconds: Float
            if let lastTimestamp = currentTokens.last(where: { $0 > timeToken }) {
                durationSeconds = Float(lastTimestamp - timeToken) * secondsPerTimeToken
            } else {
                durationSeconds = Float(segmentSize) / Float(sampleRate)
            }
            
            // Decode segment text
            let wordTokens = decodingResult.tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
            let segmentTextTokens = options.skipSpecialTokens ? wordTokens : decodingResult.tokens
            let segmentText = tokenizer.decode(tokens: segmentTextTokens)
            
            let newSegment = TranscriptionSegment(
                id: allSegmentsCount + currentSegments.count,
                seek: seek,
                start: timeOffset,
                end: timeOffset + durationSeconds,
                text: segmentText,
                tokens: decodingResult.tokens,
                tokenLogProbs: decodingResult.tokenLogProbs,
                temperature: decodingResult.temperature,
                avgLogprob: decodingResult.avgLogProb,
                compressionRatio: decodingResult.compressionRatio,
                noSpeechProb: decodingResult.noSpeechProb
            )
            currentSegments.append(newSegment)
            
            seek += segmentSize
        }
        
        return (seek, currentSegments)
    }
    
    // MARK: - Word Timestamps
    
    func dynamicTimeWarping(withMatrix matrix: MLMultiArray) throws -> (textIndices: [Int], timeIndices: [Int]) {
        guard matrix.shape.count == 2,
              let numberOfRows = matrix.shape[0] as? Int,
              let numberOfColumns = matrix.shape[1] as? Int else {
            throw WhisperError.segmentingFailed("Invalid alignment matrix shape")
        }
        
        // Flatten the MLMultiArray to a 1D array for Accelerate functions
        let matrixData = Array(UnsafeBufferPointer(start: matrix.dataPointer.assumingMemoryBound(to: Double.self), count: numberOfRows * numberOfColumns))
        
        // Prepare cost matrix and direction matrix
        var costMatrix = [Double](repeating: .infinity, count: (numberOfRows + 1) * (numberOfColumns + 1))
        var directionMatrix = [Int](repeating: -1, count: (numberOfRows + 1) * (numberOfColumns + 1))
        
        costMatrix[0] = 0
        for i in 1...numberOfColumns {
            directionMatrix[i] = 2
        }
        for i in 1...numberOfRows {
            directionMatrix[i * (numberOfColumns + 1)] = 1
        }
        
        // Perform DTW using optimized loops
        for row in 1...numberOfRows {
            for column in 1...numberOfColumns {
                let matrixValue = -matrixData[(row - 1) * numberOfColumns + (column - 1)]
                let index = row * (numberOfColumns + 1) + column
                let costDiagonal = costMatrix[(row - 1) * (numberOfColumns + 1) + (column - 1)]
                let costUp = costMatrix[(row - 1) * (numberOfColumns + 1) + column]
                let costLeft = costMatrix[row * (numberOfColumns + 1) + (column - 1)]
                
                let (computedCost, traceValue) = minCostAndTrace(
                    costDiagonal: costDiagonal,
                    costUp: costUp,
                    costLeft: costLeft,
                    matrixValue: matrixValue
                )
                
                costMatrix[index] = computedCost
                directionMatrix[index] = traceValue
            }
        }
        
        let dtw = backtrace(fromDirectionMatrix: directionMatrix, numberOfRows: numberOfRows, numberOfColumns: numberOfColumns)
        
        return dtw
    }
    
    func minCostAndTrace(costDiagonal: Double, costUp: Double, costLeft: Double, matrixValue: Double) -> (Double, Int) {
        let c0 = costDiagonal + matrixValue
        let c1 = costUp + matrixValue
        let c2 = costLeft + matrixValue
        
        if c0 <= c1 && c0 <= c2 {
            return (c0, 0)
        } else if c1 <= c0 && c1 <= c2 {
            return (c1, 1)
        } else {
            return (c2, 2)
        }
    }
    
    func backtrace(fromDirectionMatrix directionMatrix: [Int], numberOfRows: Int, numberOfColumns: Int) -> (textIndices: [Int], timeIndices: [Int]) {
        var i = numberOfRows
        var j = numberOfColumns
        
        var textIndices = [Int]()
        var timeIndices = [Int]()
        
        let width = numberOfColumns + 1
        
        while i > 0 || j > 0 {
            textIndices.append(i - 1)
            timeIndices.append(j - 1)
            
            let dir = directionMatrix[i * width + j]
            switch dir {
            case 0:
                i -= 1
                j -= 1
            case 1:
                i -= 1
            case 2:
                j -= 1
            default:
                break
            }
        }
        
        return (textIndices.reversed(), timeIndices.reversed())
    }
    
    func mergePunctuations(alignment: [WordTiming], prepended: String, appended: String) -> [WordTiming] {
        var mergedAlignment = [WordTiming]()
        
        for timing in alignment {
            let trimmedWord = timing.word.trimmingCharacters(in: .whitespaces)
            if prepended.contains(trimmedWord), let last = mergedAlignment.last {
                // Merge with previous word
                var updatedLast = last
                updatedLast.word += timing.word
                updatedLast.tokens += timing.tokens
                mergedAlignment[mergedAlignment.count - 1] = updatedLast
            } else if appended.contains(trimmedWord), var last = mergedAlignment.last {
                // Merge with previous word
                last.word += timing.word
                last.tokens += timing.tokens
                mergedAlignment[mergedAlignment.count - 1] = last
            } else {
                mergedAlignment.append(timing)
            }
        }
        
        return mergedAlignment
    }
    
    func findAlignment(
        wordTokenIds: [Int],
        alignmentWeights: MLMultiArray,
        tokenLogProbs: [Float],
        tokenizer: WhisperTokenizer,
        timings: TranscriptionTimings? = nil
    ) throws -> [WordTiming] {
        let (textIndices, timeIndices) = try dynamicTimeWarping(withMatrix: alignmentWeights)
        let (words, wordTokens) = tokenizer.splitToWordTokens(tokenIds: wordTokenIds)
        
        if wordTokens.count <= 1 {
            return []
        }
        
        // Calculate start times and end times
        var startTimes = [Float](repeating: 0.0, count: wordTokens.count)
        var endTimes = [Float](repeating: 0.0, count: wordTokens.count)
        
        var tokenIndex = 0
        var wordIndex = 0
        
        let secondsPerTimeToken = Float(WhisperKit.secondsPerTimeToken)
        
        while tokenIndex < textIndices.count && wordIndex < wordTokens.count {
            let startTime = Float(timeIndices[tokenIndex]) * secondsPerTimeToken
            startTimes[wordIndex] = startTime
            
            let wordTokenCount = wordTokens[wordIndex].count
            tokenIndex += wordTokenCount - 1
            
            let endTime = Float(timeIndices[min(tokenIndex, timeIndices.count - 1)]) * secondsPerTimeToken
            endTimes[wordIndex] = endTime
            
            tokenIndex += 1
            wordIndex += 1
        }
        
        var wordTimings = [WordTiming]()
        for (index, word) in words.enumerated() {
            let probability = tokenLogProbs[index].rounded(toPlaces: 2)
            let wordTiming = WordTiming(
                word: word,
                tokens: wordTokens[index],
                start: startTimes[index],
                end: endTimes[index],
                probability: pow(10, probability)
            )
            wordTimings.append(wordTiming)
        }
        
        return wordTimings
    }
    
    public func addWordTimestamps(
        segments: [TranscriptionSegment],
        alignmentWeights: MLMultiArray,
        tokenizer: WhisperTokenizer,
        seek: Int,
        segmentSize: Int,
        prependPunctuations: String,
        appendPunctuations: String,
        lastSpeechTimestamp: Float,
        options: DecodingOptions,
        timings: TranscriptionTimings
    ) throws -> [TranscriptionSegment]? {
        // Prepare data for alignment
        var wordTokenIds = [Int]()
        var filteredLogProbs = [Float]()
        var filteredIndices = [Int]()
        
        var indexOffset = 0
        for segment in segments {
            for (index, token) in segment.tokens.enumerated() {
                wordTokenIds.append(token)
                filteredIndices.append(index + indexOffset)
                if let logProb = segment.tokenLogProbs[index][token] {
                    filteredLogProbs.append(logProb)
                }
            }
            indexOffset += segment.tokens.count
        }
        
        // Efficiently filter alignmentWeights using filteredIndices
        let shape = alignmentWeights.shape
        guard let columnCount = shape.last?.intValue else {
            throw WhisperError.segmentingFailed("Invalid shape in alignmentWeights")
        }
        
        let filteredAlignmentWeights = try filterAlignmentWeights(
            alignmentWeights: alignmentWeights,
            filteredIndices: filteredIndices,
            rowCount: filteredIndices.count,
            columnCount: columnCount
        )
        
        let alignment = try findAlignment(
            wordTokenIds: wordTokenIds,
            alignmentWeights: filteredAlignmentWeights,
            tokenLogProbs: filteredLogProbs,
            tokenizer: tokenizer,
            timings: timings
        )
        
        // Process alignment for punctuations
        let mergedAlignment = mergePunctuations(alignment: alignment, prepended: prependPunctuations, appended: appendPunctuations)
        
        var wordIndex = 0
        let timeOffset = Float(seek) / Float(WhisperKit.sampleRate)
        var updatedSegments = [TranscriptionSegment]()
        
        for segment in segments {
            var savedTokens = 0
            let textTokens = segment.tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
            var wordsInSegment = [WordTiming]()
            
            while wordIndex < mergedAlignment.count && savedTokens < textTokens.count {
                let timing = mergedAlignment[wordIndex]
                wordIndex += 1
                
                let timingTokens = timing.tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                if timingTokens.isEmpty {
                    continue
                }
                
                let start = (timeOffset + timing.start).rounded(toPlaces: 2)
                let end = (timeOffset + timing.end).rounded(toPlaces: 2)
                let probability = timing.probability.rounded(toPlaces: 2)
                let wordTiming = WordTiming(word: timing.word,
                                            tokens: timingTokens,
                                            start: start,
                                            end: end,
                                            probability: probability)
                wordsInSegment.append(wordTiming)
                
                savedTokens += timingTokens.count
            }
            
            // Create an updated segment with the word timings
            var updatedSegment = segment
            updatedSegment.words = wordsInSegment
            updatedSegments.append(updatedSegment)
        }
        
        return updatedSegments
    }
    
    private func filterAlignmentWeights(
        alignmentWeights: MLMultiArray,
        filteredIndices: [Int],
        rowCount: Int,
        columnCount: Int
    ) throws -> MLMultiArray {
        let filteredAlignmentWeights = try MLMultiArray(shape: [rowCount, columnCount] as [NSNumber], dataType: .double)
        
        let sourcePointer = alignmentWeights.dataPointer.assumingMemoryBound(to: Double.self)
        let destinationPointer = filteredAlignmentWeights.dataPointer.assumingMemoryBound(to: Double.self)
        
        for (newIndex, originalIndex) in filteredIndices.enumerated() {
            let sourceRow = sourcePointer.advanced(by: originalIndex * columnCount)
            let destinationRow = destinationPointer.advanced(by: newIndex * columnCount)
            destinationRow.update(from: sourceRow, count: columnCount)
        }
        
        return filteredAlignmentWeights
    }
}

extension Float {
    func rounded(toPlaces places: Int) -> Float {
        let divisor = pow(10, Float(places))
        return (self * divisor).rounded() / divisor
    }
}
