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
open class SegmentSeeker: SegmentSeeking {
    public init() {}

    // MARK: - Seek & Segments

    // TODO: simplify this interface
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
        // check if we need to skip this segment entirely
        // if so, reset currentSegments, continue to next window, otherwise:
        var seek = currentSeek
        let timeOffset = Float(seek) / Float(sampleRate)
        let secondsPerTimeToken = WhisperKit.secondsPerTimeToken
        if let threshold = options.noSpeechThreshold {
            // check no speech threshold for segment
            var shouldSkip = decodingResult.noSpeechProb > threshold

            // check avg logprob threshold for segment
            if let logProbThreshold = options.logProbThreshold,
               decodingResult.avgLogProb > logProbThreshold
            {
                // Confidence in overall segment overrides no speech threshold
                shouldSkip = false
            }

            if shouldSkip {
                // skip one full segment, this one is silent
                seek += segmentSize
                return (seek, nil)
            }
        }

        var currentSegments: [TranscriptionSegment] = []

        // loop through all consecutive timestamps and turn them into `TranscriptionSegments`
        let currentTokens = decodingResult.tokens
        let currentLogProbs = decodingResult.tokenLogProbs
        let isTimestampToken = currentTokens.map { $0 >= timeToken }

        // check if single or double timestamp ending
        let lastThreeTokens = isTimestampToken.suffix(3)
        let singleTimestampEnding = lastThreeTokens == [false, true, false]
        let noTimestampEnding = lastThreeTokens == [false, false, false]

        // find all end indexes of time token pairs
        var sliceIndexes = [Int]()

        var previousTokenIsTimestamp = false
        for (currentTokenIsTimestampIndex, currentTokenIsTimestamp) in isTimestampToken.enumerated() {
            if previousTokenIsTimestamp && currentTokenIsTimestamp {
                sliceIndexes.append(currentTokenIsTimestampIndex)
            }
            previousTokenIsTimestamp = currentTokenIsTimestamp
        }

        // Window contains multiple consecutive timestamps, split into sub-segments
        if !sliceIndexes.isEmpty {
            // If the last timestamp is not consecutive, we need to add it as the final slice manually
            if singleTimestampEnding {
                let singleTimestampEndingIndex = isTimestampToken.lastIndex(where: { $0 })!
                sliceIndexes.append(singleTimestampEndingIndex + 1)
            } else if noTimestampEnding {
                sliceIndexes.append(currentTokens.count)
            }

            var lastSliceStart = 0
            for currentSliceEnd in sliceIndexes {
                let slicedTokens = Array(currentTokens[lastSliceStart..<currentSliceEnd])
                let slicedTokenLogProbs = Array(currentLogProbs[lastSliceStart..<currentSliceEnd])
                let timestampTokens = slicedTokens.filter { $0 >= timeToken }

                let startTimestampSeconds = Float(timestampTokens.first! - timeToken) * secondsPerTimeToken
                let endTimestampSeconds = Float(timestampTokens.last! - timeToken) * secondsPerTimeToken

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
            if !noTimestampEnding {
                let lastTimestampToken = currentTokens[lastSliceStart - (singleTimestampEnding ? 1 : 0)] - timeToken
                let lastTimestampSeconds = Float(lastTimestampToken) * secondsPerTimeToken
                let lastTimestampSamples = Int(lastTimestampSeconds * Float(sampleRate))
                seek += lastTimestampSamples
            } else {
                seek += segmentSize
            }
        } else {
            // Model is not giving any consecutive timestamps, so lump all the current tokens together
            var durationSeconds = Float(segmentSize) / Float(sampleRate)

            // Find any timestamp that is not 0.00
            let timestampTokens = currentTokens.filter { $0 > timeToken }

            // If there are no consecutive timestamps at all, check if there is at least one timestamp at the end
            // If there is at least one, use that to record a more accurate end time
            if !timestampTokens.isEmpty, let lastTimestamp = timestampTokens.last {
                durationSeconds = Float(lastTimestamp - timeToken) * secondsPerTimeToken
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

            // Model has told us there is no more speech in this segment, move on to next
            seek += segmentSize
            // TODO: use this logic instead once we handle no speech
            // seek += Int(durationSeconds * Float(sampleRate))
        }

        return (seek, currentSegments)
    }

    // MARK: - Word Timestamps

    /// Matrix is a 2D array of alignment weights of shape (n, m) where n is the number of rows representing text tokens
    /// and m is the number of columns representing audio tokens
    func dynamicTimeWarping(withMatrix matrix: MLMultiArray) throws -> (textIndices: [Int], timeIndices: [Int]) {
        guard matrix.shape.count == 2,
              let numberOfRows = matrix.shape[0] as? Int,
              let numberOfColumns = matrix.shape[1] as? Int
        else {
            throw WhisperError.segmentingFailed("Invalid alignment matrix shape")
        }

        // Initialize cost matrix and trace matrix
        var costMatrix = Array(repeating: Array(repeating: Double.infinity, count: numberOfColumns + 1), count: numberOfRows + 1)
        var traceMatrix = Array(repeating: Array(repeating: -1, count: numberOfColumns + 1), count: numberOfRows + 1)

        costMatrix[0][0] = 0
        for i in 1...numberOfColumns {
            traceMatrix[0][i] = 2
        }
        for i in 1...numberOfRows {
            traceMatrix[i][0] = 1
        }

        for row in 1...numberOfRows {
            for column in 1...numberOfColumns {
                let matrixValue = -matrix[(row - 1) * numberOfColumns + (column - 1)].doubleValue
                let costDiagonal = costMatrix[row - 1][column - 1]
                let costUp = costMatrix[row - 1][column]
                let costLeft = costMatrix[row][column - 1]

                let (computedCost, traceValue) = minCostAndTrace(
                    costDiagonal: costDiagonal,
                    costUp: costUp,
                    costLeft: costLeft,
                    matrixValue: matrixValue
                )

                costMatrix[row][column] = computedCost
                traceMatrix[row][column] = traceValue
            }
        }

        let dtw = backtrace(fromTraceMatrix: traceMatrix)

        return dtw
    }

    func minCostAndTrace(costDiagonal: Double, costUp: Double, costLeft: Double, matrixValue: Double) -> (Double, Int) {
        let c0 = costDiagonal + matrixValue
        let c1 = costUp + matrixValue
        let c2 = costLeft + matrixValue

        if c0 < c1 && c0 < c2 {
            return (c0, 0)
        } else if c1 < c0 && c1 < c2 {
            return (c1, 1)
        } else {
            return (c2, 2)
        }
    }

    func backtrace(fromTraceMatrix traceMatrix: [[Int]]) -> (textIndices: [Int], timeIndices: [Int]) {
        var i = traceMatrix.count - 1
        var j = traceMatrix[0].count - 1

        var textIndices = [Int]()
        var timeIndices = [Int]()

        while i > 0 || j > 0 {
            textIndices.append(i - 1)
            timeIndices.append(j - 1)

            switch traceMatrix[i][j] {
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
        var prependedAlignment = [WordTiming]()
        var appendedAlignment = [WordTiming]()

        // Include the first word if it's not a prepended punctuation
        if !alignment.isEmpty && !prepended.contains(alignment[0].word.trimmingCharacters(in: .whitespaces)) {
            prependedAlignment.append(alignment[0])
        }

        // Merge prepended punctuations
        for i in 1..<alignment.count {
            var currentWord = alignment[i]
            let previousWord = alignment[i - 1]
            // Check if the previous word starts with a whitespace character and is part of the prepended punctuations
            if let firstChar = previousWord.word.unicodeScalars.first,
               CharacterSet.whitespaces.contains(firstChar),
               prepended.contains(previousWord.word.trimmingCharacters(in: .whitespaces))
            {
                currentWord.word = previousWord.word + currentWord.word
                currentWord.tokens = previousWord.tokens + currentWord.tokens
                prependedAlignment[prependedAlignment.count - 1] = currentWord
            } else {
                prependedAlignment.append(currentWord)
            }
        }

        // Include the first word always for append checks
        if !prependedAlignment.isEmpty {
            appendedAlignment.append(prependedAlignment[0])
        }

        // Merge appended punctuations
        for i in 1..<prependedAlignment.count {
            let currentWord = prependedAlignment[i]
            var previousWord = prependedAlignment[i - 1]
            if !previousWord.word.hasSuffix(" "), appended.contains(currentWord.word.trimmingCharacters(in: .whitespaces)) {
                previousWord.word = previousWord.word + currentWord.word
                previousWord.tokens = previousWord.tokens + currentWord.tokens
                appendedAlignment[appendedAlignment.count - 1] = previousWord
            } else {
                appendedAlignment.append(currentWord)
            }
        }

        // Filter out the empty word timings and punctuation words that have been merged
        let mergedAlignment = appendedAlignment.filter { !$0.word.isEmpty && !appended.contains($0.word) && !prepended.contains($0.word) }
        return mergedAlignment
    }

    func findAlignment(
        wordTokenIds: [Int],
        alignmentWeights: MLMultiArray,
        tokenLogProbs: [Float],
        tokenizer: WhisperTokenizer,
        timings: TranscriptionTimings? = nil
    ) throws -> [WordTiming] {
        // TODO: Use accelerate framework for these two, they take roughly the same time
        let (textIndices, timeIndices) = try dynamicTimeWarping(withMatrix: alignmentWeights)
        let (words, wordTokens) = tokenizer.splitToWordTokens(tokenIds: wordTokenIds)

        if wordTokens.count <= 1 {
            return []
        }

        // Calculate start times and end times
        var startTimes: [Float] = [0.0]
        var endTimes = [Float]()
        var currentTokenIndex = textIndices.first ?? 0
        for index in 0..<textIndices.count {
            // Check if the current token index is different from the previous one
            if textIndices[index] != currentTokenIndex {
                // We found a new token, so calculate the time for this token and add it to the list
                currentTokenIndex = textIndices[index]

                let time = Float(timeIndices[index]) * Float(WhisperKit.secondsPerTimeToken)

                startTimes.append(time)
                endTimes.append(time)
            }
        }
        endTimes.append(Float(timeIndices.last ?? 1500) * Float(WhisperKit.secondsPerTimeToken))

        var wordTimings = [WordTiming]()
        currentTokenIndex = 0

        for (index, wordTokenArray) in wordTokens.enumerated() {
            let wordStartTime: Float
            let wordEndTime: Float
            let startIndex = currentTokenIndex

            // Get the start time of the first token in the current word
            wordStartTime = startTimes[currentTokenIndex]

            // Update the currentTokenIndex to the end of the current wordTokenArray
            currentTokenIndex += wordTokenArray.count - 1

            // Get the end time of the last token in the current word
            wordEndTime = endTimes[currentTokenIndex]

            // Move the currentTokenIndex to the next token for the next iteration
            currentTokenIndex += 1

            // Calculate the probability
            let probs = tokenLogProbs[startIndex..<currentTokenIndex]
            let probability = probs.reduce(0, +) / Float(probs.count)

            let wordTiming = WordTiming(
                word: words[index],
                tokens: wordTokenArray,
                start: wordStartTime,
                end: wordEndTime,
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
        // Initialize arrays to hold the extracted and filtered data
        var wordTokenIds = [Int]()
        var filteredLogProbs = [Float]()
        var filteredIndices = [Int]()
        var lastSpeechTimestamp = lastSpeechTimestamp

        // Iterate through each segment
        var indexOffset = 0
        for segment in segments {
            for (index, token) in segment.tokens.enumerated() {
                wordTokenIds.append(token)
                filteredIndices.append(index + indexOffset) // Add the index to filteredIndices

                // Assuming tokenLogProbs is structured as [[Int: Float]]
                if let logProb = segment.tokenLogProbs[index][token] {
                    filteredLogProbs.append(logProb)
                }
            }

            // Update the indexOffset as we start a new segment
            indexOffset += segment.tokens.count
        }

        // Filter alignmentWeights using filteredIndices
        let shape = alignmentWeights.shape
        guard let columnCount = shape.last?.intValue else {
            throw WhisperError.segmentingFailed("Invalid shape in alignmentWeights")
        }

        let filteredAlignmentWeights = initMLMultiArray(shape: [filteredIndices.count, columnCount] as [NSNumber], dataType: alignmentWeights.dataType, initialValue: FloatType(0))

        alignmentWeights.withUnsafeMutableBytes { weightsPointer, weightsStride in
            filteredAlignmentWeights.withUnsafeMutableBytes { filteredWeightsPointer, filteredWeightsStride in
                for (newIndex, originalIndex) in filteredIndices.enumerated() {
                    let sourcePointer = weightsPointer.baseAddress!.advanced(by: Int(originalIndex * columnCount * MemoryLayout<FloatType>.stride))
                    let destinationPointer = filteredWeightsPointer.baseAddress!.advanced(by: Int(newIndex * columnCount * MemoryLayout<FloatType>.stride))

                    memcpy(destinationPointer, sourcePointer, columnCount * MemoryLayout<FloatType>.stride)
                }
            }
        }

        Logging.debug("Alignment weights shape: \(filteredAlignmentWeights.shape)")

        var alignment = try findAlignment(
            wordTokenIds: wordTokenIds,
            alignmentWeights: filteredAlignmentWeights,
            tokenLogProbs: filteredLogProbs,
            tokenizer: tokenizer,
            timings: timings
        )

        // TODO: This section is considered a "hack" in the source repo
        // Reference: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/timing.py#L305
        var wordDurations = alignment.map { $0.end - $0.start }
        wordDurations = wordDurations.filter { $0 > 0 }

        let medianDuration: Float = wordDurations.isEmpty ? 0.0 : wordDurations.sorted(by: <)[wordDurations.count / 2]
        let constrainedMedianDuration = min(0.7, medianDuration)
        let maxDuration = constrainedMedianDuration * 2

        // Truncate long words at sentence boundaries
        let sentenceEndMarks = [".", "。", "!", "！", "?", "？"]
        if !wordDurations.isEmpty {
            for i in 1..<alignment.count {
                if alignment[i].end - alignment[i].start > maxDuration {
                    if sentenceEndMarks.contains(alignment[i].word) {
                        alignment[i].end = alignment[i].start + maxDuration
                    } else if i > 0, sentenceEndMarks.contains(alignment[i - 1].word) {
                        alignment[i].start = alignment[i].end - maxDuration
                    }
                }
            }
        }

        // Process alignment for punctuations
        let mergedAlignment = mergePunctuations(alignment: alignment, prepended: prependPunctuations, appended: appendPunctuations)

        var wordIndex = 0
        let timeOffset = Float(seek) / Float(WhisperKit.sampleRate)
        var updatedSegments = [TranscriptionSegment]()

        for segment in segments {
            var savedTokens = 0
            let textTokens = segment.tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
            var wordsInSegment = [WordTiming]()

            for timing in mergedAlignment[wordIndex...] where savedTokens < textTokens.count {
                wordIndex += 1

                // Remove special tokens and retokenize if needed
                let timingTokens = timing.tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                if timingTokens.isEmpty {
                    continue
                }

                let start = (timeOffset + timing.start).rounded(2)
                let end = (timeOffset + timing.end).rounded(2)
                let probability = timing.probability.rounded(2)
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

            // TODO: This section is considered a "hack" in the source repo
            // Reference: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/timing.py#L342
            // Truncate long words at segment boundaries
            if let firstWord = wordsInSegment.first, let lastWord = wordsInSegment.last {
                // Logic for the first word
                if firstWord.end - lastSpeechTimestamp > constrainedMedianDuration * 4 &&
                    (firstWord.end - firstWord.start > maxDuration ||
                        (wordsInSegment.count > 1 && wordsInSegment[1].end - firstWord.start > maxDuration * 2))
                {
                    if wordsInSegment.count > 1 && wordsInSegment[1].end - wordsInSegment[1].start > maxDuration {
                        let boundary = max(wordsInSegment[1].end / 2, wordsInSegment[1].end - maxDuration)
                        wordsInSegment[0].end = boundary
                        wordsInSegment[1].start = boundary
                    }
                    wordsInSegment[0].start = max(lastSpeechTimestamp, firstWord.end - maxDuration)
                }

                // Prefer segment-level start timestamp if the first word is too long.
                if segment.start < firstWord.end && segment.start - 0.5 > firstWord.start {
                    wordsInSegment[0].start = max(0, min(firstWord.end - constrainedMedianDuration, segment.start))
                } else {
                    updatedSegment.start = firstWord.start
                }

                // Prefer segment-level end timestamp if the last word is too long.
                if updatedSegment.end > lastWord.start && segment.end + 0.5 < lastWord.end {
                    wordsInSegment[wordsInSegment.count - 1].end = max(lastWord.start + constrainedMedianDuration, segment.end)
                } else {
                    updatedSegment.end = lastWord.end
                }

                lastSpeechTimestamp = updatedSegment.end
            }

            updatedSegment.words = wordsInSegment
            updatedSegments.append(updatedSegment)
        }

        return updatedSegments
    }
}
