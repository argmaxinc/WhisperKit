//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import Tokenizers

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
public protocol SegmentSeeking {
    func findSeekPointAndSegments(decodingResult: DecodingResult, options: DecodingOptions, allSegmentsCount: Int, currentSeek seek: Int, segmentSize: Int, sampleRate: Int, timeToken: Int, specialToken: Int, tokenizer: Tokenizer) -> (Int, [TranscriptionSegment]?)
}

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
public class SegmentSeeker: SegmentSeeking {
    public init() {}

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
        tokenizer: Tokenizer
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
        let isTimestampToken = currentTokens.map { $0 >= timeToken }

        // check if single or double timestamp ending
        let lastThreeTokens = isTimestampToken.suffix(3)
        let singleTimestampEnding = lastThreeTokens == [false, true, false]

        // find all indexes of time token pairs
        var consecutive = [(start: Int, end: Int)]()

        var previousTokenIsTimestamp = false
        for (i, tokenIsTimestamp) in isTimestampToken.enumerated() {
            if previousTokenIsTimestamp && tokenIsTimestamp {
                consecutive.append((i - 1, i))
            }
            previousTokenIsTimestamp = tokenIsTimestamp
        }

        if !consecutive.isEmpty {
            // Window contains multiple consecutive timestamps, split into sub-segments
            var sliceIndexes = consecutive.map { $0.end }

            // If the last timestamp is not consecutive, we need to add it as the final slice manually
            if singleTimestampEnding {
                let singleTimestampEndingIndex = isTimestampToken.lastIndex(where: { $0 })!
                sliceIndexes.append(singleTimestampEndingIndex)
            }

            var lastSliceStart = 0
            for currentSliceEnd in sliceIndexes {
                let slicedTokens = Array(currentTokens[lastSliceStart...currentSliceEnd])
                let timestampTokens = slicedTokens.filter { $0 >= timeToken }

                let startTimestampSeconds = Float(timestampTokens.first! - timeToken) * secondsPerTimeToken
                let endTimestampSeconds = Float(timestampTokens.last! - timeToken) * secondsPerTimeToken

                // Decode segment text
                let wordTokens = slicedTokens.filter { $0 < tokenizer.specialTokenBegin }
                let slicedTextTokens = options.skipSpecialTokens ? wordTokens : slicedTokens
                let sliceText = tokenizer.decode(tokens: slicedTextTokens)

                let newSegment = TranscriptionSegment(
                    id: allSegmentsCount + currentSegments.count,
                    seek: seek,
                    start: timeOffset + startTimestampSeconds,
                    end: timeOffset + endTimestampSeconds,
                    text: sliceText,
                    tokens: slicedTokens,
                    temperature: decodingResult.temperature,
                    avgLogprob: decodingResult.avgLogProb,
                    compressionRatio: decodingResult.compressionRatio,
                    noSpeechProb: decodingResult.noSpeechProb
                )
                currentSegments.append(newSegment)
                lastSliceStart = currentSliceEnd
            }

            // Seek to the last timestamp in the segment
            let lastTimestampToken = currentTokens[lastSliceStart] - timeToken
            let lastTimestampSeconds = Float(lastTimestampToken) * secondsPerTimeToken
            let lastTimestampSamples = Int(lastTimestampSeconds * Float(sampleRate))
            seek += lastTimestampSamples
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
            let wordTokens = decodingResult.tokens.filter { $0 < tokenizer.specialTokenBegin }
            let segmentTextTokens = options.skipSpecialTokens ? wordTokens : decodingResult.tokens
            let segmentText = tokenizer.decode(tokens: segmentTextTokens)

            let newSegment = TranscriptionSegment(
                id: allSegmentsCount + currentSegments.count,
                seek: seek,
                start: timeOffset,
                end: timeOffset + durationSeconds,
                text: segmentText,
                tokens: decodingResult.tokens,
                temperature: decodingResult.temperature,
                avgLogprob: decodingResult.avgLogProb,
                compressionRatio: decodingResult.compressionRatio,
                noSpeechProb: decodingResult.noSpeechProb
            )
            currentSegments.append(newSegment)

            // Model has told us there is no more speech in this segment, move on to next
            seek += segmentSize
            // TODO: use this logic instead once we handle no speech
//            seek += Int(durationSeconds * Float(sampleRate))
        }

        return (seek, currentSegments)
    }
}
