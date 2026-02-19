//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
@testable import WhisperKit

final class WorkerPoolTestSegmentSeekerMock: SegmentSeeking {
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
    ) -> (Int, [TranscriptionSegment]?) {
        let segment = TranscriptionSegment(
            id: allSegmentsCount,
            seek: seek,
            start: Float(seek) / Float(sampleRate),
            end: Float(seek + segmentSize) / Float(sampleRate),
            text: decodingResult.text,
            tokens: decodingResult.tokens,
            tokenLogProbs: decodingResult.tokenLogProbs,
            temperature: decodingResult.temperature,
            avgLogprob: decodingResult.avgLogProb,
            compressionRatio: decodingResult.compressionRatio,
            noSpeechProb: decodingResult.noSpeechProb
        )
        return (seek + segmentSize, [segment])
    }

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
    ) throws -> [TranscriptionSegment]? {
        segments
    }
}
