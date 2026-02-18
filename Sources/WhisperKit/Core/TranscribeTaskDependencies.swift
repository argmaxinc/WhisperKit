//
//  File 2.swift
//  whisperkit
//
//  Created by Aykut Güven on 18.02.26.
//

import Foundation

final class TranscribeTaskDependencies: @unchecked Sendable {
    let currentTimings: TranscriptionTimings
    let audioProcessor: any AudioProcessing
    let audioEncoder: any AudioEncoding
    let featureExtractor: any FeatureExtracting
    let segmentSeeker: any SegmentSeeking
    let textDecoder: any TextDecoding
    let tokenizer: any WhisperTokenizer
    let voiceActivityDetector: VoiceActivityDetector?

    init(
        currentTimings: TranscriptionTimings,
        audioProcessor: any AudioProcessing,
        audioEncoder: any AudioEncoding,
        featureExtractor: any FeatureExtracting,
        segmentSeeker: any SegmentSeeking,
        textDecoder: any TextDecoding,
        tokenizer: any WhisperTokenizer,
        voiceActivityDetector: VoiceActivityDetector?
    ) {
        self.currentTimings = currentTimings
        self.audioProcessor = audioProcessor
        self.audioEncoder = audioEncoder
        self.featureExtractor = featureExtractor
        self.segmentSeeker = segmentSeeker
        self.textDecoder = textDecoder
        self.tokenizer = tokenizer
        self.voiceActivityDetector = voiceActivityDetector
    }

    func makeTranscribeTask(progress: Progress) -> TranscribeTask {
        return TranscribeTask(
            currentTimings: currentTimings,
            progress: progress,
            audioProcessor: audioProcessor,
            audioEncoder: audioEncoder,
            featureExtractor: featureExtractor,
            segmentSeeker: segmentSeeker,
            textDecoder: textDecoder,
            tokenizer: tokenizer
        )
    }
}
