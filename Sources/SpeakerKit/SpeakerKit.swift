//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import WhisperKit
import ArgmaxCore

// MARK: - SpeakerKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class SpeakerKit: @unchecked Sendable {
    private var _diarizer: SpeakerKitDiarizer?

    /// Creates SpeakerKit with pre-loaded Pyannote models.
    /// - Parameter models: Loaded segmenter and embedder models plus config.
    public init(models: PyannoteModels) throws {
        let diarizerConfig = DiarizerConfig(
            segmenterModel: models.segmenter,
            embedderModel: models.embedder,
            clusterer: VBxClustering(),
            verbose: models.config.verbose,
            concurrentEmbedderWorkers: models.config.concurrentEmbedderWorkers,
            models: models
        )
        self._diarizer = PyannoteDiarizer(config: diarizerConfig)
    }

    /// Loads (and optionally downloads) Pyannote models and creates SpeakerKit.
    /// - Parameter config: Model paths, download/repo settings, and runtime options.
    public convenience init(_ config: PyannoteConfig) async throws {
        let manager = SpeakerKitModelManager(config: config)
        if config.download {
            try await manager.downloadModels()
        }
        try await manager.loadModels()
        guard let models = manager.models as? PyannoteModels else {
            throw SpeakerKitError.modelUnavailable("Failed to load SpeakerKit models")
        }
        try self.init(models: models)
    }

    /// For subclasses that provide their own backend call this with their diarizer.
    /// - Parameter diarizer: Speaker diarizer to use with SpeakerKit.
    public init(diarizer: SpeakerKitDiarizer?) {
        self._diarizer = diarizer
    }

    // MARK: - Subclass hooks

    /// Override or use from subclasses to access the active diarizer backend.
    open var diarizer: SpeakerKitDiarizer? { _diarizer }

    // MARK: - Diarization

    /// Unloads segmenter and embedder and clears the diarizer.
    public func unloadModels() async {
        await _diarizer?.unloadModels()
        _diarizer = nil
    }

    /// Processes audio and returns labeled speaker segments.
    /// - Parameters:
    ///   - audioArray: 16 kHz mono PCM samples to diarize.
    ///   - options: Diarization options. Nil uses defaults.
    ///   - progressCallback: Optional callback for progress updates.
    /// - Returns: Labeled speaker segments with timings.
    public func diarize(
        audioArray: [Float],
        options: (any DiarizationOptionsProtocol)? = nil,
        progressCallback: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> DiarizationResult {
        guard let diarizer = _diarizer else {
            throw SpeakerKitError.invalidConfiguration("Diarizer is not initialized")
        }
        return try await diarizer.diarize(audioArray: audioArray, options: options, progressCallback: progressCallback)
    }

    /// Builds RTTM lines from a diarization result, optionally aligned to a transcription.
    /// - Parameters:
    ///   - diarizationResult: Result from `diarize(audioArray:options:)`.
    ///   - strategy: How to assign speaker info to words (e.g. `.subsegment`).
    ///   - transcription: Optional word-level transcription to align speakers to words.
    ///   - fileName: File ID used in the RTTM output (default `"audio"`).
    /// - Returns: RTTM lines ready to write or print.
    open class func generateRTTM(
        from diarizationResult: DiarizationResult,
        strategy: SpeakerInfoStrategy = .subsegment,
        transcription: [TranscriptionResult]? = nil,
        fileName: String = "audio"
    ) -> [RTTMLine] {
        if let transcription = transcription {
            let segments = diarizationResult.addSpeakerInfo(to: transcription, strategy: strategy)
            let wordsWithSpeakers = segments.flatMap { segmentGroup in
                segmentGroup.flatMap { segment in
                    segment.speakerWords.map { word in
                        WordWithSpeaker(wordTiming: word.wordTiming, speaker: word.speaker.speakerId)
                    }
                }
            }
            return RTTMLine.fromWords(wordsWithSpeakers, fileName: fileName)
        } else {
            var noOffsetResult = diarizationResult
            noOffsetResult.updateSegments(minActiveOffset: 0.0)
            return noOffsetResult.segments.map { segment in
                return RTTMLine(
                    fileId: fileName,
                    speakerId: segment.speaker.speakerId ?? -1,
                    startTime: segment.startTime,
                    duration: segment.endTime - segment.startTime
                )
            }
        }
    }
}

// MARK: - Error

public enum SpeakerKitError: Error, LocalizedError {
    case modelUnavailable(String)
    case invalidConfiguration(String)
    case invalidModelOutput(String)
    case generic(String)

    public var errorDescription: String? {
        switch self {
        case .modelUnavailable(let msg),
             .invalidConfiguration(let msg),
             .invalidModelOutput(let msg),
             .generic(let msg):
            return msg
        }
    }
}

// MARK: - Diarizer Protocol

/// Shared diarizer interface so SpeakerKit can delegate to the active backend.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public protocol SpeakerKitDiarizer: AnyObject, Sendable {
    func unloadModels() async
    func diarize(audioArray: [Float], options: (any DiarizationOptionsProtocol)?, progressCallback: (@Sendable (Progress) -> Void)?) async throws -> DiarizationResult
}
