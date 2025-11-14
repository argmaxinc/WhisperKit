//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import WhisperKit
#if canImport(FluidAudio)
import FluidAudio
#endif

struct DiarizedTranscriptionPayload: Codable {
    let segments: [DiarizedSegment]
    let language: String
    let speakers: [DiarizedSpeaker]
}

struct DiarizedSegment: Codable {
    let start: Double
    let end: Double
    let text: String
    let speakerId: String
    let words: [DiarizedWord]
}

struct DiarizedWord: Codable {
    let start: Double
    let end: Double
    let word: String
}

struct DiarizedSpeaker: Codable {
    let id: String
    let name: String
}

#if canImport(FluidAudio)
@available(macOS 14.0, *)
final class DiarizationService {
    private let manager: OfflineDiarizerManager
    private let verbose: Bool
    private var isPrepared = false

    init(config: OfflineDiarizerConfig = .init(), verbose: Bool) {
        self.manager = OfflineDiarizerManager(config: config)
        self.verbose = verbose
    }

    func prepareModelsIfNeeded() async throws {
        guard !isPrepared else { return }
        if verbose {
            print("\nPreparing diarization models…")
        }
        try await manager.prepareModels()
        isPrepared = true
        if verbose {
            print("Diarization models ready.")
        }
    }

    func diarize(audioPath: String, transcription: TranscriptionResult) async throws -> DiarizedTranscriptionPayload {
        try await prepareModelsIfNeeded()
        if verbose {
            print("\nRunning diarization for \(audioPath)")
        }
        let audioURL = URL(fileURLWithPath: audioPath)
        let diarizationResult = try await manager.process(audioURL)
        return DiarizedTranscriptionPayload.build(
            from: transcription,
            diarizationSegments: diarizationResult.segments
        )
    }
}

extension DiarizedTranscriptionPayload {
    static func build(
        from transcription: TranscriptionResult,
        diarizationSegments: [TimedSpeakerSegment]
    ) -> DiarizedTranscriptionPayload {
        let speakerMapping = SpeakerMapping(diarizationSegments: diarizationSegments)
        var previousSpeakerId: String?
        let simplifiedSegments = transcription.segments.map { segment -> DiarizedSegment in
            let assignedSpeakerId = assignSpeaker(
                to: segment,
                diarizationSegments: diarizationSegments,
                speakerMapping: speakerMapping,
                previousSpeakerId: previousSpeakerId
            )
            previousSpeakerId = assignedSpeakerId

            let simplifiedWords = (segment.words ?? []).map {
                DiarizedWord(start: Double($0.start), end: Double($0.end), word: $0.word)
            }

            return DiarizedSegment(
                start: Double(segment.start),
                end: Double(segment.end),
                text: segment.text,
                speakerId: assignedSpeakerId,
                words: simplifiedWords
            )
        }

        return DiarizedTranscriptionPayload(
            segments: simplifiedSegments,
            language: transcription.language,
            speakers: speakerMapping.speakers
        )
    }

    private static func assignSpeaker(
        to segment: TranscriptionSegment,
        diarizationSegments: [TimedSpeakerSegment],
        speakerMapping: SpeakerMapping,
        previousSpeakerId: String?
    ) -> String {
        var bestCandidate: (id: String, overlap: Float)?

        for diarizationSegment in diarizationSegments {
            let overlap = overlapDuration(
                segmentStart: segment.start,
                segmentEnd: segment.end,
                diarizationStart: diarizationSegment.startTimeSeconds,
                diarizationEnd: diarizationSegment.endTimeSeconds
            )

            guard overlap > 0 else { continue }
            guard let sanitizedId = speakerMapping.sanitizedId(for: diarizationSegment.speakerId) else {
                continue
            }

            if let currentBest = bestCandidate {
                if overlap > currentBest.overlap {
                    bestCandidate = (sanitizedId, overlap)
                }
            } else {
                bestCandidate = (sanitizedId, overlap)
            }
        }

        if let bestCandidate {
            return bestCandidate.id
        }

        if let previousSpeakerId {
            return previousSpeakerId
        }

        return speakerMapping.fallbackSpeakerId
    }

    private static func overlapDuration(
        segmentStart: Float,
        segmentEnd: Float,
        diarizationStart: Float,
        diarizationEnd: Float
    ) -> Float {
        let start = max(segmentStart, diarizationStart)
        let end = min(segmentEnd, diarizationEnd)
        return max(0, end - start)
    }
}

private struct SpeakerMapping {
    private let rawToSanitized: [String: String]
    let speakers: [DiarizedSpeaker]
    let fallbackSpeakerId: String

    init(diarizationSegments: [TimedSpeakerSegment]) {
        var sanitized: [String: String] = [:]
        var orderedSpeakers: [DiarizedSpeaker] = []
        var seen = Set<String>()

        let orderedSegments = diarizationSegments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
        for segment in orderedSegments {
            let rawId = segment.speakerId
            guard !seen.contains(rawId) else { continue }

            let nextIndex = orderedSpeakers.count
            let sanitizedId = "speaker_\(nextIndex)"
            sanitized[rawId] = sanitizedId
            orderedSpeakers.append(
                DiarizedSpeaker(id: sanitizedId, name: "Speaker \(nextIndex)")
            )
            seen.insert(rawId)
        }

        if orderedSpeakers.isEmpty {
            orderedSpeakers = [DiarizedSpeaker(id: "speaker_0", name: "Speaker 0")]
        }

        self.rawToSanitized = sanitized
        self.speakers = orderedSpeakers
        self.fallbackSpeakerId = orderedSpeakers.first?.id ?? "speaker_0"
    }

    func sanitizedId(for rawId: String) -> String? {
        rawToSanitized[rawId]
    }
}
#endif
