//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

actor TranscriptionWorker {
    private let dependencies: TranscribeTaskDependencies

    init(dependencies: TranscribeTaskDependencies) {
        self.dependencies = dependencies
    }

    func transcribe(
        index: Int,
        audioArray: [Float],
        decodeOptions: DecodingOptions?,
        callback: TranscriptionCallback?,
        segmentCallback: SegmentDiscoveryCallback?,
        progress: Progress
    ) async -> (index: Int, result: Result<[TranscriptionResult], Swift.Error>) {
        defer {
            progress.totalUnitCount = max(1, progress.totalUnitCount)
            progress.completedUnitCount = progress.totalUnitCount
        }

        do {
            let transcribeResult = try await transcribe(
                audioArray: audioArray,
                decodeOptions: decodeOptions,
                callback: callback,
                segmentCallback: segmentCallback,
                progress: progress
            )
            return (index: index, result: .success(transcribeResult))
        } catch {
            return (index: index, result: .failure(error))
        }
    }

    private func transcribe(
        audioArray: [Float],
        decodeOptions: DecodingOptions?,
        callback: TranscriptionCallback?,
        segmentCallback: SegmentDiscoveryCallback?,
        progress: Progress
    ) async throws -> [TranscriptionResult] {
        let windowSamples = dependencies.featureExtractor.windowSamples ?? Constants.defaultWindowSamples
        let isChunkable = audioArray.count > windowSamples

        switch (isChunkable, decodeOptions?.chunkingStrategy) {
            case (true, .vad):
                let vad = dependencies.voiceActivityDetector ?? EnergyVAD()
                let chunker = VADAudioChunker(vad: vad)
                let audioChunks: [AudioChunk] = try await chunker.chunkAll(
                    audioArray: audioArray,
                    maxChunkLength: windowSamples,
                    decodeOptions: decodeOptions
                )

                progress.totalUnitCount = max(progress.totalUnitCount, Int64(audioChunks.count))

                var chunkedOptions = decodeOptions
                chunkedOptions?.clipTimestamps = []

                var chunkedResults = [Result<[TranscriptionResult], Swift.Error>]()
                chunkedResults.reserveCapacity(audioChunks.count)

                for audioChunk in audioChunks {
                    let chunkProgress = Progress(totalUnitCount: 1)
                    progress.addChild(chunkProgress, withPendingUnitCount: 1)

                    let chunkSegmentCallback: SegmentDiscoveryCallback? = if let segmentCallback {
                        { segments in
                            var adjustedSegments = segments
                            for i in 0..<adjustedSegments.count {
                                adjustedSegments[i].seek += audioChunk.seekOffsetIndex
                            }
                            segmentCallback(adjustedSegments)
                        }
                    } else {
                        nil
                    }

                    do {
                        let chunkResult = try await runTranscribeTask(
                            audioArray: audioChunk.audioSamples,
                            decodeOptions: chunkedOptions,
                            callback: callback,
                            segmentCallback: chunkSegmentCallback,
                            progress: chunkProgress
                        )
                        chunkedResults.append(.success(chunkResult))
                    } catch {
                        chunkProgress.completedUnitCount = chunkProgress.totalUnitCount
                        chunkedResults.append(.failure(error))
                    }
                }

                return chunker.updateSeekOffsetsForResults(
                    chunkedResults: chunkedResults,
                    audioChunks: audioChunks
                )
            default:
                return try await runTranscribeTask(
                    audioArray: audioArray,
                    decodeOptions: decodeOptions,
                    callback: callback,
                    segmentCallback: segmentCallback,
                    progress: progress
                )
        }
    }

    private func runTranscribeTask(
        audioArray: [Float],
        decodeOptions: DecodingOptions?,
        callback: TranscriptionCallback?,
        segmentCallback: SegmentDiscoveryCallback?,
        progress: Progress
    ) async throws -> [TranscriptionResult] {
        let workerTask = dependencies.makeTranscribeTask(progress: progress)
        workerTask.segmentDiscoveryCallback = segmentCallback

        let transcribeTaskResult = try await workerTask.run(
            audioArray: audioArray,
            decodeOptions: decodeOptions,
            callback: callback
        )

        if let decodeOptions, decodeOptions.verbose {
            transcribeTaskResult.logTimings()
        }

        return [transcribeTaskResult]
    }
}
