//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

/// Responsible for transcribing audio chunk to text using the provided models and configurations.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
final class TranscribeTask {
    private var timings: TranscriptionTimings
    private let progress: Progress
    private let audioEncoder: any AudioEncoding
    private let featureExtractor: any FeatureExtracting
    private let segmentSeeker: any SegmentSeeking
    private let textDecoder: any TextDecoding
    private let tokenizer: any WhisperTokenizer

    init(
        currentTimings: TranscriptionTimings,
        progress: Progress?,
        audioEncoder: any AudioEncoding,
        featureExtractor: any FeatureExtracting,
        segmentSeeker: any SegmentSeeking,
        textDecoder: any TextDecoding,
        tokenizer: any WhisperTokenizer
    ) {
        self.timings = currentTimings
        self.progress = progress ?? Progress()
        self.audioEncoder = audioEncoder
        self.featureExtractor = featureExtractor
        self.segmentSeeker = segmentSeeker
        self.textDecoder = textDecoder
        self.tokenizer = tokenizer
    }

    func run(
        audioArray: [Float],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> TranscriptionResult {
        let interval = Logging.beginSignpost("TranscribeAudio", signposter: Logging.TranscribeTask.signposter)
        defer { Logging.endSignpost("TranscribeAudio", interval: interval, signposter: Logging.TranscribeTask.signposter) }

        timings.pipelineStart = CFAbsoluteTimeGetCurrent()

        var options = decodeOptions ?? DecodingOptions()
        options.verbose = Logging.shared.logLevel != .none

        var detectedLanguage: String?

        let contentFrames = audioArray.count
        timings.inputAudioSeconds = Double(Int(contentFrames) / WhisperKit.sampleRate) - Double(decodeOptions?.clipTimestamps.first ?? 0)

        // MARK: Init decoder inputs

        // These accumulate across windows
        var allSegments: [TranscriptionSegment] = []
        var allTokens: [Int] = []
        var transcription = ""

        let startDecoderInit = CFAbsoluteTimeGetCurrent()
        var decoderInputs = try textDecoder.prepareDecoderInputs(withPrompt: [tokenizer.specialTokens.startOfTranscriptToken])
        let decoderInitTime = CFAbsoluteTimeGetCurrent() - startDecoderInit
        timings.decodingInit = decoderInitTime
        Logging.debug("Decoder init time: \(decoderInitTime)")

        // MARK: - Prefill KV Cache

        let prefillStartTime = CFAbsoluteTimeGetCurrent()
        var prefilledCacheSize = 0
        if options.usePrefillPrompt {
            let prefilledInputs = try await textDecoder.prefillDecoderInputs(decoderInputs, withOptions: options)
            decoderInputs = prefilledInputs
            prefilledCacheSize = decoderInputs.cacheLength[0].intValue
        }
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStartTime
        timings.prefill = prefillTime

        // Setup masks based on prefill values
        prefilledCacheSize += 1 // Add 1 for initial masked cache update
        decoderInputs.kvCacheUpdateMask[prefilledCacheSize - 1] = 1.0
        for i in 0..<prefilledCacheSize {
            decoderInputs.decoderKeyPaddingMask[i] = 0.0
        }

        Logging.debug("Prefill time: \(prefillTime)")
        Logging.debug("Prefill prompt: \(decoderInputs.initialPrompt.map { tokenizer.convertIdToToken($0) ?? "" })")

        // MARK: - Main decoder loop

        var fallbackCount: Double = 0

        // Process seek clips
        var seekPoints: [Int] = options.clipTimestamps.map { Int(round($0 * Float(WhisperKit.sampleRate))) }
        if seekPoints.count == 0 {
            seekPoints.append(0)
        }
        if seekPoints.count % 2 == 1 {
            seekPoints.append(contentFrames)
        }

        var seekClips: [(start: Int, end: Int)] = []
        for i in stride(from: 0, to: seekPoints.count, by: 2) {
            let start = seekPoints[i]
            let end = i + 1 < seekPoints.count ? seekPoints[i + 1] : contentFrames
            seekClips.append((start, end))
        }

        let startDecodeLoopTime = CFAbsoluteTimeGetCurrent()

        let totalSeekDuration = seekClips.reduce(0) { $0 + ($1.end - $1.start) }
        progress.totalUnitCount = Int64(totalSeekDuration)
        defer { progress.completedUnitCount = progress.totalUnitCount }
        for (seekClipStart, seekClipEnd) in seekClips {
            // Loop through the current clip until we reach the end
            // Typically this will be the full audio file, unless seek points are explicitly provided
            var seek: Int = seekClipStart

            let previousSeekProgress = progress.completedUnitCount

            let windowPadding = 16000 // prevent hallucinations at the end of the clip by stopping up to 1.0s early
            while seek < seekClipEnd - windowPadding {
                // calculate new encoder segment features
                Logging.debug("Decoding Seek: \(seek)")
                let timeOffset = Float(seek) / Float(WhisperKit.sampleRate)
                let segmentSize = min(WhisperKit.windowSamples, contentFrames - seek, seekClipEnd - seek)
                let timeOffsetEnd = Float(seek + segmentSize) / Float(WhisperKit.sampleRate)

                let audioProcessingStart = Date()
                guard let audioSamples = AudioProcessor.padOrTrimAudio(fromArray: audioArray, startAt: seek, toLength: WhisperKit.windowSamples) else {
                    throw WhisperError.transcriptionFailed("Audio samples are nil")
                }
                let processTime = Date().timeIntervalSince(audioProcessingStart)
                timings.audioProcessing += processTime
                timings.totalAudioProcessingRuns += 1

                try Task.checkCancellation()
                let melStart = Date()
                guard let melOutput = try await featureExtractor.logMelSpectrogram(fromAudio: audioSamples) else {
                    throw WhisperError.transcriptionFailed("Mel output is nil")
                }
                let melTime = Date().timeIntervalSince(melStart)
                timings.logmels += melTime
                timings.totalLogmelRuns += 1

                try Task.checkCancellation()
                let encoderStart = Date()
                guard let encoderOutput = try await audioEncoder.encodeFeatures(melOutput) else {
                    throw WhisperError.transcriptionFailed("Encoder output is nil")
                }
                let encoderTime = Date().timeIntervalSince(encoderStart)
                timings.encoding += encoderTime
                timings.totalEncodingRuns += 1

                // All features are computed, now we can decode
                Logging.info("Decoding \(timeOffset)s - \(timeOffsetEnd)s")

                try Task.checkCancellation()
                // Send to decoder to predict text tokens with fallback
                let decodingResult = try await decodeWithFallback(encoderSegment: encoderOutput, decodingOptions: options, callback: callback)

                // MARK: Windowing

                // At this point we have a completed window aka segment
                let windowingStart = Date()

                let previousSeek = seek
                var (newSeek, currentSegments) = segmentSeeker.findSeekPointAndSegments(
                    decodingResult: decodingResult,
                    options: options,
                    allSegmentsCount: allSegments.count,
                    currentSeek: seek,
                    segmentSize: segmentSize,
                    sampleRate: WhisperKit.sampleRate,
                    timeToken: tokenizer.specialTokens.timeTokenBegin,
                    specialToken: tokenizer.specialTokens.specialTokenBegin,
                    tokenizer: tokenizer
                )

                // Update seek point without moving backward
                seek = max(seek, newSeek)

                // Optionally add word timestamps
                if options.wordTimestamps,
                   let alignmentWeights = decodingResult.cache?.alignmentWeights
                {
                    let wordTimestampsStart = Date()
                    currentSegments = try segmentSeeker.addWordTimestamps(
                        segments: currentSegments ?? [],
                        alignmentWeights: alignmentWeights,
                        tokenizer: tokenizer,
                        seek: previousSeek,
                        segmentSize: segmentSize,
                        prependPunctuations: "\"'“¿([{-",
                        appendPunctuations: "\"'.。,，!！?？:：”)]}、",
                        lastSpeechTimestamp: Float(Double(previousSeek) / Double(WhisperKit.sampleRate)),
                        options: options,
                        timings: timings
                    )

                    timings.decodingWordTimestamps += Date().timeIntervalSince(wordTimestampsStart)
                    timings.totalTimestampAlignmentRuns += 1

                    // Update seek point with new (more accurate) segments
                    if let lastSpeechTimestamp = currentSegments?.last?.end {
                        seek = max(seek, Int(lastSpeechTimestamp * Float(WhisperKit.sampleRate)))
                    }

                    if options.verbose {
                        Logging.debug("Word timestamps:")
                        for segment in currentSegments ?? [] {
                            for word in segment.words ?? [] {
                                Logging.debug("[\(word.start.formatted(.number.precision(.significantDigits(3)))) -> \(word.end.formatted(.number.precision(.significantDigits(3))))] prob: \(word.probability), word: \(word.word)")
                            }
                        }
                    }
                }

                guard let currentSegments = currentSegments else {
                    // No current segment found, skip to next window
                    continue
                }

                if options.verbose {
                    let lines = formatSegments(currentSegments)
                    for line in lines {
                        Logging.debug(line)
                    }
                }

                // add them to the `allSegments` list
                allSegments.append(contentsOf: currentSegments)
                let allCurrentTokens = currentSegments.flatMap { $0.tokens }
                allTokens.append(contentsOf: allCurrentTokens)

                timings.decodingWindowing += Date().timeIntervalSince(windowingStart)
                timings.totalDecodingWindows += 1

                // Reset cache and move on to the next window
                decoderInputs.reset(
                    prefilledCacheSize: prefilledCacheSize,
                    maxTokenContext: decodeOptions?.sampleLength ?? Constants.maxTokenContext
                )

                let clipProgress = min(seek, seekClipEnd) - seekClipStart
                progress.completedUnitCount = previousSeekProgress + Int64(clipProgress)
            }
        }

        func decodeWithFallback(
            encoderSegment encoderOutput: MLMultiArray,
            decodingOptions options: DecodingOptions,
            callback: TranscriptionCallback = nil
        ) async throws -> DecodingResult {
            let interval = Logging.beginSignpost("Decode", signposter: Logging.TranscribeTask.signposter)
            defer { Logging.endSignpost("Decode", interval: interval, signposter: Logging.TranscribeTask.signposter) }

            // Fallback `options.temperatureFallbackCount` times with increasing temperatures, starting at `options.temperature`
            let temperatures = (0...options.temperatureFallbackCount).map { FloatType(options.temperature) + FloatType($0) * FloatType(options.temperatureIncrementOnFallback) }

            Logging.debug("Decoding with temperatures \(temperatures)")

            var decodingResult: DecodingResult?

            for (i, temp) in temperatures.enumerated() {
                Logging.info("Decoding Temperature: \(temp)")
                let decodeWithFallbackStart = Date()

                let tokenSampler = GreedyTokenSampler(temperature: temp, eotToken: tokenizer.specialTokens.endToken, decodingOptions: options)

                var currentDecodingOptions = options
                // For a multilingual model, if language is not passed and detectLanguage is true, detect language and set in options
                if textDecoder.isModelMultilingual, options.language == nil, options.detectLanguage {
                    let languageDecodingResult: DecodingResult? = try? await textDecoder.detectLanguage(
                        from: encoderOutput,
                        using: decoderInputs,
                        sampler: tokenSampler,
                        options: options,
                        temperature: temp
                    )

                    // Update the language decoding options
                    currentDecodingOptions.language = languageDecodingResult?.language
                    detectedLanguage = languageDecodingResult?.language

                    // Update prompt and KV Cache if needed
                    if options.usePrefillPrompt {
                        decoderInputs = try await textDecoder.prefillDecoderInputs(decoderInputs, withOptions: currentDecodingOptions)
                    }
                    Logging.debug("Prefill prompt updated to: \(decoderInputs.initialPrompt.map { tokenizer.convertIdToToken($0) ?? "" })")

                    // Update timings from the language detection
                    if let languageDecodingTimings = languageDecodingResult?.timings {
                        timings.decodingPredictions += languageDecodingTimings.decodingPredictions
                        timings.decodingSampling += languageDecodingTimings.decodingSampling
                    }
                }

                decodingResult = try await textDecoder.decodeText(
                    from: encoderOutput,
                    using: decoderInputs,
                    sampler: tokenSampler,
                    options: currentDecodingOptions,
                    callback: callback
                )

                // Use the predicted language if it was not detected ahead of time
                if detectedLanguage == nil {
                    detectedLanguage = decodingResult?.language
                }

                // Update timings from the decoder main loop
                if let decodingTimings = decodingResult?.timings {
                    if timings.firstTokenTime == 0 {
                        timings.firstTokenTime = decodingTimings.firstTokenTime
                    }
                    timings.decodingPredictions += decodingTimings.decodingPredictions
                    timings.totalDecodingLoops += decodingTimings.totalDecodingLoops
                    timings.decodingNonPrediction += decodingTimings.decodingNonPrediction
                    timings.decodingFiltering += decodingTimings.decodingFiltering
                    timings.decodingSampling += decodingTimings.decodingSampling
                    timings.decodingKvCaching += decodingTimings.decodingKvCaching
                    timings.totalKVUpdateRuns += decodingTimings.totalKVUpdateRuns
                }

                // MARK: Fallback checks

                if let fallback = decodingResult?.fallback, fallback.needsFallback {
                    // Reset decoder inputs for fallback
                    fallbackCount = Double(i)
                    timings.decodingFallback += Date().timeIntervalSince(decodeWithFallbackStart)
                    timings.totalDecodingFallbacks = fallbackCount
                    decoderInputs.reset(
                        prefilledCacheSize: prefilledCacheSize,
                        maxTokenContext: decodeOptions?.sampleLength ?? Constants.maxTokenContext
                    )
                    Logging.info("Fallback #\(fallbackCount + 1) (\(fallback.fallbackReason))")
                } else {
                    break
                }
            }

            guard let decodingResult else {
                throw WhisperError.decodingFailed()
            }
            return decodingResult
        }

        // MARK: Result

        timings.decodingLoop = CFAbsoluteTimeGetCurrent() - startDecodeLoopTime
        timings.fullPipeline = CFAbsoluteTimeGetCurrent() - timings.pipelineStart

        let wordTokens = allTokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        transcription = tokenizer.decode(tokens: wordTokens).trimmingCharacters(in: .whitespaces)
        return TranscriptionResult(
            text: transcription,
            segments: allSegments,
            language: detectedLanguage ?? Constants.defaultLanguageCode,
            timings: timings
        )
    }
}
