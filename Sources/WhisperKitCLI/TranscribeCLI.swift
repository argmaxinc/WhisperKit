//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser
import CoreML
import Foundation
import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
struct TranscribeCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe audio to text using WhisperKit"
    )

    @OptionGroup
    var cliArguments: CLIArguments

    mutating func validate() throws {
        if let language = cliArguments.language {
            if !Constants.languages.values.contains(language) {
                throw ValidationError("Invalid language code \"\(language)\". Supported languages: \(Constants.languages.values)")
            }
        }

        if cliArguments.audioPath.isEmpty && !cliArguments.stream {
            guard let audioFolder = cliArguments.audioFolder else {
                throw ValidationError("Either audioPath or audioFolder must be provided.")
            }
            let fileManager = FileManager.default
            let audioExtensions = ["mp3", "wav", "m4a", "flac", "aiff", "aac"]
            let audioFiles = try fileManager.contentsOfDirectory(atPath: audioFolder)
                .filter { fileName in
                    let fileExtension = fileName.lowercased().components(separatedBy: ".").last
                    return audioExtensions.contains(fileExtension ?? "")
                }

            cliArguments.audioPath = audioFiles.map { audioFolder + "/" + $0 }
        }

        if ChunkingStrategy(rawValue: cliArguments.chunkingStrategy) == nil {
            throw ValidationError("Wrong chunking strategy \"\(cliArguments.chunkingStrategy)\", valid strategies: \(ChunkingStrategy.allCases.map { $0.rawValue })")
        }
    }

    mutating func run() async throws {
        if cliArguments.stream {
            try await transcribeStream()
        } else if cliArguments.streamSimulated {
            try await transcribeStreamSimulated()
        } else {
            try await transcribe()
        }
    }

    private func transcribe() async throws {
        let resolvedAudioPaths = cliArguments.audioPath.map { resolveAbsolutePath($0) }
        for resolvedAudioPath in resolvedAudioPaths {
            guard FileManager.default.fileExists(atPath: resolvedAudioPath) else {
                throw CocoaError.error(.fileNoSuchFile)
            }
        }

        let task: DecodingTask
        if cliArguments.task.lowercased() == "translate" {
            task = .translate
        } else {
            task = .transcribe
        }

        if cliArguments.verbose {
            print("Task: \(task.description.capitalized) audio at \(cliArguments.audioPath)")
            print("Initializing models...")
        }

        let whisperKit = try await setupWhisperKit()

        if cliArguments.verbose {
            print("Models initialized")
        }

        var options = decodingOptions(task: task)
        if let promptText = cliArguments.prompt, promptText.count > 0, let tokenizer = whisperKit.tokenizer {
            options.promptTokens = tokenizer.encode(text: " " + promptText.trimmingCharacters(in: .whitespaces)).filter { $0 < tokenizer.specialTokens.specialTokenBegin }
            options.usePrefillPrompt = true
        }

        if let prefixText = cliArguments.prefix, prefixText.count > 0, let tokenizer = whisperKit.tokenizer {
            options.prefixTokens = tokenizer.encode(text: " " + prefixText.trimmingCharacters(in: .whitespaces)).filter { $0 < tokenizer.specialTokens.specialTokenBegin }
            options.usePrefillPrompt = true
        }

        let transcribeResult: [Result<[TranscriptionResult], Swift.Error>] = await whisperKit.transcribeWithResults(
            audioPaths: resolvedAudioPaths,
            decodeOptions: options
        )

        // TODO: we need to track the results between audio files, and shouldnt merge them
        // ONLY merge results for different chunks of the same audio file or array
        let allSuccessfulResults = transcribeResult.compactMap { try? $0.get() }.flatMap { $0 }

        // Log timings for full transcription run
        if cliArguments.verbose {
            let mergedResult = mergeTranscriptionResults(allSuccessfulResults)
            mergedResult.logTimings()
        }

        for (audioPath, result) in zip(resolvedAudioPaths, transcribeResult) {
            do {
                let partialResult = try result.get()
                let mergedPartialResult = mergeTranscriptionResults(partialResult)
                processTranscriptionResult(audioPath: audioPath, transcribeResult: mergedPartialResult)
            } catch {
                print("Error when transcribing \(audioPath): \(error)")
            }
        }
    }

    private func transcribeStream() async throws {
        if cliArguments.verbose {
            print("Task: stream transcription, using microphone audio")
            print("Initializing models...")
        }

        let whisperKit = try await setupWhisperKit()
        guard let tokenizer = whisperKit.tokenizer else {
            throw WhisperError.tokenizerUnavailable()
        }

        if cliArguments.verbose {
            print("Models initialized")
        }

        let decodingOptions = decodingOptions(task: .transcribe)

        let audioStreamTranscriber = AudioStreamTranscriber(
            audioEncoder: whisperKit.audioEncoder,
            featureExtractor: whisperKit.featureExtractor,
            segmentSeeker: whisperKit.segmentSeeker,
            textDecoder: whisperKit.textDecoder,
            tokenizer: tokenizer,
            audioProcessor: whisperKit.audioProcessor,
            decodingOptions: decodingOptions
        ) { oldState, newState in
            guard oldState.currentText != newState.currentText ||
                oldState.unconfirmedSegments != newState.unconfirmedSegments ||
                oldState.confirmedSegments != newState.confirmedSegments
            else {
                return
            }
            // TODO: Print only net new text without any repeats
            print("---")
            for segment in newState.confirmedSegments {
                print("Confirmed segment: \(segment.text)")
            }
            for segment in newState.unconfirmedSegments {
                print("Unconfirmed segment: \(segment.text)")
            }
            print("Current text: \(newState.currentText)")
        }
        print("Transcribing audio stream, press Ctrl+C to stop.")
        try await audioStreamTranscriber.startStreamTranscription()
    }

    private func transcribeStreamSimulated() async throws {
        guard let audioPath = cliArguments.audioPath.first else {
            throw CocoaError.error(.fileNoSuchFile)
        }
        let resolvedAudioPath = resolveAbsolutePath(audioPath)
        guard FileManager.default.fileExists(atPath: resolvedAudioPath) else {
            throw CocoaError.error(.fileNoSuchFile)
        }

        if cliArguments.verbose {
            print("Task: simulated stream transcription, using audio file at \(audioPath)")
            print("Initializing models...")
        }

        let whisperKit = try await setupWhisperKit()

        if cliArguments.verbose {
            print("Models initialized")
        }

        let audioBuffer = try AudioProcessor.loadAudio(fromPath: resolvedAudioPath)
        let audioArray = AudioProcessor.convertBufferToArray(buffer: audioBuffer)

        var results: [TranscriptionResult?] = []
        var prevResult: TranscriptionResult?
        var lastAgreedSeconds: Float = 0.0
        let agreementCountNeeded = 2
        var hypothesisWords: [WordTiming] = []
        var prevWords: [WordTiming] = []
        var lastAgreedWords: [WordTiming] = []
        var confirmedWords: [WordTiming] = []

        let options = decodingOptions(task: .transcribe)

        for seekSample in stride(from: 16000, to: audioArray.count, by: 16000) {
            let endSample = min(seekSample + 16000, audioArray.count)
            if cliArguments.verbose {
                print("[whisperkit-cli] \(lastAgreedSeconds)-\(Double(endSample) / 16000.0) seconds")
            }

            let simulatedStreamingAudio = Array(audioArray[..<endSample])
            var streamOptions = options
            streamOptions.clipTimestamps = [lastAgreedSeconds]
            let lastAgreedTokens = lastAgreedWords.flatMap { $0.tokens }
            streamOptions.prefixTokens = lastAgreedTokens
            do {
                let result: TranscriptionResult? = try await whisperKit.transcribe(audioArray: simulatedStreamingAudio, decodeOptions: streamOptions).first
                var skipAppend = false
                if let result = result, let _ = result.segments.first?.words {
                    hypothesisWords = result.allWords.filter { $0.start >= lastAgreedSeconds }

                    if let prevResult = prevResult {
                        prevWords = prevResult.allWords.filter { $0.start >= lastAgreedSeconds }
                        let commonPrefix = findLongestCommonPrefix(prevWords, hypothesisWords)
                        if cliArguments.verbose {
                            print("[whisperkit-cli] Prev \"\((prevWords.map { $0.word }).joined())\"")
                            print("[whisperkit-cli] Next \"\((hypothesisWords.map { $0.word }).joined())\"")
                            print("[whisperkit-cli] Found common prefix \"\((commonPrefix.map { $0.word }).joined())\"")
                        }

                        if commonPrefix.count >= agreementCountNeeded {
                            lastAgreedWords = commonPrefix.suffix(agreementCountNeeded)
                            lastAgreedSeconds = lastAgreedWords.first!.start
                            if cliArguments.verbose {
                                print("[whisperkit-cli] Found new last agreed word \(lastAgreedWords.first!.word) at \(lastAgreedSeconds) seconds")
                            }

                            confirmedWords.append(contentsOf: commonPrefix.prefix(commonPrefix.count - agreementCountNeeded))
                            let currentWords = confirmedWords.map { $0.word }.joined()
                            if cliArguments.verbose {
                                print("[whisperkit-cli] Current: \(lastAgreedSeconds) -> \(Double(endSample) / 16000.0) \(currentWords)")
                            }
                        } else {
                            if cliArguments.verbose {
                                print("[whisperkit-cli] Using same last agreed time \(lastAgreedSeconds)")
                            }
                            skipAppend = true
                        }
                    }
                    prevResult = result
                } else {
                    if cliArguments.verbose {
                        print("[whisperkit-cli] No word timings found, this may be due to alignment weights missing from the model being used")
                    }
                }
                if !skipAppend {
                    results.append(result)
                }
            } catch {
                if cliArguments.verbose {
                    print("Error: \(error.localizedDescription)")
                }
            }
        }

        let final = lastAgreedWords + findLongestDifferentSuffix(prevWords, hypothesisWords)
        confirmedWords.append(contentsOf: final)

        let mergedResult = mergeTranscriptionResults(results, confirmedWords: confirmedWords)

        processTranscriptionResult(audioPath: audioPath, transcribeResult: mergedResult)
    }

    private func setupWhisperKit() async throws -> WhisperKit {
        var audioEncoderComputeUnits = cliArguments.audioEncoderComputeUnits.asMLComputeUnits
        let textDecoderComputeUnits = cliArguments.textDecoderComputeUnits.asMLComputeUnits

        // Use gpu for audio encoder on macOS below 14
        if audioEncoderComputeUnits == .cpuAndNeuralEngine {
            if #unavailable(macOS 14.0) {
                audioEncoderComputeUnits = .cpuAndGPU
            }
        }

        let computeOptions = ModelComputeOptions(
            audioEncoderCompute: audioEncoderComputeUnits,
            textDecoderCompute: textDecoderComputeUnits
        )

        let downloadTokenizerFolder: URL? =
            if let filePath = cliArguments.downloadTokenizerPath {
                URL(filePath: filePath)
            } else {
                nil
            }

        let downloadModelFolder: URL? =
            if let filePath = cliArguments.downloadModelPath {
                URL(filePath: filePath)
            } else {
                nil
            }

        let modelName: String? =
            if let modelVariant = cliArguments.model {
                cliArguments.modelPrefix + "*" + modelVariant
            } else {
                nil
            }

        let config = WhisperKitConfig(model: modelName,
                                      downloadBase: downloadModelFolder,
                                      modelFolder: cliArguments.modelPath,
                                      tokenizerFolder: downloadTokenizerFolder,
                                      computeOptions: computeOptions,
                                      verbose: cliArguments.verbose,
                                      logLevel: .debug,
                                      prewarm: false,
                                      load: true,
                                      useBackgroundDownloadSession: false)
        return try await WhisperKit(config)
    }

    private func decodingOptions(task: DecodingTask) -> DecodingOptions {
        return DecodingOptions(
            verbose: cliArguments.verbose,
            task: task,
            language: cliArguments.language,
            temperature: cliArguments.temperature,
            temperatureIncrementOnFallback: cliArguments.temperatureIncrementOnFallback,
            temperatureFallbackCount: cliArguments.temperatureFallbackCount,
            topK: cliArguments.bestOf,
            usePrefillPrompt: cliArguments.usePrefillPrompt || cliArguments.language != nil,
            usePrefillCache: cliArguments.usePrefillCache,
            skipSpecialTokens: cliArguments.skipSpecialTokens,
            withoutTimestamps: cliArguments.withoutTimestamps,
            wordTimestamps: cliArguments.wordTimestamps || cliArguments.streamSimulated,
            clipTimestamps: cliArguments.clipTimestamps,
            supressTokens: cliArguments.supressTokens,
            compressionRatioThreshold: cliArguments.compressionRatioThreshold ?? 2.4,
            logProbThreshold: cliArguments.logprobThreshold ?? -1.0,
            firstTokenLogProbThreshold: cliArguments.firstTokenLogProbThreshold,
            noSpeechThreshold: cliArguments.noSpeechThreshold ?? 0.6,
            concurrentWorkerCount: cliArguments.concurrentWorkerCount,
            chunkingStrategy: ChunkingStrategy(rawValue: cliArguments.chunkingStrategy)
        )
    }

    private func processTranscriptionResult(
        audioPath: String,
        transcribeResult: TranscriptionResult?
    ) {
        let audioFile = URL(fileURLWithPath: audioPath).lastPathComponent
        let audioFileName = audioFile.components(separatedBy: ".").first!
        let transcription = transcribeResult?.text ?? "Transcription failed"

        if cliArguments.report, let result = transcribeResult {
            // Write SRT (SubRip Subtitle Format) for the transcription
            let srtReportWriter = WriteSRT(outputDir: cliArguments.reportPath)
            let savedSrtReport = srtReportWriter.write(result: result, to: audioFileName)
            if cliArguments.verbose {
                switch savedSrtReport {
                    case let .success(reportPath):
                        print("\n\nSaved SRT Report: \n\n\(reportPath)\n")
                    case let .failure(error):
                        print("\n\nCouldn't save report: \(error)\n")
                }
            }

            // Write JSON for all metadata
            let jsonReportWriter = WriteJSON(outputDir: cliArguments.reportPath)
            let savedJsonReport = jsonReportWriter.write(result: result, to: audioFileName)
            if cliArguments.verbose {
                switch savedJsonReport {
                    case let .success(reportPath):
                        print("\n\nSaved JSON Report: \n\n\(reportPath)\n")
                    case let .failure(error):
                        print("\n\nCouldn't save report: \(error)\n")
                }
            }
        }

        if cliArguments.verbose {
            print("\n\nTranscription of \(audioFile): \n\n\(transcription)\n")
        } else {
            print(transcription)
        }
    }
}
