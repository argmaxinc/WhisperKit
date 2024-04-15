//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser
import CoreML
import Foundation
import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
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
        let resolvedAudioPath = resolveAbsolutePath(cliArguments.audioPath)
        guard FileManager.default.fileExists(atPath: resolvedAudioPath) else {
            throw CocoaError.error(.fileNoSuchFile)
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
        if let promptText = cliArguments.prompt, let tokenizer = whisperKit.tokenizer  {
            options.promptTokens = tokenizer.encode(text: " " + promptText.trimmingCharacters(in: .whitespaces)).filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        }

        if let prefixText = cliArguments.prefix, let tokenizer = whisperKit.tokenizer {
            options.prefixTokens = tokenizer.encode(text: " " + prefixText.trimmingCharacters(in: .whitespaces)).filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        }

        let transcribeResult = try await whisperKit.transcribe(
            audioPath: resolvedAudioPath,
            decodeOptions: options
        )

        processTranscriptionResult(transcribeResult)
    }

    private func transcribeStream() async throws {
        if cliArguments.verbose {
            print("Task: stream transcription, using microphone audio")
            print("Initializing models...")
        }

        let whisperKit = try await setupWhisperKit()

        if cliArguments.verbose {
            print("Models initialized")
        }

        let decodingOptions = decodingOptions(task: .transcribe)

        let audioStreamTranscriber = AudioStreamTranscriber(
            audioProcessor: whisperKit.audioProcessor,
            transcriber: whisperKit,
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
        let resolvedAudioPath = resolveAbsolutePath(cliArguments.audioPath)
        guard FileManager.default.fileExists(atPath: resolvedAudioPath) else {
            throw CocoaError.error(.fileNoSuchFile)
        }

        if cliArguments.verbose {
            print("Task: simulated stream transcription, using audio file at \(cliArguments.audioPath)")
            print("Initializing models...")
        }

        let whisperKit = try await setupWhisperKit()

        if cliArguments.verbose {
            print("Models initialized")
        }

        guard let audioBuffer = AudioProcessor.loadAudio(fromPath: resolvedAudioPath) else {
            print("Failed to load audio buffer")
            return
        }

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
                let result: TranscriptionResult? = try await whisperKit.transcribe(audioArray: simulatedStreamingAudio, decodeOptions: streamOptions)
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

        processTranscriptionResult(mergedResult)
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

        return try await WhisperKit(
            model: modelName,
            downloadBase: downloadModelFolder,
            modelFolder: cliArguments.modelPath,
            tokenizerFolder: downloadTokenizerFolder,
            computeOptions: computeOptions,
            verbose: cliArguments.verbose,
            logLevel: .debug,
            load: true,
            useBackgroundDownloadSession: false
        )
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
            supressTokens: cliArguments.supressTokens,
            compressionRatioThreshold: cliArguments.compressionRatioThreshold ?? 2.4,
            logProbThreshold: cliArguments.logprobThreshold ?? -1.0,
            firstTokenLogProbThreshold: cliArguments.firstTokenLogProbThreshold,
            noSpeechThreshold: cliArguments.noSpeechThreshold ?? 0.6
        )
    }

    private func processTranscriptionResult(_ transcribeResult: TranscriptionResult?) {
        let transcription = transcribeResult?.text ?? "Transcription failed"

        if cliArguments.report, let result = transcribeResult {
            let audioFileName = URL(fileURLWithPath: cliArguments.audioPath).lastPathComponent.components(separatedBy: ".").first!

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
            print("\n\nTranscription: \n\n\(transcription)\n")
        } else {
            print(transcription)
        }
    }
}
