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

    mutating func run() async throws {
        if cliArguments.stream {
            try await transcribeStream()
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
        }

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

        if cliArguments.verbose {
            print("Initializing models...")
        }

        let whisperKit = try await WhisperKit(
            model: cliArguments.model,
            downloadBase: downloadModelFolder,
            modelFolder: cliArguments.modelPath,
            tokenizerFolder: downloadTokenizerFolder,
            computeOptions: computeOptions,
            verbose: cliArguments.verbose,
            logLevel: .debug,
            useBackgroundDownloadSession: false
        )

        if cliArguments.verbose {
            print("Models initialized")
        }

        let options = DecodingOptions(
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
            wordTimestamps: cliArguments.wordTimestamps,
            supressTokens: cliArguments.supressTokens,
            compressionRatioThreshold: cliArguments.compressionRatioThreshold,
            logProbThreshold: cliArguments.logprobThreshold,
            firstTokenLogProbThreshold: cliArguments.firstTokenLogProbThreshold,
            noSpeechThreshold: cliArguments.noSpeechThreshold
        )

        let transcribeResult = try await whisperKit.transcribe(
            audioPath: resolvedAudioPath,
            decodeOptions: options
        )

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

    private func transcribeStream() async throws {
        let computeOptions = ModelComputeOptions(
            audioEncoderCompute: cliArguments.audioEncoderComputeUnits.asMLComputeUnits,
            textDecoderCompute: cliArguments.textDecoderComputeUnits.asMLComputeUnits
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

        if cliArguments.verbose {
            print("Initializing models...")
        }

        let whisperKit = try await WhisperKit(
            model: cliArguments.model,
            downloadBase: downloadModelFolder,
            modelFolder: cliArguments.modelPath,
            tokenizerFolder: downloadTokenizerFolder,
            computeOptions: computeOptions,
            verbose: cliArguments.verbose,
            logLevel: .debug,
            useBackgroundDownloadSession: false
        )

        if cliArguments.verbose {
            print("Models initialized")
        }
        let decodingOptions = DecodingOptions(
            verbose: cliArguments.verbose,
            task: .transcribe,
            language: cliArguments.language,
            temperature: cliArguments.temperature,
            temperatureIncrementOnFallback: cliArguments.temperatureIncrementOnFallback,
            temperatureFallbackCount: 3, // limit fallbacks for realtime
            sampleLength: 224, // reduced sample length for realtime
            topK: cliArguments.bestOf,
            usePrefillPrompt: cliArguments.usePrefillPrompt,
            usePrefillCache: cliArguments.usePrefillCache,
            skipSpecialTokens: cliArguments.skipSpecialTokens,
            withoutTimestamps: cliArguments.withoutTimestamps,
            clipTimestamps: [],
            suppressBlank: false,
            supressTokens: cliArguments.supressTokens,
            compressionRatioThreshold: cliArguments.compressionRatioThreshold ?? 2.4,
            logProbThreshold: cliArguments.logprobThreshold ?? -1.0,
            firstTokenLogProbThreshold: cliArguments.firstTokenLogProbThreshold ?? -0.7,
            noSpeechThreshold: cliArguments.noSpeechThreshold ?? 0.6
        )

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
}
