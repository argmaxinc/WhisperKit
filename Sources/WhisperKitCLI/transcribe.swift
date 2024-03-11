//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser
import CoreML
import Foundation
import WhisperKit

struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Transcribe audio to text using WhisperKit"
    )

    @OptionGroup 
    var commandLineOptions: WhisperKitArguments

    mutating func run() async throws {
        if commandLineOptions.stream {
            try await transcribeStream(modelPath: commandLineOptions.modelPath)
        } else {
            let audioURL = URL(fileURLWithPath: commandLineOptions.audioPath)
            if commandLineOptions.verbose {
                print("Transcribing audio at \(audioURL)")
            }
            try await transcribe(audioPath: commandLineOptions.audioPath, modelPath: commandLineOptions.modelPath)
        }
    }

    private func transcribe(audioPath: String, modelPath: String) async throws {
        let resolvedModelPath = resolveAbsolutePath(modelPath)
        guard FileManager.default.fileExists(atPath: resolvedModelPath) else {
            fatalError("Model path does not exist \(resolvedModelPath)")
        }

        let resolvedAudioPath = resolveAbsolutePath(audioPath)
        guard FileManager.default.fileExists(atPath: resolvedAudioPath) else {
            fatalError("Resource path does not exist \(resolvedAudioPath)")
        }

        let computeOptions = ModelComputeOptions(
            audioEncoderCompute: commandLineOptions.audioEncoderComputeUnits.asMLComputeUnits,
            textDecoderCompute: commandLineOptions.textDecoderComputeUnits.asMLComputeUnits
        )

        print("Initializing models...")
        let whisperKit = try await WhisperKit(
            modelFolder: modelPath,
            computeOptions: computeOptions,
            verbose: commandLineOptions.verbose,
            logLevel: .debug
        )
        print("Models initialized")

        let options = DecodingOptions(
            verbose: commandLineOptions.verbose,
            task: .transcribe,
            language: commandLineOptions.language,
            temperature: commandLineOptions.temperature,
            temperatureIncrementOnFallback: commandLineOptions.temperatureIncrementOnFallback,
            temperatureFallbackCount: commandLineOptions.temperatureFallbackCount,
            topK: commandLineOptions.bestOf,
            usePrefillPrompt: commandLineOptions.usePrefillPrompt,
            usePrefillCache: commandLineOptions.usePrefillCache,
            skipSpecialTokens: commandLineOptions.skipSpecialTokens,
            withoutTimestamps: commandLineOptions.withoutTimestamps,
            wordTimestamps: commandLineOptions.wordTimestamps,
            supressTokens: commandLineOptions.supressTokens,
            compressionRatioThreshold: commandLineOptions.compressionRatioThreshold,
            logProbThreshold: commandLineOptions.logprobThreshold,
            noSpeechThreshold: commandLineOptions.noSpeechThreshold
        )

        let transcribeResult = try await whisperKit.transcribe(
            audioPath: resolvedAudioPath, 
            decodeOptions: options
        )

        let transcription = transcribeResult?.text ?? "Transcription failed"

        if commandLineOptions.report, let result = transcribeResult {
            let audioFileName = URL(fileURLWithPath: audioPath).lastPathComponent.components(separatedBy: ".").first!

            // Write SRT (SubRip Subtitle Format) for the transcription
            let srtReportWriter = WriteSRT(outputDir: commandLineOptions.reportPath)
            let savedSrtReport = srtReportWriter.write(result: result, to: audioFileName)
            if commandLineOptions.verbose {
                switch savedSrtReport {
                    case let .success(reportPath):
                        print("\n\nSaved SRT Report: \n\n\(reportPath)\n")
                    case let .failure(error):
                        print("\n\nCouldn't save report: \(error)\n")
                }
            }

            // Write JSON for all metadata
            let jsonReportWriter = WriteJSON(outputDir: commandLineOptions.reportPath)
            let savedJsonReport = jsonReportWriter.write(result: result, to: audioFileName)
            if commandLineOptions.verbose {
                switch savedJsonReport {
                    case let .success(reportPath):
                        print("\n\nSaved JSON Report: \n\n\(reportPath)\n")
                    case let .failure(error):
                        print("\n\nCouldn't save report: \(error)\n")
                }
            }
        }

        if commandLineOptions.verbose {
            print("\n\nTranscription: \n\n\(transcription)\n")
        } else {
            print(transcription)
        }
    }

    private func transcribeStream(modelPath: String) async throws {
        let computeOptions = ModelComputeOptions(
            audioEncoderCompute: commandLineOptions.audioEncoderComputeUnits.asMLComputeUnits,
            textDecoderCompute: commandLineOptions.textDecoderComputeUnits.asMLComputeUnits
        )

        print("Initializing models...")
        let whisperKit = try await WhisperKit(
            modelFolder: modelPath,
            computeOptions: computeOptions,
            verbose: commandLineOptions.verbose,
            logLevel: .debug
        )
        print("Models initialized")

        let decodingOptions = DecodingOptions(
            verbose: commandLineOptions.verbose,
            task: .transcribe,
            language: commandLineOptions.language,
            temperature: commandLineOptions.temperature,
            temperatureIncrementOnFallback: commandLineOptions.temperatureIncrementOnFallback,
            temperatureFallbackCount: 3, // limit fallbacks for realtime
            sampleLength: 224, // reduced sample length for realtime
            topK: commandLineOptions.bestOf,
            usePrefillPrompt: commandLineOptions.usePrefillPrompt,
            usePrefillCache: commandLineOptions.usePrefillCache,
            skipSpecialTokens: commandLineOptions.skipSpecialTokens,
            withoutTimestamps: commandLineOptions.withoutTimestamps,
            clipTimestamps: [],
            suppressBlank: false,
            supressTokens: commandLineOptions.supressTokens,
            compressionRatioThreshold: commandLineOptions.compressionRatioThreshold ?? 2.4,
            logProbThreshold: commandLineOptions.logprobThreshold ?? -1.0,
            noSpeechThreshold: commandLineOptions.noSpeechThreshold ?? 0.6
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
