//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser
import CoreML
import Foundation
import WhisperKit

@main
struct WhisperKitCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "WhisperKit Transcribe CLI",
        discussion: "Swift native speech recognition with Whisper for Apple Silicon"
    )

    @OptionGroup 
    var cliArguments: CLIArguments

    mutating func run() async throws {
        if cliArguments.stream {
            try await transcribeStream(modelPath: cliArguments.modelPath)
        } else {
            let audioURL = URL(fileURLWithPath: cliArguments.audioPath)
            if cliArguments.verbose {
                print("Transcribing audio at \(audioURL)")
            }
            try await transcribe(audioPath: cliArguments.audioPath, modelPath: cliArguments.modelPath)
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
            audioEncoderCompute: cliArguments.audioEncoderComputeUnits.asMLComputeUnits,
            textDecoderCompute: cliArguments.textDecoderComputeUnits.asMLComputeUnits
        )

        print("Initializing models...")
        let whisperKit = try await WhisperKit(
            modelFolder: modelPath,
            computeOptions: computeOptions,
            verbose: cliArguments.verbose,
            logLevel: .debug
        )
        print("Models initialized")

        let options = DecodingOptions(
            verbose: cliArguments.verbose,
            task: .transcribe,
            language: cliArguments.language,
            temperature: cliArguments.temperature,
            temperatureIncrementOnFallback: cliArguments.temperatureIncrementOnFallback,
            temperatureFallbackCount: cliArguments.temperatureFallbackCount,
            topK: cliArguments.bestOf,
            usePrefillPrompt: cliArguments.usePrefillPrompt,
            usePrefillCache: cliArguments.usePrefillCache,
            skipSpecialTokens: cliArguments.skipSpecialTokens,
            withoutTimestamps: cliArguments.withoutTimestamps,
            wordTimestamps: cliArguments.wordTimestamps,
            supressTokens: cliArguments.supressTokens,
            compressionRatioThreshold: cliArguments.compressionRatioThreshold,
            logProbThreshold: cliArguments.logprobThreshold,
            noSpeechThreshold: cliArguments.noSpeechThreshold
        )

        let transcribeResult = try await whisperKit.transcribe(
            audioPath: resolvedAudioPath, 
            decodeOptions: options
        )

        let transcription = transcribeResult?.text ?? "Transcription failed"

        if cliArguments.report, let result = transcribeResult {
            let audioFileName = URL(fileURLWithPath: audioPath).lastPathComponent.components(separatedBy: ".").first!

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

    private func transcribeStream(modelPath: String) async throws {
        let computeOptions = ModelComputeOptions(
            audioEncoderCompute: cliArguments.audioEncoderComputeUnits.asMLComputeUnits,
            textDecoderCompute: cliArguments.textDecoderComputeUnits.asMLComputeUnits
        )

        print("Initializing models...")
        let whisperKit = try await WhisperKit(
            modelFolder: modelPath,
            computeOptions: computeOptions,
            verbose: cliArguments.verbose,
            logLevel: .debug
        )
        print("Models initialized")

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
