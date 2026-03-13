//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgumentParser
import Foundation
import SpeakerKit
import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
struct DiarizeCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "diarize",
        abstract: "Speaker diarization tool",
        discussion: "Process audio files for speaker diarization. Identifies and labels different speakers in the audio."
    )

    @Option(help: "Path to the audio file to process")
    var audioPath: String

    @Option(help: "Path to save the diarization output as RTTM")
    var rttmPath: String?

    @Option(help: "Path of local model files (skips download)")
    var modelPath: String?

    @Option(help: "HuggingFace model repository")
    var modelRepo: String?

    @Option(help: "HuggingFace API token")
    var modelToken: String?

    @Option(help: "Path to save downloaded models")
    var downloadModelPath: String?

    @Option(help: "Number of speakers to detect (default: automatic)")
    var numSpeakers: Int?

    @Option(help: "Cluster distance threshold for VBx clustering")
    var clusterDistanceThreshold: Float = 0.6

    @Flag(help: "Use exclusive reconciliation in post processing")
    var useExclusiveReconciliation: Bool = false

    @Flag(help: "Disable full redundancy in segmenter")
    var disableFullRedundancy: Bool = false

    @Flag(help: "Enable verbose output")
    var verbose: Bool = false

    mutating func validate() throws {
        guard FileManager.default.fileExists(atPath: audioPath) else {
            throw ValidationError("Audio file does not exist at path: \(audioPath)")
        }
    }

    mutating func run() async throws {
        if verbose {
            Logging.shared.logLevel = .debug
        }

        Logging.info("Loading audio from \(audioPath)...")
        let audioFrames = try AudioProcessor.loadAudioAsFloatArray(fromPath: audioPath)

        Logging.info("Initializing SpeakerKit...")
        let speakerKit = try await setupSpeakerKit()

        Logging.info("Starting diarization...")

        let options = PyannoteDiarizationOptions(
            numberOfSpeakers: numSpeakers,
            clusterDistanceThreshold: clusterDistanceThreshold,
            useExclusiveReconciliation: useExclusiveReconciliation
        )
        let diarizationResult = try await speakerKit.diarize(audioArray: audioFrames, options: options)

        Logging.info("Generating RTTM...")
        let audioURL = URL(filePath: audioPath)
        let fileName = audioURL.deletingPathExtension().lastPathComponent
        let rttmLines = SpeakerKit.generateRTTM(
            from: diarizationResult,
            fileName: fileName
        )

        let rttmContent = rttmLines.map(\.description).joined(separator: "\n")
        if let rttmPath = rttmPath {
            try rttmContent.write(to: URL(filePath: rttmPath), atomically: true, encoding: .utf8)
            Logging.info("RTTM file saved to \(rttmPath)")
        } else {
            print(rttmContent)
        }

        if verbose, let timingDescription = diarizationResult.timings?.debugDescription {
            print(timingDescription)
        }
    }

    private func setupSpeakerKit() async throws -> SpeakerKit {
        let modelFolder: URL? = modelPath.map { URL(filePath: $0) }

        let downloadFolder: URL? = downloadModelPath.map { URL(filePath: $0) }

        let config = PyannoteConfig(
            downloadBase: modelFolder == nil ? downloadFolder : nil,
            modelRepo: modelRepo,
            modelToken: modelToken,
            modelFolder: modelFolder,
            download: modelFolder == nil,
            verbose: verbose,
            fullRedundancy: !disableFullRedundancy
        )

        let manager = SpeakerKitModelManager(config: config)
        if modelFolder == nil {
            try await manager.downloadModels()
        }
        try await manager.loadModels()

        guard let models = manager.models as? PyannoteModels else {
            throw SpeakerKitError.modelUnavailable("Failed to load SpeakerKit models")
        }

        return try SpeakerKit(models: models)
    }
}
