//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser

struct CLIArguments: ParsableArguments {
    @Option(help: "Path to audio file")
    var audioPath: String = "Tests/WhisperKitTests/Resources/jfk.wav"

    @Option(help: "Path of model files")
    var modelPath: String?

    @Option(help: "Model to download if no modelPath is provided")
    var model: String?

    @Option(help: "Text to add in front of the model name to specify between different types of the same variant (values: \"openai\", \"distil\")")
    var modelPrefix: String = "openai"

    @Option(help: "Path to save the downloaded model")
    var downloadModelPath: String?

    @Option(help: "Path to save the downloaded tokenizer files")
    var downloadTokenizerPath: String?

    @Option(help: "Compute units for audio encoder model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine,random}")
    var audioEncoderComputeUnits: ComputeUnits = .cpuAndNeuralEngine

    @Option(help: "Compute units for text decoder model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine,random}")
    var textDecoderComputeUnits: ComputeUnits = .cpuAndNeuralEngine

    @Flag(help: "Verbose mode")
    var verbose: Bool = false

    @Option(help: "Task to perform (transcribe or translate)")
    var task: String = "transcribe"

    @Option(help: "Language spoken in the audio")
    var language: String?

    @Option(help: "Temperature to use for sampling")
    var temperature: Float = 0

    @Option(help: "Temperature to increase on fallbacks during decoding")
    var temperatureIncrementOnFallback: Float = 0.2

    @Option(help: "Number of times to increase temperature when falling back during decoding")
    var temperatureFallbackCount: Int = 5

    @Option(help: "Number of candidates when sampling with non-zero temperature")
    var bestOf: Int = 5

    @Flag(help: "Force initial prompt tokens based on language, task, and timestamp options")
    var usePrefillPrompt: Bool = false

    @Flag(help: "Use decoder prefill data for faster initial decoding")
    var usePrefillCache: Bool = false

    @Flag(help: "Skip special tokens in the output")
    var skipSpecialTokens: Bool = false

    @Flag(help: "Force no timestamps when decoding")
    var withoutTimestamps: Bool = false

    @Flag(help: "Add timestamps for each word in the output")
    var wordTimestamps: Bool = false

    @Argument(help: "Supress given tokens in the output")
    var supressTokens: [Int] = []

    @Option(help: "Gzip compression ratio threshold for decoding failure")
    var compressionRatioThreshold: Float?

    @Option(help: "Average log probability threshold for decoding failure")
    var logprobThreshold: Float?

    @Option(help: "Probability threshold to consider a segment as silence")
    var noSpeechThreshold: Float?

    @Flag(help: "Output a report of the results")
    var report: Bool = false

    @Option(help: "Directory to save the report")
    var reportPath: String = "."

    @Flag(help: "Process audio directly from the microphone")
    var stream: Bool = false
}
