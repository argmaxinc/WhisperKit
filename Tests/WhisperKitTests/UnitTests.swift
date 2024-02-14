//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import CoreML
import Tokenizers
@testable import WhisperKit
import XCTest

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
final class UnitTests: XCTestCase {
    func testInit() async {
        let whisperKit = try? await WhisperKit(prewarm: false, load: false, download: false)
        XCTAssertNotNil(whisperKit)
    }

    // MARK: - Model Loading Test

    func testInitTiny() async {
        let modelPath = tinyModelPath()
        let whisperKit = try? await WhisperKit(modelFolder: modelPath, logLevel: .error)
        XCTAssertNotNil(whisperKit)
    }

    // MARK: - Audio Tests

    func testAudioFileLoading() {
        guard let audioFilePath = Bundle.module.path(forResource: "jfk", ofType: "wav") else {
            XCTFail("Audio file not found")
            return
        }
        let audioBuffer = AudioProcessor.loadAudio(fromPath: audioFilePath)
        XCTAssertNotNil(audioBuffer, "Failed to load audio file at path: \(audioFilePath)")
        XCTAssertEqual(audioBuffer!.format.sampleRate, 16000)
        XCTAssertEqual(audioBuffer!.format.channelCount, 1)
    }

    func testAudioPad() {
        let audioSamples = [Float](repeating: 0.0, count: 1000)
        let paddedSamples = AudioProcessor.padOrTrimAudio(fromArray: audioSamples, startAt: 0, toLength: 1600)
        XCTAssertNotNil(paddedSamples, "Failed to pad audio samples")
        XCTAssertEqual(paddedSamples?.count, 1600, "Padded or trimmed samples count is not as expected")
    }

    func testAudioTrim() {
        let audioSamples = [Float](repeating: 0.0, count: 2000)
        let paddedSamples = AudioProcessor.padOrTrimAudio(fromArray: audioSamples, startAt: 0, toLength: 1600)
        XCTAssertNotNil(paddedSamples, "Failed to trim audio samples")
        XCTAssertEqual(paddedSamples?.count, 1600, "Padded or trimmed samples count is not as expected")
    }

    func testAudioResample() {
        guard let audioFileURL = Bundle.module.url(forResource: "jfk", withExtension: "wav") else {
            XCTFail("Audio file not found")
            return
        }
        let audioFile = try? AVAudioFile(forReading: audioFileURL)

        let targetSampleRate = 44100.0
        let targetChannelCount: AVAudioChannelCount = 2
        let resampledAudio = AudioProcessor.resampleAudio(fromFile: audioFile!, toSampleRate: targetSampleRate, channelCount: 2)
        XCTAssertNotNil(resampledAudio, "Failed to resample audio")
        XCTAssertEqual(resampledAudio?.format.sampleRate, targetSampleRate, "Resampled audio sample rate is not as expected")
        XCTAssertEqual(resampledAudio?.format.channelCount, targetChannelCount, "Resampled audio channels is not as expected")
    }

    func testAudioEnergy() {
        let samples = [Float](repeating: 0.0, count: 16000)
        let silence = samples.map { _ in Float(0.0) }
        let energy = AudioProcessor.calculateEnergy(of: silence).avg
        XCTAssertEqual(energy, 0.0, "Audio energy is not silent")

        let loudNoise = samples.map { _ in Float.random(in: -1...1) }
        let energyLoud = AudioProcessor.calculateEnergy(of: loudNoise).avg
        XCTAssertGreaterThan(energyLoud, energy, "Audio energy is not loud")

        let veryLoudNoise = samples.map { _ in Float.random(in: -10...10) }
        let energyVeryLoud = AudioProcessor.calculateEnergy(of: veryLoudNoise).avg
        XCTAssertGreaterThan(energyVeryLoud, energyLoud, "Audio energy is not very loud")
    }

    // MARK: - Feature Extractor Tests

    func testLogmelOutput() async {
        let audioSamples = [Float](repeating: 0.0, count: 16000)
        guard let paddedSamples = AudioProcessor.padOrTrimAudio(fromArray: audioSamples, startAt: 0, toLength: 480_000) else {
            XCTFail("Failed to pad audio samples")
            return
        }
        var featureExtractor = FeatureExtractor()
        let modelPath = URL(filePath: tinyModelPath()).appending(path: "MelSpectrogram.mlmodelc")
        try? await featureExtractor.loadModel(at: modelPath, computeUnits: ModelComputeOptions().melCompute)
        guard let melSpectrogram = try? await featureExtractor.logMelSpectrogram(fromAudio: paddedSamples) else {
            XCTFail("Failed to produce Mel spectrogram from audio samples")
            return
        }
        let expectedShape: [NSNumber] = [1, 80, 1, 3000]
        XCTAssertNotNil(melSpectrogram, "Failed to produce Mel spectrogram from audio samples")
        XCTAssertEqual(melSpectrogram.shape, expectedShape, "Mel spectrogram shape is not as expected")
    }

    func testCompressionRatioIntArray() {
        let uniqueArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let uniqueRatio = compressionRatio(of: uniqueArray)
        let repeatedArray = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        let repeatedRatio = compressionRatio(of: repeatedArray)
        let repeatedLongArray = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        let repeatedLongRatio = compressionRatio(of: repeatedLongArray)

        XCTAssertLessThan(uniqueRatio, repeatedRatio)
        XCTAssertLessThan(repeatedRatio, repeatedLongRatio)
    }

    func testCompressionRatioString() {
        let uniqueString = "This is a unique string"
        let uniqueRatio = compressionRatio(of: uniqueString)
        let repeatedString = String(repeating: "Repeated text string", count: 5)
        let repeatedRatio = compressionRatio(of: repeatedString)
        let repeatedLongString = String(repeating: "Longer repeated text string", count: 10)
        let repeatedLongRatio = compressionRatio(of: repeatedLongString)

        XCTAssertLessThan(uniqueRatio, repeatedRatio)
        XCTAssertLessThan(repeatedRatio, repeatedLongRatio)
    }

    // MARK: Encoder Tests

    func testEncoderOutput() async {
        var audioEncoder = AudioEncoder()
        let modelPath = URL(filePath: tinyModelPath()).appending(path: "AudioEncoder.mlmodelc")
        try? await audioEncoder.loadModel(at: modelPath, computeUnits: ModelComputeOptions().audioEncoderCompute)

        let encoderInput = try! MLMultiArray(shape: [1, 80, 1, 3000], dataType: .float16)
        let expectedShape: [NSNumber] = [1, 384, 1, 1500]

        let encoderOutput = try? await audioEncoder.encodeFeatures(encoderInput)
        XCTAssertNotNil(encoderOutput, "Failed to encode features")
        XCTAssertEqual(encoderOutput?.shape, expectedShape, "Encoder output shape is not as expected")
    }

    // MARK: Decoder Tests

    func testDecoderOutput() async {
        var textDecoder = TextDecoder()
        let decodingOptions = DecodingOptions()
        let modelPath = URL(filePath: tinyModelPath()).appending(path: "TextDecoder.mlmodelc")
        do {
            try await textDecoder.loadModel(at: modelPath, computeUnits: ModelComputeOptions().textDecoderCompute)
            textDecoder.tokenizer = try await loadTokenizer(for: .tiny)
        } catch {
            XCTFail("Failed to load the model/tokenizer \(error)")
            return
        }

        let tokenSampler = GreedyTokenSampler(temperature: 0, eotToken: textDecoder.tokenizer!.endToken, decodingOptions: decodingOptions)

        let encoderInput = try! MLMultiArray(shape: [1, 384, 1, 1500], dataType: .float16)
        let decoderInputs = textDecoder.prepareDecoderInputs(withPrompt: [textDecoder.tokenizer!.startOfTranscriptToken])
        let expectedShape = 1

        guard let inputs = decoderInputs else {
            XCTFail("Failed to prepare decoder inputs")
            return
        }

        let decoderOutput = try! await textDecoder.decodeText(from: encoderInput, using: inputs, sampler: tokenSampler, options: decodingOptions)
        XCTAssertNotNil(decoderOutput, "Failed to decode text")
        XCTAssertEqual(decoderOutput.count, expectedShape, "Decoder output shape is not as expected")
    }

    // MARK: - Tokenizer Tests

    func testDecoderTokenizer() async {
        // This token index does not change with v3
        let tokenText = "<|startoftranscript|>"

        let textDecoder = TextDecoder()
        textDecoder.tokenizer = try! await loadTokenizer(for: .tiny)
        let encodedToken = textDecoder.tokenizer!.convertTokenToId(tokenText)!
        let decodedToken = textDecoder.tokenizer!.decode(tokens: [encodedToken])

        textDecoder.tokenizer = try! await loadTokenizer(for: .largev3)
        let encodedTokenLarge = textDecoder.tokenizer!.convertTokenToId(tokenText)!
        let decodedTokenLarge = textDecoder.tokenizer?.decode(tokens: [encodedTokenLarge])

        // Test successful tokenizing
        XCTAssertNotNil(decodedToken)
        XCTAssertNotNil(decodedTokenLarge)
        XCTAssertEqual(tokenText, decodedToken)
        XCTAssertEqual(tokenText, decodedTokenLarge)
        XCTAssertEqual(decodedToken, decodedTokenLarge)

        // Test non shifted tokens are equal
        XCTAssertEqual(encodedToken, encodedTokenLarge)

        // This token index changes with v3
        let tokenTextShifted = "<|0.00|>"

        textDecoder.tokenizer = try? await loadTokenizer(for: .tiny)
        let encodedTokenShifted = textDecoder.tokenizer!.convertTokenToId(tokenTextShifted)!
        let decodedTokenShifted = textDecoder.tokenizer?.decode(tokens: [encodedTokenShifted])

        textDecoder.tokenizer = try? await loadTokenizer(for: .largev3)
        let encodedTokenLargeShifted = textDecoder.tokenizer!.convertTokenToId(tokenTextShifted)!
        let decodedTokenLargeShifted = textDecoder.tokenizer?.decode(tokens: [encodedTokenLargeShifted])

        // Test success tokenizing
        XCTAssertNotNil(decodedTokenShifted)
        XCTAssertNotNil(decodedTokenLargeShifted)
        XCTAssertEqual(tokenTextShifted, decodedTokenShifted)
        XCTAssertEqual(tokenTextShifted, decodedTokenLargeShifted)

        // Test shifted tokens are not equal
        XCTAssertNotEqual(encodedTokenShifted, encodedTokenLargeShifted)
    }

    func testTokenizerOutput() async {
        let tokenInputs = [50364, 400, 370, 452, 7177, 6280, 1029, 406, 437, 428, 1941, 393, 360, 337, 291, 1029, 437, 291, 393, 360, 337, 428, 1941, 13, 50889]

        let tokenizer = try? await loadTokenizer(for: .largev3)

        let decodedText = tokenizer!.decode(tokens: tokenInputs)

        XCTAssertNotNil(decodedText)
        XCTAssertEqual(decodedText, "<|notimestamps|> And so my fellow Americans ask not what your country can do for you ask what you can do for your country.<|10.48|>")
    }

    func testWindowing() async {
        let computeOptions = ModelComputeOptions()
        let whisperKit = try? await WhisperKit(modelFolder: tinyModelPath(), computeOptions: computeOptions, verbose: true, logLevel: .debug)

        guard let audioFilePath = Bundle.module.path(forResource: "jfk", ofType: "wav") else {
            XCTFail("Audio file not found")
            return
        }
        guard let audioBuffer = AudioProcessor.loadAudio(fromPath: audioFilePath) else {
            XCTFail("Failed to load audio buffer")
            return
        }

        let audioSamples = AudioProcessor.convertBufferToArray(buffer: audioBuffer)
        let silence = [Float](repeating: 0.0, count: 30 * 16000)
        let multiWindowSamples = audioSamples + silence + audioSamples

        let options = DecodingOptions(usePrefillPrompt: true, withoutTimestamps: false)

        let result = try? await whisperKit!.transcribe(audioArray: multiWindowSamples, decodeOptions: options)
        XCTAssertEqual(result?.segments.count, 2, "Expected 2 segments")

        // Compare last timestamp to the length of the audio
        guard let endTimestamp = result?.segments.last!.end else {
            XCTFail("Failed to get end time")
            return
        }
        XCTAssertEqual(endTimestamp, Float(multiWindowSamples.count / 16000), accuracy: 1.0, "Expected last timestamp to be near the length of the audio")
    }

    // MARK: - Options Tests

    func testSampleLength() async {
        let desiredDecodingLoops = 5
        let targetTokenCount = 7 // Account for the first token and the end of transcript token, which dont require decoding loops

        let options = [
            DecodingOptions(sampleLength: desiredDecodingLoops, usePrefillPrompt: false, skipSpecialTokens: false),
            DecodingOptions(sampleLength: desiredDecodingLoops, usePrefillPrompt: true, skipSpecialTokens: false),
            DecodingOptions(sampleLength: desiredDecodingLoops, usePrefillPrompt: false, skipSpecialTokens: true),
            DecodingOptions(sampleLength: desiredDecodingLoops, usePrefillPrompt: true, skipSpecialTokens: true),
        ]

        for option in options {
            guard let result = try? await transcribe(with: .tiny, options: option) else {
                XCTFail("Failed to transcribe")
                return
            }
            XCTAssertEqual(result.segments.first?.tokens.count, targetTokenCount)
        }
    }

    // Multilingual Tests
    // NOTE: These are purely for consistency checks and do not reflect the ground truth translations
    func testTranslateSpanish() async {
        let targetLanguage = "es"
        let options = DecodingOptions(task: .translate, language: targetLanguage, temperatureFallbackCount: 0)

        guard let result = try? await transcribe(with: .tiny, options: options, audioFile: "es_test_clip.wav") else {
            XCTFail("Failed to transcribe")
            return
        }

        XCTAssertEqual(result.text.split(separator: " ").prefix(2).joined(separator: " "), "This is")
    }

    func testTranscribeSpanish() async {
        let sourceLanguage = "es"
        let options = DecodingOptions(task: .transcribe, language: sourceLanguage, temperatureFallbackCount: 0)

        guard let result = try? await transcribe(with: .tiny, options: options, audioFile: "es_test_clip.wav") else {
            XCTFail("Failed to transcribe")
            return
        }
        XCTAssertEqual(result.text.split(separator: " ").prefix(4).joined(separator: " "), "Esta es una grabación")
    }

    func testTranslateJapanese() async {
        let targetLanguage = "ja"
        let options = DecodingOptions(task: .translate, language: targetLanguage, temperatureFallbackCount: 0)

        guard let result = try? await transcribe(with: .tiny, options: options, audioFile: "ja_test_clip.wav") else {
            XCTFail("Failed to transcribe")
            return
        }

        XCTAssertEqual(result.text.split(separator: " ").first, "Tokyo")
    }

    func testTranscribeJapanese() async {
        let sourceLanguage = "ja"
        let options = DecodingOptions(task: .transcribe, language: sourceLanguage, temperatureFallbackCount: 0)

        guard let result = try? await transcribe(with: .tiny, options: options, audioFile: "ja_test_clip.wav") else {
            XCTFail("Failed to transcribe")
            return
        }
        XCTAssertEqual(result.text.prefix(4), "東京は晴")
    }

    func testNoTimestamps() async {
        let options = DecodingOptions(withoutTimestamps: true)

        guard let result = try? await transcribe(with: .tiny, options: options) else {
            XCTFail("Failed to transcribe")
            return
        }

        XCTAssertEqual(result.segments.first?.text.normalized, "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.<|endoftext|>".normalized)
    }

    func testSkipSpecialTokens() async {
        let options = DecodingOptions(skipSpecialTokens: true, withoutTimestamps: true)

        guard let result = try? await transcribe(with: .tiny, options: options) else {
            XCTFail("Failed to transcribe")
            return
        }

        XCTAssertEqual(result.segments.first?.text.normalized, " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.".normalized)
    }

    func testPrefill() async {
        let options = DecodingOptions(usePrefillPrompt: true)

        do {
            let result = try await transcribe(with: .tiny, options: options)
            XCTAssertNotNil(result?.text)
        } catch {
            XCTFail("Failed to transcribe \(error.localizedDescription)")
        }
    }

    func testNoPrefill() async {
        let options = DecodingOptions(usePrefillPrompt: false)

        guard let result = try? await transcribe(with: .tiny, options: options) else {
            XCTFail("Failed to transcribe")
            return
        }
        XCTAssertNotNil(result.text)
    }

    func testSilence() async {
        let whisperKit = try? await WhisperKit(modelFolder: tinyModelPath(), verbose: true, logLevel: .debug)

        let audioSamples = [Float](repeating: 0.0, count: 30 * 16000)

        let options = DecodingOptions(usePrefillPrompt: false, skipSpecialTokens: false)

        guard let result = try? await whisperKit!.transcribe(audioArray: audioSamples, decodeOptions: options) else {
            XCTFail("Failed to transcribe")
            return
        }
        XCTAssertTrue(result.segments.first!.tokens.contains(whisperKit!.tokenizer!.noSpeechToken))
    }

    func testTemperatureIncrement() async {
        let whisperKit = try? await WhisperKit(modelFolder: tinyModelPath(), verbose: true, logLevel: .debug)

        // Generate random audio samples
        let audioSamples = (0..<(30 * 16000)).map { _ in Float.random(in: -0.5...0.5) }

        // Define options with temperature increment settings
        let initialTemperature: Float = 0
        let temperatureIncrement: Float = 0.1
        let fallbackCount = 1
        let options = DecodingOptions(
            temperature: initialTemperature,
            temperatureIncrementOnFallback: temperatureIncrement,
            temperatureFallbackCount: fallbackCount,
            logProbThreshold: 0
        )

        // Perform transcription
        guard let result = try? await whisperKit!.transcribe(audioArray: audioSamples, decodeOptions: options) else {
            XCTFail("Failed to transcribe")
            return
        }

        let expectedTemperature = initialTemperature + temperatureIncrement * Float(fallbackCount)
        XCTAssertEqual(result.segments.first!.temperature, expectedTemperature, "Temperature was not incremented correctly after fallbacks")
    }

    func testTopK() async {
        var options = DecodingOptions(temperature: 0.5, topK: 10000)

        guard let result10000 = try? await transcribe(with: .tiny, options: options) else {
            XCTFail("Failed to transcribe")
            return
        }

        options = DecodingOptions(temperature: 0.5)

        guard let result5 = try? await transcribe(with: .tiny, options: options) else {
            XCTFail("Failed to transcribe")
            return
        }

        XCTAssertLessThan(Float(result5.timings!.decodingSampling), Float(result10000.timings!.decodingSampling), "topK=5 should be faster than topK=10000")
    }

    func testSeekClips() async {
        var options = DecodingOptions(clipTimestamps: [0])

        guard let resultFull = try? await transcribe(with: .tiny, options: options) else {
            XCTFail("Failed to transcribe")
            return
        }

        let seekTime: Float = 3.0
        options = DecodingOptions(clipTimestamps: [seekTime])

        guard let resultSeek = try? await transcribe(with: .tiny, options: options) else {
            XCTFail("Failed to transcribe")
            return
        }

        XCTAssertNotEqual(resultFull.text, resultSeek.text)
        XCTAssertTrue(resultFull.text.normalized.contains(resultSeek.text.normalized), "Seeking should be a subset of the full clip")
        XCTAssertFalse(resultSeek.text.normalized.contains(resultFull.text.normalized), "Seeking should be a subset of the full clip")
        XCTAssertEqual(resultSeek.segments.first?.start, seekTime, "Seek segment should have the input start time")
        XCTAssertNotEqual(resultFull.segments.first?.start, resultSeek.segments.first?.start, "Segments should have the different start times")
        XCTAssertEqual(resultFull.segments.first?.end, resultSeek.segments.first?.end, "Segments should have the same end time")
    }

    // MARK: - Utils Tests

    func testFillIndexesWithValue() throws {
        let logits = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        logits.fill(indexes: [], with: -FloatType.infinity)
        XCTAssertEqual(logits.data(for: 2), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        let logits2 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        let indexes2: [[NSNumber]] = [[0, 0, 0], [0, 0, 1], [0, 0, 5]]
        logits2.fill(indexes: indexes2, with: -FloatType.infinity)
        XCTAssertEqual(logits2.data(for: 2), [-.infinity, -.infinity, 0.3, 0.4, 0.5, -.infinity, 0.7])
    }

    // MARK: - LogitsFilter Tests

    func testSuppressTokensFilter() throws {
        let tokensFilter1 = SuppressTokensFilter(suppressTokens: [])
        let logits1 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        let result1 = tokensFilter1.filterLogits(logits1, withTokens: [])
        XCTAssertEqual(result1.data(for: 2), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        let tokensFilter2 = SuppressTokensFilter(suppressTokens: [0])
        let logits2 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        let result2 = tokensFilter2.filterLogits(logits2, withTokens: [])
        XCTAssertEqual(result2.data(for: 2), [-.infinity, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        let tokensFilter3 = SuppressTokensFilter(suppressTokens: [0, 2, 5, 6])
        let logits3 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        let result3 = tokensFilter3.filterLogits(logits3, withTokens: [])
        XCTAssertEqual(result3.data(for: 2), [-.infinity, 0.2, -.infinity, 0.4, 0.5, -.infinity, -.infinity])
    }

    func testSuppressBlankFilter() throws {
        let tokensFilter1 = SuppressBlankFilter(suppressBlankTokens: [], sampleBegin: 0)
        let logits1 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        let result1 = tokensFilter1.filterLogits(logits1, withTokens: [])
        XCTAssertEqual(result1.data(for: 2), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        let tokensFilter2 = SuppressBlankFilter(suppressBlankTokens: [0], sampleBegin: 0)
        let logits2 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        let result2 = tokensFilter2.filterLogits(logits2, withTokens: [])
        XCTAssertEqual(result2.data(for: 2), [-.infinity, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        let tokensFilter3 = SuppressBlankFilter(suppressBlankTokens: [0, 2, 6], sampleBegin: 0)
        let logits3 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        let result3 = tokensFilter3.filterLogits(logits3, withTokens: [])
        XCTAssertEqual(result3.data(for: 2), [-.infinity, 0.2, -.infinity, 0.4, 0.5, 0.6, -.infinity])

        let tokensFilter4 = SuppressBlankFilter(suppressBlankTokens: [0, 2, 6], sampleBegin: 3)
        let logits4 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        let result4 = tokensFilter4.filterLogits(logits4, withTokens: [1, 2, 3])
        XCTAssertEqual(result4.data(for: 2), [-.infinity, 0.2, -.infinity, 0.4, 0.5, 0.6, -.infinity])

        let tokensFilter5 = SuppressBlankFilter(suppressBlankTokens: [0, 2, 6], sampleBegin: 5)
        let logits5 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        let result5 = tokensFilter5.filterLogits(logits5, withTokens: [1, 2, 3])
        XCTAssertEqual(result5.data(for: 2), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    }
}

// MARK: Helpers

extension MLMultiArray {
    /// Create `MLMultiArray` of shape [1, 1, arr.count] and fill up the last
    /// dimension with with values from arr.
    static func logits(_ arr: [FloatType]) throws -> MLMultiArray {
        let logits = try MLMultiArray(shape: [1, 1, arr.count] as [NSNumber], dataType: .float16)
        let ptr = UnsafeMutablePointer<FloatType>(OpaquePointer(logits.dataPointer))
        for (index, value) in arr.enumerated() {
            let linearOffset = logits.linearOffset(for: [0, 0, index as NSNumber])
            ptr[linearOffset] = value
        }
        return logits
    }

    /// Get the data from `MLMultiArray` for given dimension
    func data(for dimension: Int) -> [FloatType] {
        let count = shape[dimension].intValue
        let indexes = stride(from: 0, to: count, by: 1).map { [0, 0, $0 as NSNumber] }
        var result = [FloatType]()
        let ptr = UnsafeMutablePointer<FloatType>(OpaquePointer(dataPointer))
        for index in indexes {
            let linearOffset = linearOffset(for: index as [NSNumber])
            result.append(ptr[linearOffset])
        }
        return result
    }
}

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
extension XCTestCase {
    func transcribe(with variant: ModelVariant, options: DecodingOptions, audioFile: String = "jfk.wav", file: StaticString = #file, line: UInt = #line) async throws -> TranscriptionResult? {
        var modelPath = tinyModelPath()
        switch variant {
        case .largev3:
            modelPath = largev3ModelPath()
        default:
            modelPath = tinyModelPath()
        }
        let computeOptions = ModelComputeOptions(
            melCompute: .cpuOnly,
            audioEncoderCompute: .cpuOnly,
            textDecoderCompute: .cpuOnly,
            prefillCompute: .cpuOnly
        )
        let whisperKit = try await WhisperKit(modelFolder: modelPath, computeOptions: computeOptions, verbose: true, logLevel: .debug)
        trackForMemoryLeaks(on: whisperKit, file: file, line: line)

        let audioComponents = audioFile.components(separatedBy: ".")
        guard let audioFileURL = Bundle.module.path(forResource: audioComponents.first, ofType: audioComponents.last) else {
            return nil
        }

        let result = try await whisperKit.transcribe(audioPath: audioFileURL, decodeOptions: options)
        return result
    }

    func tinyModelPath() -> String {
        let modelDir = "whisperkit-coreml/openai_whisper-tiny"
        guard let modelPath = Bundle.module.urls(forResourcesWithExtension: "mlmodelc", subdirectory: modelDir)?.first?.deletingLastPathComponent().path else {
            print("Failed to load model, ensure \"Models/\(modelDir)\" exists via Makefile command: `make download-models`")
            return ""
        }
        return modelPath
    }

    func largev3ModelPath() -> String {
        let modelDir = "whisperkit-coreml/openai_whisper-large-v3" // use faster to compile model for tests
        guard let modelPath = Bundle.module.urls(forResourcesWithExtension: "mlmodelc", subdirectory: modelDir)?.first?.deletingLastPathComponent().path else {
            print("Failed to load model, ensure \"Models/\(modelDir)\" exists via Makefile command: `make download-models`")
            return ""
        }
        return modelPath
    }

    func allModelPaths() -> [String] {
        let fileManager = FileManager.default
        var modelPaths: [String] = []
        let directory = "whisperkit-coreml"

        do {
            let resourceKeys: [URLResourceKey] = [.isDirectoryKey]
            guard let baseurl = Bundle.module.resourceURL?.appendingPathComponent(directory) else {
                print("Base URL for directory \(directory) not found.")
                return []
            }

            let directoryContents = try fileManager.contentsOfDirectory(at: baseurl, includingPropertiesForKeys: resourceKeys, options: .skipsHiddenFiles)

            for folderURL in directoryContents {
                let resourceValues = try folderURL.resourceValues(forKeys: Set(resourceKeys))
                if resourceValues.isDirectory == true {
                    // Check if the directory name contains the quantization pattern
                    // Only test large quantized models
                    let dirName = folderURL.lastPathComponent
                    if !(dirName.contains("q") && !dirName.contains("large")) {
                        modelPaths.append(folderURL.absoluteString)
                    }
                }
            }
        } catch {
            print(error.localizedDescription)
        }

        return modelPaths
    }

    func trackForMemoryLeaks(on instance: AnyObject, file: StaticString = #filePath, line: UInt = #line) {
        addTeardownBlock { [weak instance] in
            XCTAssertNil(instance, "Detected potential memory leak", file: file, line: line)
        }
    }
}

extension String {
    var normalized: String {
        // Trim whitespace and newlines
        let trimmedString = self.trimmingCharacters(in: .whitespacesAndNewlines)

        // Convert to lowercase
        let lowercaseString = trimmedString.lowercased()

        // Remove punctuation
        let noPunctuationString = lowercaseString.components(separatedBy: .punctuationCharacters).joined()

        // Replace multiple spaces with a single space
        let singleSpacedString = noPunctuationString.replacingOccurrences(of: " +", with: " ", options: .regularExpression)

        return singleSpacedString
    }
}
