//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import CoreML
import Tokenizers
import Hub
@testable import WhisperKit
import XCTest

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
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

    // MARK: - Encoder Tests

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

    // MARK: - Decoder Tests

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
        let computeOptions = ModelComputeOptions(
            melCompute: .cpuOnly
        )
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
        XCTAssertEqual(result?.segments.count, 3, "Expected 3 segments")

        // Compare last timestamp to the length of the audio
        guard let endTimestamp = result?.segments.last!.end else {
            XCTFail("Failed to get end time")
            return
        }
        XCTAssertEqual(endTimestamp, Float(multiWindowSamples.count / 16000), accuracy: 1.0, "Expected last timestamp to be near the length of the audio")
    }

    func testSplitToWordTokens() async {
        let tokenizer = try? await loadTokenizer(for: .tiny)

        // Hello, world! This is a test, isn't it?
        let tokenIds = [50364, 2425, 11, 1002, 0, 50414, 50414, 639, 307, 257, 220, 31636, 11, 1943, 380, 309, 30, 50257]
        let originalWords = tokenIds.map { tokenizer!.convertIdToToken($0) }

        let (words, wordTokens) = tokenizer!.splitToWordTokens(tokenIds: tokenIds)

        let expectedWords = ["<|0.00|>", " Hello", ",", " world", "!", "<|1.00|>", "<|1.00|>", " This", " is", " a", " test", ",", " isn't", " it", "?", "<|endoftext|>"]
        let expectedWordTokens = [[50364], [2425], [11], [1002], [0], [50414], [50414], [639], [307], [257], [220, 31636], [11], [1943, 380], [309], [30], [50257]]

        XCTAssertNotEqual(originalWords, words, "Should not directly convert into tokens from ids")
        XCTAssertEqual(words, expectedWords, "Words did not match expected output.")
        XCTAssertEqual(wordTokens, expectedWordTokens, "Word tokens did not match expected output.")
    }

    func testSplitToWordTokensSpanish() async {
        let tokenizer = try? await loadTokenizer(for: .tiny)

        // ¡Hola Mundo! Esta es una prueba, ¿no?
        let tokenIds = [50363, 24364, 48529, 376, 6043, 0, 20547, 785, 2002, 48241, 11, 3841, 1771, 30, 50257]
        let originalWords = tokenIds.map { tokenizer!.convertIdToToken($0) }

        let (words, wordTokens) = tokenizer!.splitToWordTokens(tokenIds: tokenIds)

        let expectedWords = ["<|notimestamps|>", "¡Hola", " Mundo", "!", " Esta", " es", " una", " prueba", ",", " ¿no", "?", "<|endoftext|>"]
        let expectedWordTokens = [[50363], [24364, 48529], [376, 6043], [0], [20547], [785], [2002], [48241], [11], [3841, 1771], [30], [50257]]

        XCTAssertNotEqual(originalWords, words, "Should not directly convert into tokens from ids")
        XCTAssertEqual(words, expectedWords, "Words did not match expected output.")
        XCTAssertEqual(wordTokens, expectedWordTokens, "Word tokens did not match expected output.")
    }

    func testSplitToWordTokensJapanese() async {
        let tokenizer = try? await loadTokenizer(for: .tiny)

        // こんにちは、世界！これはテストですよね？
        let tokenIds = [50364, 38088, 1231, 24486, 171, 120, 223, 25212, 22985, 40498, 4767, 30346, 171, 120, 253, 50257]
        let originalWords = tokenIds.map { tokenizer!.convertIdToToken($0) }

        let (words, wordTokens) = tokenizer!.splitToWordTokens(tokenIds: tokenIds)

        let expectedWords = ["<|0.00|>", "こんにちは", "、", "世界", "！", "これは", "テ", "スト", "です", "よね", "？", "<|endoftext|>"]
        let expectedWordTokens = [[50364], [38088], [1231], [24486], [171, 120, 223], [25212], [22985], [40498], [4767], [30346], [171, 120, 253], [50257]]

        XCTAssertNotEqual(originalWords, words, "Should not directly convert into tokens from ids")
        XCTAssertEqual(words, expectedWords, "Words did not match expected output in Unicode split.")
        XCTAssertEqual(wordTokens, expectedWordTokens, "Word tokens did not match expected output in Unicode split.")
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

    /// Multilingual Tests
    /// NOTE: These are purely for consistency checks and do not reflect the ground truth translations
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
        XCTAssertEqual(result.text.prefix(3), "東京は")
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
        var options = DecodingOptions(withoutTimestamps: true, clipTimestamps: [0])

        guard let resultFull = try? await transcribe(with: .tiny, options: options) else {
            XCTFail("Failed to transcribe")
            return
        }

        let seekTime: Float = 3.0
        options = DecodingOptions(withoutTimestamps: true, clipTimestamps: [seekTime])

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
        logits.fill(indexes: [] as [[NSNumber]], with: -FloatType.infinity)
        XCTAssertEqual(logits.data(for: 2), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        let logits2 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        let indexes2: [[NSNumber]] = [[0, 0, 0], [0, 0, 1], [0, 0, 5]]
        logits2.fill(indexes: indexes2, with: -FloatType.infinity)
        XCTAssertEqual(logits2.data(for: 2), [-.infinity, -.infinity, 0.3, 0.4, 0.5, -.infinity, 0.7])

        let logits3 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        logits3.fillLastDimension(indexes: 0..<1, with: -FloatType.infinity)
        XCTAssertEqual(logits3.data(for: 2), [-.infinity, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        let logits4 = try MLMultiArray.logits([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        logits4.fillLastDimension(indexes: 2..<5, with: -FloatType.infinity)
        XCTAssertEqual(logits4.data(for: 2), [0.1, 0.2, -.infinity, -.infinity, -.infinity, 0.6, 0.7])
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
    
    func testTimestampRulesFilter() throws {
        // NOTE: for non-multilingual models we supress tokens immediately
        let tokensFilter1 = TimestampRulesFilter(
            transcribeToken: 100,
            translateToken: 101,
            noTimestampsToken: 2,
            timeTokenBegin: 4,
            endToken: 3,
            sampleBegin: 2,
            maxInitialTimestampIndex: nil,
            isModelMultilingual: false
        )
        let logits1 = try MLMultiArray.logits([1.1, 5.2, 0.3, 0.4, 0.2, 0.1, 0.2])
        let result1 = tokensFilter1.filterLogits(logits1, withTokens: [])
        XCTAssertEqual(result1.data(for: 2), [1.1, 5.2, -.infinity, 0.4, 0.2, 0.1, 0.2])

        let tokensFilter2 = TimestampRulesFilter(
            transcribeToken: 100,
            translateToken: 101,
            noTimestampsToken: 2,
            timeTokenBegin: 4,
            endToken: 3,
            sampleBegin: 2,
            maxInitialTimestampIndex: nil,
            isModelMultilingual: false
        )
        let logits2 = try MLMultiArray.logits([1.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.2])
        let result2 = tokensFilter2.filterLogits(logits2, withTokens: [])
        XCTAssertEqual(result2.data(for: 2), [-.infinity, -.infinity, -.infinity, -.infinity, 0.2, 0.1, 0.2])
    }

    func testTimestampRulesFilterMultilingual() throws {
        // NOTE: for multilingual models we supress tokens only after transcribe or translate token
        let tokensFilter1 = TimestampRulesFilter(
            transcribeToken: 100,
            translateToken: 101,
            noTimestampsToken: 2,
            timeTokenBegin: 4,
            endToken: 3,
            sampleBegin: 2,
            maxInitialTimestampIndex: nil,
            isModelMultilingual: true
        )
        let logits1 = try MLMultiArray.logits([1.1, 5.2, 0.3, 0.4, 0.2, 0.1, 0.2])
        let result1 = tokensFilter1.filterLogits(logits1, withTokens: [])
        XCTAssertEqual(result1.data(for: 2), [1.1, 5.2, 0.3, 0.4, 0.2, 0.1, 0.2])
        
        let tokensFilter2 = TimestampRulesFilter(
            transcribeToken: 100,
            translateToken: 101,
            noTimestampsToken: 2,
            timeTokenBegin: 4,
            endToken: 3,
            sampleBegin: 2,
            maxInitialTimestampIndex: nil,
            isModelMultilingual: true
        )
        let logits2 = try MLMultiArray.logits([1.1, 5.2, 0.3, 0.4, 0.2, 0.1, 0.2])
        let result2 = tokensFilter2.filterLogits(logits2, withTokens: [100])
        XCTAssertEqual(result2.data(for: 2), [1.1, 5.2, -.infinity, 0.4, 0.2, 0.1, 0.2])

        let tokensFilter3 = TimestampRulesFilter(
            transcribeToken: 100,
            translateToken: 101,
            noTimestampsToken: 2,
            timeTokenBegin: 4,
            endToken: 3,
            sampleBegin: 2,
            maxInitialTimestampIndex: nil,
            isModelMultilingual: true
        )
        let logits3 = try MLMultiArray.logits([1.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.2])
        let result3 = tokensFilter3.filterLogits(logits3, withTokens: [101])
        XCTAssertEqual(result3.data(for: 2), [-.infinity, -.infinity, -.infinity, -.infinity, 0.2, 0.1, 0.2])
    }

    // MARK: - Word Timestamp Tests

    func testDynamicTimeWarpingSimpleMatrix() {
        let matrix = [
            [1.0, 1.0, 1.0],
            [5.0, 2.0, 1.0],
            [1.0, 5.0, 2.0],
        ]

        let numRows = matrix.count
        let numColumns = matrix[0].count
        let mlMatrix = try! MLMultiArray(shape: [numRows, numColumns] as [NSNumber], dataType: .double)
        let ptr = UnsafeMutablePointer<Double>(OpaquePointer(mlMatrix.dataPointer))
        for (i, row) in matrix.enumerated() {
            for (j, value) in row.enumerated() {
                let linearOffset = mlMatrix.linearOffset(for: [i, j] as [NSNumber])
                ptr[linearOffset] = value
            }
        }

        let segmentSeeker = SegmentSeeker()
        do {
            let result = try segmentSeeker.dynamicTimeWarping(withMatrix: mlMatrix)
            let expected = (
                textIndices: [0, 1, 1, 2, 2],
                timeIndices: [0, 0, 1, 1, 2]
            )
            XCTAssertEqual(result.textIndices, expected.textIndices, "dynamicTimeWarping function did not return the expected path.")
            XCTAssertEqual(result.timeIndices, expected.timeIndices, "dynamicTimeWarping function did not return the expected path.")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testDynamicTimeWarpingLargeMatrix() {
        // Create a large matrix with non-linear characteristics
        let numberOfRows: NSNumber = 448
        let numberOfColumns: NSNumber = 1500

        // Populate the matrix with some non-linear data
        let matrix = try! MLMultiArray(shape: [numberOfRows, numberOfColumns], dataType: .float16)
        for i in 0..<numberOfRows.intValue {
            for j in 0..<numberOfColumns.intValue {
                matrix[i * numberOfColumns.intValue + j] = NSNumber(value: Double.random(in: 0...1))
            }
        }

        let segmentSeeker = SegmentSeeker()
        do {
            let result = try segmentSeeker.dynamicTimeWarping(withMatrix: matrix)
            // Validate the output dimensions
            XCTAssertFalse(result.textIndices.isEmpty, "Result should not be empty.")
            XCTAssertFalse(result.timeIndices.isEmpty, "Result should not be empty.")

            // Validate start and end points
            XCTAssertEqual(result.textIndices.first, 0, "Path should start at (0, 0).")
            XCTAssertEqual(result.timeIndices.first, 0, "Path should start at (0, 0).")
            XCTAssertEqual(result.textIndices.last, numberOfRows.intValue - 1, "Path should end at (N-1, M-1).")
            XCTAssertEqual(result.timeIndices.last, numberOfColumns.intValue - 1, "Path should end at (N-1, M-1).")

            // Check path continuity and bounds
            for i in 1..<result.textIndices.count {
                let (prevRow, prevCol) = (result.textIndices[i - 1], result.timeIndices[i - 1])
                let (currentRow, currentCol) = (result.textIndices[i], result.timeIndices[i])

                let rowDiff = currentRow - prevRow
                let colDiff = currentCol - prevCol

                // Assert that the row difference is 0 or 1
                XCTAssertTrue(rowDiff == 0 || rowDiff == 1, "Row difference should be 0 or 1")

                // Assert that the column difference is 0 or 1
                XCTAssertTrue(colDiff == 0 || colDiff == 1, "Column difference should be 0 or 1")

                // Assert that at least one of rowDiff or colDiff is 1
                XCTAssertTrue(rowDiff == 1 || colDiff == 1, "At least one of rowDiff or colDiff should be 1")

                // Assert that rowDiff and colDiff are not both 0
                XCTAssertFalse(rowDiff == 0 && colDiff == 0, "Both rowDiff and colDiff should not be 0 at the same time")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testFindAlignment() async {
        let numberOfRows: NSNumber = 448
        let numberOfColumns: NSNumber = 1500

        // Populate the matrix with some non-linear data
        let matrix = try! MLMultiArray(shape: [numberOfRows, numberOfColumns], dataType: .float16)
        let tokenProbs = Array(repeating: 0.0, count: numberOfRows.intValue).map { _ in Float.random(in: -1..<0) }
        for i in 0..<numberOfRows.intValue {
            for j in 0..<numberOfColumns.intValue {
                matrix[i * numberOfColumns.intValue + j] = NSNumber(value: Double.random(in: 0...1))
            }
        }

        let tokenizer = try! await loadTokenizer(for: .tiny)

        let wordTokenIds = [400, 370, 452, 7177, 6280, 11, 1029, 406, 437, 428, 1941, 393, 360, 337, 291, 11, 1029, 437, 291, 393, 360, 337, 428, 1941, 13]
        do {
            let result = try SegmentSeeker().findAlignment(
                wordTokenIds: wordTokenIds,
                alignmentWeights: matrix,
                tokenLogProbs: tokenProbs,
                tokenizer: tokenizer
            )

            XCTAssertFalse(result.isEmpty, "Result should not be empty.")

            var previousEndTime: Float = -1.0
            for wordTiming in result {
                XCTAssertFalse(wordTiming.word.isEmpty, "Word should not be empty.")
                XCTAssertFalse(wordTiming.tokens.isEmpty, "Tokens should not be empty.")
                XCTAssert(wordTiming.tokens.allSatisfy { $0 >= 0 }, "All token IDs should be non-negative.")
                XCTAssertLessThanOrEqual(wordTiming.start, wordTiming.end, "Start should be less than or equal to end.")
                XCTAssertGreaterThanOrEqual(wordTiming.start, previousEndTime, "Start time should not be earlier than the previous end time.")
                XCTAssertGreaterThanOrEqual(wordTiming.probability, 0.0, "Probability should not be negative.")
                XCTAssertLessThanOrEqual(wordTiming.probability, 1.0, "Probability should not be greater than 1.")

                previousEndTime = wordTiming.end
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testMergePunctuations() async {
        // Hello, world! This is a test, isn't it?
        let wordTimings = [
            WordTiming(word: "<|0.00|>", tokens: [50364], start: 0, end: 1, probability: 1),
            WordTiming(word: " Hello", tokens: [2425], start: 1, end: 2, probability: 1),
            WordTiming(word: ",", tokens: [11], start: 2, end: 3, probability: 1),
            WordTiming(word: " world", tokens: [1002], start: 3, end: 4, probability: 1),
            WordTiming(word: "!", tokens: [0], start: 4, end: 5, probability: 1),
            WordTiming(word: "<|1.00|>", tokens: [50414], start: 5, end: 6, probability: 1),
            WordTiming(word: "<|1.00|>", tokens: [50414], start: 6, end: 7, probability: 1),
            WordTiming(word: " This", tokens: [639], start: 7, end: 8, probability: 1),
            WordTiming(word: " is", tokens: [307], start: 8, end: 9, probability: 1),
            WordTiming(word: " a", tokens: [257], start: 9, end: 10, probability: 1),
            WordTiming(word: " test", tokens: [220, 31636], start: 10, end: 11, probability: 1),
            WordTiming(word: ",", tokens: [11], start: 11, end: 12, probability: 1),
            WordTiming(word: " isn't", tokens: [1943, 380], start: 12, end: 13, probability: 1),
            WordTiming(word: " it", tokens: [309], start: 13, end: 14, probability: 1),
            WordTiming(word: "?", tokens: [30], start: 14, end: 15, probability: 1),
            WordTiming(word: "<|endoftext|>", tokens: [50257], start: 15, end: 16, probability: 1),
        ]

        let mergedAlignmentTiming = SegmentSeeker().mergePunctuations(alignment: wordTimings, prepended: "\"'“¿([{-", appended: "\"'.。,，!！?？:：”)]}、")

        let expectedWordTimings = [
            WordTiming(word: "<|0.00|>", tokens: [50364], start: 0, end: 1, probability: 1),
            WordTiming(word: " Hello,", tokens: [2425, 11], start: 1, end: 3, probability: 1),
            WordTiming(word: " world!", tokens: [1002, 0], start: 3, end: 5, probability: 1),
            WordTiming(word: "<|1.00|>", tokens: [50414], start: 5, end: 6, probability: 1),
            WordTiming(word: "<|1.00|>", tokens: [50414], start: 6, end: 7, probability: 1),
            WordTiming(word: " This", tokens: [639], start: 7, end: 8, probability: 1),
            WordTiming(word: " is", tokens: [307], start: 8, end: 9, probability: 1),
            WordTiming(word: " a", tokens: [257], start: 9, end: 10, probability: 1),
            WordTiming(word: " test,", tokens: [220, 31636, 11], start: 10, end: 12, probability: 1),
            WordTiming(word: " isn't", tokens: [1943, 380], start: 12, end: 13, probability: 1),
            WordTiming(word: " it?", tokens: [309, 30], start: 13, end: 15, probability: 1),
            WordTiming(word: "<|endoftext|>", tokens: [50257], start: 15, end: 16, probability: 1),
        ]

        // First, assert the counts are as expected
        XCTAssertEqual(mergedAlignmentTiming.count, expectedWordTimings.count, "Merged timings count does not match expected count")

        // Then, iterate through each expected timing and assert properties
        for i in 0..<expectedWordTimings.count {
            XCTAssertEqual(mergedAlignmentTiming[i].word, expectedWordTimings[i].word, "Word text at index \(i) does not match")
            XCTAssertEqual(mergedAlignmentTiming[i].tokens, expectedWordTimings[i].tokens, "Tokens at index \(i) do not match")
            XCTAssertEqual(mergedAlignmentTiming[i].start, expectedWordTimings[i].start, "Start time at index \(i) does not match")
            XCTAssertEqual(mergedAlignmentTiming[i].end, expectedWordTimings[i].end, "End time at index \(i) does not match")
            XCTAssertEqual(mergedAlignmentTiming[i].probability, expectedWordTimings[i].probability, "Probability at index \(i) does not match")
        }
    }

    func testMergePunctuationsSpanish() async {
        // Spanish text: ¡Hola Mundo! Esta es una prueba, ¿no?
        let wordTimings = [
            WordTiming(word: "<|notimestamps|>", tokens: [50363], start: 0, end: 1, probability: 1),
            WordTiming(word: "¡Hola", tokens: [24364, 48529], start: 1, end: 2, probability: 1),
            WordTiming(word: " Mundo", tokens: [376, 6043], start: 2, end: 3, probability: 1),
            WordTiming(word: "!", tokens: [0], start: 3, end: 4, probability: 1),
            WordTiming(word: " Esta", tokens: [20547], start: 4, end: 5, probability: 1),
            WordTiming(word: " es", tokens: [785], start: 5, end: 6, probability: 1),
            WordTiming(word: " una", tokens: [2002], start: 6, end: 7, probability: 1),
            WordTiming(word: " prueba", tokens: [48241], start: 7, end: 8, probability: 1),
            WordTiming(word: ",", tokens: [11], start: 8, end: 9, probability: 1),
            WordTiming(word: " ¿no", tokens: [3841, 1771], start: 9, end: 10, probability: 1),
            WordTiming(word: "?", tokens: [30], start: 10, end: 11, probability: 1),
            WordTiming(word: "<|endoftext|>", tokens: [50257], start: 11, end: 12, probability: 1),
        ]

        let mergedAlignmentTiming = SegmentSeeker().mergePunctuations(alignment: wordTimings, prepended: "\"'“¿([{-", appended: "\"'.。,，!！?？:：”)]}、")

        let expectedWordTimings = [
            WordTiming(word: "<|notimestamps|>", tokens: [50363], start: 0, end: 1, probability: 1),
            WordTiming(word: "¡Hola", tokens: [24364, 48529], start: 1, end: 2, probability: 1),
            WordTiming(word: " Mundo!", tokens: [376, 6043, 0], start: 2, end: 4, probability: 1),
            WordTiming(word: " Esta", tokens: [20547], start: 4, end: 5, probability: 1),
            WordTiming(word: " es", tokens: [785], start: 5, end: 6, probability: 1),
            WordTiming(word: " una", tokens: [2002], start: 6, end: 7, probability: 1),
            WordTiming(word: " prueba,", tokens: [48241, 11], start: 7, end: 9, probability: 1),
            WordTiming(word: " ¿no?", tokens: [3841, 1771, 30], start: 9, end: 11, probability: 1),
            WordTiming(word: "<|endoftext|>", tokens: [50257], start: 11, end: 12, probability: 1),
        ]

        // First, assert the counts are as expected
        XCTAssertEqual(mergedAlignmentTiming.count, expectedWordTimings.count, "Merged timings count does not match expected count")

        // Then, iterate through each expected timing and assert properties
        for i in 0..<expectedWordTimings.count {
            XCTAssertEqual(mergedAlignmentTiming[i].word, expectedWordTimings[i].word, "Word text at index \(i) does not match")
            XCTAssertEqual(mergedAlignmentTiming[i].tokens, expectedWordTimings[i].tokens, "Tokens at index \(i) do not match")
            XCTAssertEqual(mergedAlignmentTiming[i].start, expectedWordTimings[i].start, "Start time at index \(i) does not match")
            XCTAssertEqual(mergedAlignmentTiming[i].end, expectedWordTimings[i].end, "End time at index \(i) does not match")
            XCTAssertEqual(mergedAlignmentTiming[i].probability, expectedWordTimings[i].probability, "Probability at index \(i) does not match")
        }
    }

    func testMergePunctuationsJapanese() async {
        // Japanese text: こんにちは、世界！これはテストですよね？
        let wordTimings = [
            WordTiming(word: "<|0.00|>", tokens: [50364], start: 0, end: 1, probability: 1),
            WordTiming(word: "こんにちは", tokens: [38088], start: 1, end: 2, probability: 1),
            WordTiming(word: "、", tokens: [1231], start: 2, end: 3, probability: 1),
            WordTiming(word: "世界", tokens: [24486], start: 3, end: 4, probability: 1),
            WordTiming(word: "！", tokens: [171, 120, 223], start: 4, end: 5, probability: 1),
            WordTiming(word: "これは", tokens: [25212], start: 5, end: 6, probability: 1),
            WordTiming(word: "テ", tokens: [22985], start: 6, end: 7, probability: 1),
            WordTiming(word: "スト", tokens: [40498], start: 7, end: 8, probability: 1),
            WordTiming(word: "です", tokens: [4767], start: 8, end: 9, probability: 1),
            WordTiming(word: "よね", tokens: [30346], start: 9, end: 10, probability: 1),
            WordTiming(word: "？", tokens: [171, 120, 253], start: 10, end: 11, probability: 1),
            WordTiming(word: "<|endoftext|>", tokens: [50257], start: 11, end: 12, probability: 1),
        ]

        let mergedAlignmentTiming = SegmentSeeker().mergePunctuations(alignment: wordTimings, prepended: "\"'“¿([{-", appended: "\"'.。,，!！?？:：”)]}、")

        let expectedWordTimings = [
            WordTiming(word: "<|0.00|>", tokens: [50364], start: 0, end: 1, probability: 1),
            WordTiming(word: "こんにちは、", tokens: [38088, 1231], start: 1, end: 3, probability: 1),
            WordTiming(word: "世界！", tokens: [24486, 171, 120, 223], start: 3, end: 5, probability: 1),
            WordTiming(word: "これは", tokens: [25212], start: 5, end: 6, probability: 1),
            WordTiming(word: "テ", tokens: [22985], start: 6, end: 7, probability: 1),
            WordTiming(word: "スト", tokens: [40498], start: 7, end: 8, probability: 1),
            WordTiming(word: "です", tokens: [4767], start: 8, end: 9, probability: 1),
            WordTiming(word: "よね？", tokens: [30346, 171, 120, 253], start: 9, end: 11, probability: 1),
            WordTiming(word: "<|endoftext|>", tokens: [50257], start: 11, end: 12, probability: 1),
        ]

        // First, assert the counts are as expected
        XCTAssertEqual(mergedAlignmentTiming.count, expectedWordTimings.count, "Merged timings count does not match expected count")

        // Then, iterate through each expected timing and assert properties
        for i in 0..<expectedWordTimings.count {
            XCTAssertEqual(mergedAlignmentTiming[i].word, expectedWordTimings[i].word, "Word text at index \(i) does not match")
            XCTAssertEqual(mergedAlignmentTiming[i].tokens, expectedWordTimings[i].tokens, "Tokens at index \(i) do not match")
            XCTAssertEqual(mergedAlignmentTiming[i].start, expectedWordTimings[i].start, "Start time at index \(i) does not match")
            XCTAssertEqual(mergedAlignmentTiming[i].end, expectedWordTimings[i].end, "End time at index \(i) does not match")
            XCTAssertEqual(mergedAlignmentTiming[i].probability, expectedWordTimings[i].probability, "Probability at index \(i) does not match")
        }
    }

    func testWordTimestampCorrectness() async {
        let options = DecodingOptions(wordTimestamps: true)

        guard let result = try? await transcribe(with: .tiny, options: options) else {
            XCTFail("Failed to transcribe")
            return
        }

        let wordTimings = result.segments.compactMap { $0.words }.flatMap { $0 }

        let expectedWordTimings = [
            WordTiming(word: " And", tokens: [400], start: 0.32, end: 0.68, probability: 0.85),
            WordTiming(word: " so", tokens: [370], start: 0.68, end: 1.1, probability: 1.0),
            WordTiming(word: " my", tokens: [452], start: 1.1, end: 1.36, probability: 0.51),
            WordTiming(word: " fellow", tokens: [7177], start: 1.36, end: 1.74, probability: 0.52),
            WordTiming(word: " Americans", tokens: [6280], start: 1.74, end: 2.26, probability: 0.82),
            WordTiming(word: " ask", tokens: [1029], start: 2.26, end: 3.82, probability: 0.4),
            WordTiming(word: " not", tokens: [406], start: 3.82, end: 4.56, probability: 1.0),
            WordTiming(word: " what", tokens: [437], start: 4.56, end: 5.68, probability: 0.91),
            WordTiming(word: " your", tokens: [428], start: 5.68, end: 5.92, probability: 0.22),
            WordTiming(word: " country", tokens: [1941], start: 5.92, end: 6.38, probability: 0.64),
            WordTiming(word: " can", tokens: [393], start: 6.38, end: 6.76, probability: 0.52),
            WordTiming(word: " do", tokens: [360], start: 6.76, end: 6.98, probability: 0.85),
            WordTiming(word: " for", tokens: [337], start: 6.98, end: 7.22, probability: 0.97),
            WordTiming(word: " you,", tokens: [291, 11], start: 7.22, end: 8.36, probability: 0.97),
            WordTiming(word: " ask", tokens: [1029], start: 8.36, end: 8.66, probability: 0.93),
            WordTiming(word: " what", tokens: [437], start: 8.66, end: 8.86, probability: 0.98),
            WordTiming(word: " you", tokens: [291], start: 8.86, end: 9.22, probability: 0.06),
            WordTiming(word: " can", tokens: [393], start: 9.22, end: 9.44, probability: 0.58),
            WordTiming(word: " do", tokens: [360], start: 9.44, end: 9.64, probability: 0.87),
            WordTiming(word: " for", tokens: [337], start: 9.64, end: 9.86, probability: 0.95),
            WordTiming(word: " your", tokens: [428], start: 9.86, end: 10.06, probability: 0.96),
            WordTiming(word: " country.", tokens: [1941, 13], start: 10.06, end: 10.5, probability: 0.91)
        ]

        XCTAssertEqual(wordTimings.count, expectedWordTimings.count, "Number of word timings should match")

        for (index, wordTiming) in wordTimings.enumerated() {
            let expectedWordTiming = expectedWordTimings[index]

            XCTAssertEqual(wordTiming.word.normalized, expectedWordTiming.word.normalized, "Word should match at index \(index) (expected: \(expectedWordTiming.word), actual: \(wordTiming.word))")

            XCTAssertEqual(wordTiming.start, expectedWordTiming.start, accuracy: 0.5, "Start time difference for word '\(wordTiming.word)' should be within +/- 0.1 seconds (expected: \(expectedWordTiming.start), actual: \(wordTiming.start))")

            XCTAssertEqual(wordTiming.end, expectedWordTiming.end, accuracy: 0.5, "End time difference for word '\(wordTiming.word)' should be within +/- 0.1 seconds (expected: \(expectedWordTiming.end), actual: \(wordTiming.end))")
        }
    }

    // MARK: - Streaming Timestamp Tests

    func testStreamingTimestamps() async throws {
        let options = DecodingOptions(usePrefillPrompt: true, wordTimestamps: true)
//        let audioFile = "ted_60.m4a"
        let audioFile = "jfk.wav"
        let modelPath = tinyModelPath()
//        let modelPath = largev3ModelPath()
//        let computeOptions = ModelComputeOptions(
//            melCompute: .cpuOnly,
//            audioEncoderCompute: .cpuOnly,
//            textDecoderCompute: .cpuOnly,
//            prefillCompute: .cpuOnly
//        )

        let whisperKit = try await WhisperKit(modelFolder: modelPath,/* computeOptions: computeOptions,*/ verbose: true, logLevel: .debug)

        // Get the current decoderState from the textDecoder
        let textDecoder = whisperKit.textDecoder as! TextDecoder
//        var decodingState = textDecoder.decodingState


        // Create a new DecodingInputs instance if decoderState is nil
//        if decodingState == nil {
//            let decodingInputs = textDecoder.prepareDecoderInputs(withPrompt: [whisperKit.tokenizer!.startOfTranscriptToken])!
//            decodingState = DecodingState(
//                decodingInputs: decodingInputs,
//                currentTokens: decodingInputs.initialPrompt,
//                logProbs: Array(repeating: 0, count: decodingInputs.initialPrompt.count)
//            )
//        }

        let startTime = Date()
        let audioComponents = audioFile.components(separatedBy: ".")
        guard let audioFileURL = Bundle.module.path(forResource: audioComponents.first, ofType: audioComponents.last) else {
            XCTFail("Audio file not found")
            return
        }
        guard let audioBuffer = AudioProcessor.loadAudio(fromPath: audioFileURL) else {
            XCTFail("Failed to load audio buffer")
            return
        }
        let audioArray = AudioProcessor.convertBufferToArray(buffer: audioBuffer)


        var results: [TranscriptionResult?] = []
        var prevResult: TranscriptionResult?
        var lastAgreedSeconds: Float = 0.0
        let agreementCountNeeded = 3
        var hypothesisWords: [WordTiming] = []
        var prevWords: [WordTiming] = []
        var lastAgreedWords: [WordTiming] = []
        var confirmedWords: [WordTiming] = []

        for seekSample in stride(from: 0, to: audioArray.count, by: 16000) {
            let endSample = min(seekSample + 16000, audioArray.count)
            Logging.info("[testStreamingTimestamps] \(lastAgreedSeconds)-\(Double(endSample)/16000.0) seconds")

            let simulatedStreamingAudio = Array(audioArray[..<endSample])
            var streamOptions = options
            streamOptions.clipTimestamps = [lastAgreedSeconds]
            let lastAgreedTokens = lastAgreedWords.flatMap { $0.tokens }
            streamOptions.initialPromptTokens = lastAgreedTokens
            do {
                let result: TranscriptionResult? = try await whisperKit.transcribe(audioArray: simulatedStreamingAudio, decodeOptions: streamOptions)
                if let result = result {
                    hypothesisWords = result.allWords.filter { $0.start > lastAgreedSeconds - 0.1 }

                    if let prevResult = prevResult {
                        prevWords = prevResult.allWords.filter { $0.start > lastAgreedSeconds - 0.1 }
                        let commonPrefix = findLongestCommonPrefix(prevWords, hypothesisWords)
                        Logging.info("[testStreamingTimestamps] Prev \"\((prevWords.map { $0.word }).joined())\"")
                        Logging.info("[testStreamingTimestamps] Next \"\((hypothesisWords.map { $0.word }).joined())\"")
                        Logging.info("[testStreamingTimestamps] Found common prefix \"\((commonPrefix.map { $0.word }).joined())\"")

                        if commonPrefix.count >= agreementCountNeeded {
                            lastAgreedWords = commonPrefix.suffix(agreementCountNeeded)
                            lastAgreedSeconds = lastAgreedWords.first!.start
                            Logging.info("[testStreamingTimestamps] Found new last agreed word \(lastAgreedWords.first!.word) at \(lastAgreedSeconds) seconds")

                            confirmedWords.append(contentsOf: commonPrefix.prefix(commonPrefix.count - agreementCountNeeded))
                            let currentWords = confirmedWords.map { $0.word }.joined()
                            Logging.info("[testStreamingTimestamps] Current:  \(lastAgreedSeconds) -> \(Double(endSample)/16000.0) \(currentWords)")
                        } else {
                            Logging.info("[testStreamingTimestamps] Using same last agreed time \(lastAgreedSeconds)")
                        }


                    }
                    prevResult = result
                }
                results.append(result)
            } catch {
                XCTFail(error.localizedDescription)
            }
        }

        // Accept the final hypothesis because it is the last of the available audio
        let final = lastAgreedWords + findLongestDifferentSuffix(prevWords, hypothesisWords)
        confirmedWords.append(contentsOf: final)

        let finalWords = confirmedWords.map { $0.word }.joined()
        Logging.info("[testStreamingTimestamps] Current: \(finalWords)")
        Logging.info("[testStreamingTimestamps] Time taken: \(Date().timeIntervalSince(startTime)) seconds")

        // Perform assertions or further processing with the results array
        Logging.info("[testStreamingTimestamps] Done")

        XCTAssertEqual(finalWords, " And so my fellow Americans. Ask not what your country can do for you ask what you can do for your country.")
    }

    func findLongestCommonPrefix(_ words1: [WordTiming], _ words2: [WordTiming]) -> [WordTiming] {
        let commonPrefix = zip(words1, words2).prefix(while: { $0.word == $1.word })
        return commonPrefix.map { $0.0 }
    }

    func findLongestDifferentSuffix(_ words1: [WordTiming], _ words2: [WordTiming]) -> [WordTiming] {
        let commonPrefix = findLongestCommonPrefix(words1, words2)
        let remainingWords = words2[commonPrefix.count...]
        return Array(remainingWords)
    }

}

// MARK: - Helpers

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
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

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
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

    func largev3TurboModelPath() -> String {
        let modelDir = "whisperkit-coreml/openai_whisper-large-v3_turbo"
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
                    // Check if the directory contains actual data files, or if it contains pointer files.
                    // As a proxy, use the MelSpectrogramc.mlmodel/coredata.bin file.
                    let proxyFileToCheck = folderURL.appendingPathComponent("MelSpectrogram.mlmodelc/coremldata.bin")
                    if isGitLFSPointerFile(url: proxyFileToCheck) {
                        continue
                    }
                    
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
    
    // Function to check if the beginning of the file matches a Git LFS pointer pattern
    func isGitLFSPointerFile(url: URL) -> Bool {
        do {
            let fileHandle = try FileHandle(forReadingFrom: url)
            // Read the first few bytes of the file to get enough for the Git LFS pointer signature
            let data = fileHandle.readData(ofLength: 512) // Read first 512 bytes
            fileHandle.closeFile()

            if let string = String(data: data, encoding: .utf8),
               string.starts(with: "version https://git-lfs.github.com/") {
                return true
            }
        } catch {
            fatalError("Failed to read file: \(error)")
        }
        
        return false
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
