//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import XCTest
import MLX
import WhisperKitTestsUtils
import CoreML
import NaturalLanguage
@testable import WhisperKit
@testable import WhisperKitMLX

final class MLXUnitTests: XCTestCase {
    private var tinyModelPath: String!
    private let accuracy: Float = 0.00001

    override func setUp() async throws {
        try await super.setUp()
        self.tinyModelPath = try tinyMLXModelPath()
    }

    // MARK: - Feature Extractor Tests

    func testLogmelOutput() async throws {
        let audioSamples = [Float](repeating: 0.0, count: 16000)
        let paddedSamples = try XCTUnwrap(
            AudioProcessor.padOrTrimAudio(fromArray: audioSamples, startAt: 0, toLength: 480_000),
            "Failed to pad audio samples"
        )
        let featureExtractor = MLXFeatureExtractor()
        let extractedFeature = try await featureExtractor.logMelSpectrogram(fromAudio: paddedSamples)
        let melSpectrogram = try XCTUnwrap(extractedFeature, "Failed to produce Mel spectrogram from audio samples")

        let expectedShape: [NSNumber] = [1, 80, 1, 3000]
        XCTAssertNotNil(melSpectrogram, "Failed to produce Mel spectrogram from audio samples")
        XCTAssertEqual(melSpectrogram.shape, expectedShape, "Mel spectrogram shape is not as expected")
    }

    // MARK: - Encoder Tests

    func testEncoderOutput() async throws {
        let audioEncoder = MLXAudioEncoder()
        let modelPath = URL(filePath: tinyModelPath)
        try await audioEncoder.loadModel(
            at: modelPath.appending(path: "encoder.safetensors"),
            configPath: modelPath.appending(path: "config.json")
        )

        let encoderInput = try MLMultiArray(shape: [1, 80, 1, 3000], dataType: .float16)
        let expectedShape: [NSNumber] = [1, 384, 1, 1500]

        let encoderOutput = try await audioEncoder.encodeFeatures(encoderInput)
        XCTAssertNotNil(encoderOutput, "Failed to encode features")
        XCTAssertEqual(encoderOutput?.shape, expectedShape, "Encoder output shape is not as expected")
    }

    // MARK: - Decoder Tests

    func testDecoderOutput() async throws {
        let textDecoder = MLXTextDecoder()
        let decodingOptions = DecodingOptions()
        let modelPath = URL(filePath: tinyModelPath)
        await XCTAssertNoThrowAsync(
            try await textDecoder.loadModel(
                at: modelPath.appending(path: "decoder.safetensors"),
                configPath: modelPath.appending(path: "config.json")
            ),
            "Failed to load the model"
        )
        textDecoder.tokenizer = try await XCTUnwrapAsync(
            await loadTokenizer(for: .tiny),
            "Failed to load the tokenizer"
        )

        let tokenSampler = MLXGreedyTokenSampler(
            temperature: 0,
            eotToken: textDecoder.tokenizer!.specialTokens.endToken,
            decodingOptions: decodingOptions
        )

        let encoderInput = try MLMultiArray(shape: [1, 384, 1, 1500], dataType: .float16)
        let inputs = try textDecoder.prepareDecoderInputs(withPrompt: [textDecoder.tokenizer!.specialTokens.startOfTranscriptToken])

        await XCTAssertNoThrowAsync(
            try await textDecoder.decodeText(
                from: encoderInput,
                using: inputs,
                sampler: tokenSampler,
                options: decodingOptions
            )
        )
    }

    func testDecoderLogProbThresholdDecodingFallback() async throws {
        let decodingOptions = DecodingOptions(
            withoutTimestamps: true,
            compressionRatioThreshold: nil,
            logProbThreshold: 1000.0,
            firstTokenLogProbThreshold: nil,
            noSpeechThreshold: nil
        )
        let textDecoder = MLXTextDecoder()
        let modelPath = URL(filePath: tinyModelPath)
        try await textDecoder.loadModel(
            at: modelPath.appending(path: "decoder.safetensors"),
            configPath: modelPath.appending(path: "config.json")
        )
        textDecoder.tokenizer = try await loadTokenizer(for: .tiny)

        let tokenSampler = MLXGreedyTokenSampler(temperature: 0, eotToken: textDecoder.tokenizer!.specialTokens.endToken, decodingOptions: decodingOptions)

        let encoderInput = initMLMultiArray(shape: [1, 384, 1, 1500], dataType: .float16, initialValue: FloatType(0))
        let inputs = try textDecoder.prepareDecoderInputs(withPrompt: [textDecoder.tokenizer!.specialTokens.startOfTranscriptToken])
        let decoderOutput = try await textDecoder.decodeText(from: encoderInput, using: inputs, sampler: tokenSampler, options: decodingOptions)

        let fallback = try XCTUnwrap(decoderOutput.fallback, "Fallback should not be `nil`")
        XCTAssertEqual(fallback.fallbackReason, "logProbThreshold")
        XCTAssertTrue(fallback.needsFallback)
    }

    func testDecoderFirstTokenLogProbThresholdDecodingFallback() async throws {
        let decodingOptions = DecodingOptions(
            withoutTimestamps: true,
            compressionRatioThreshold: nil,
            logProbThreshold: nil,
            firstTokenLogProbThreshold: 1000.0,
            noSpeechThreshold: nil
        )
        let textDecoder = MLXTextDecoder()
        let modelPath = URL(filePath: tinyModelPath)
        try await textDecoder.loadModel(
            at: modelPath.appending(path: "decoder.safetensors"),
            configPath: modelPath.appending(path: "config.json")
        )
        textDecoder.tokenizer = try await loadTokenizer(for: .tiny)

        let tokenSampler = MLXGreedyTokenSampler(temperature: 0, eotToken: textDecoder.tokenizer!.specialTokens.endToken, decodingOptions: decodingOptions)

        let encoderInput = initMLMultiArray(shape: [1, 384, 1, 1500], dataType: .float16, initialValue: FloatType(0))
        let inputs = try textDecoder.prepareDecoderInputs(withPrompt: [textDecoder.tokenizer!.specialTokens.startOfTranscriptToken])
        let decoderOutput = try await textDecoder.decodeText(from: encoderInput, using: inputs, sampler: tokenSampler, options: decodingOptions)

        let fallback = try XCTUnwrap(decoderOutput.fallback, "Fallback should not be `nil`")
        XCTAssertEqual(fallback.fallbackReason, "firstTokenLogProbThreshold")
        XCTAssertTrue(fallback.needsFallback)
    }

    // MARK: - Options Tests

    /// Multilingual Tests
    /// NOTE: These are purely for consistency checks and do not reflect the ground truth translations
    func testTranslateSpanish() async throws {
        let targetLanguage = "es"
        let options = DecodingOptions(task: .translate, language: targetLanguage, temperatureFallbackCount: 0)

        let result = try await XCTUnwrapAsync(
            try await transcribe(
                mlxModelPath: tinyModelPath,
                options: options,
                audioFile: "es_test_clip.wav",
                featureExtractor: MLXFeatureExtractor(),
                audioEncoder: MLXAudioEncoder(),
                textDecoder: MLXTextDecoder()
            ),
            "Failed to transcribe"
        )

        XCTAssertEqual(result.text.split(separator: " ").prefix(2).joined(separator: " "), "This is")
    }

    func testTranscribeSpanish() async throws {
        let sourceLanguage = "es"
        let options = DecodingOptions(task: .transcribe, language: sourceLanguage, temperatureFallbackCount: 0)

        let result = try await XCTUnwrapAsync(
            try await transcribe(
                mlxModelPath: tinyModelPath,
                options: options,
                audioFile: "es_test_clip.wav",
                featureExtractor: MLXFeatureExtractor(),
                audioEncoder: MLXAudioEncoder(),
                textDecoder: MLXTextDecoder()
            ),
            "Failed to transcribe"
        )

        XCTAssertEqual(result.text.split(separator: " ").prefix(4).joined(separator: " "), "Esta es una grabación")
    }

    func testDetectSpanish() async throws {
        let targetLanguage = "es"
        let whisperKit = try await WhisperKit(
            mlxModelFolder: tinyModelPath,
            featureExtractor: MLXFeatureExtractor(),
            audioEncoder: MLXAudioEncoder(),
            textDecoder: MLXTextDecoder(),
            verbose: true,
            logLevel: .debug
        )

        let audioFilePath = try XCTUnwrap(
            TestResource.path(forResource: "es_test_clip", ofType: "wav"),
            "Audio file not found"
        )

        // To detect language only, set `sampleLength` to 1 and no prefill prompt
        let optionsDetectOnly = DecodingOptions(task: .transcribe, temperatureFallbackCount: 0, sampleLength: 1, detectLanguage: true)
        let resultNoPrefill: [TranscriptionResult] = try await whisperKit.transcribe(audioPath: audioFilePath, decodeOptions: optionsDetectOnly)

        XCTAssertEqual(resultNoPrefill.first?.language, targetLanguage)
    }

    func testTranslateJapaneseOptions() async throws {
        let targetLanguage = "ja"
        let options = DecodingOptions(task: .translate, language: targetLanguage, temperatureFallbackCount: 0)

        let result = try await XCTUnwrapAsync(
            try await transcribe(
                mlxModelPath: tinyModelPath,
                options: options,
                audioFile: "ja_test_clip.wav",
                featureExtractor: MLXFeatureExtractor(),
                audioEncoder: MLXAudioEncoder(),
                textDecoder: MLXTextDecoder()
            ),
            "Failed to transcribe"
        )

        XCTAssertEqual(result.text.split(separator: " ").first, "Tokyo")
    }

    func testTranscribeJapanese() async throws {
        let sourceLanguage = "ja"
        let options = DecodingOptions(task: .transcribe, language: sourceLanguage, temperatureFallbackCount: 0)

        let result = try await XCTUnwrapAsync(
            try await transcribe(
                mlxModelPath: tinyModelPath,
                options: options,
                audioFile: "ja_test_clip.wav",
                featureExtractor: MLXFeatureExtractor(),
                audioEncoder: MLXAudioEncoder(),
                textDecoder: MLXTextDecoder()
            ),
            "Failed to transcribe"
        )

        XCTAssertEqual(result.text.prefix(3), "東京は")
    }

    func testDetectJapanese() async throws {
        let targetLanguage = "ja"
        let whisperKit = try await WhisperKit(
            mlxModelFolder: tinyModelPath,
            featureExtractor: MLXFeatureExtractor(),
            audioEncoder: MLXAudioEncoder(),
            textDecoder: MLXTextDecoder(),
            verbose: true,
            logLevel: .debug
        )

        let audioFilePath = try XCTUnwrap(
            TestResource.path(forResource: "ja_test_clip", ofType: "wav"),
            "Audio file not found"
        )

        // To detect language only, set `sampleLength` to 1 and no prefill prompt
        let optionsDetectOnly = DecodingOptions(task: .transcribe, temperatureFallbackCount: 0, sampleLength: 1, detectLanguage: true)
        let result: [TranscriptionResult] = try await whisperKit.transcribe(audioPath: audioFilePath, decodeOptions: optionsDetectOnly)

        XCTAssertEqual(result.first?.language, targetLanguage)
    }

    func testDetectJapaneseOptions() async throws {
        let optionsPairs: [(options: DecodingOptions, language: String)] = [
            (DecodingOptions(task: .transcribe, temperatureFallbackCount: 0, usePrefillPrompt: true, detectLanguage: true), "ja"), // recommended usage for transcribing unknown language
            (DecodingOptions(task: .transcribe, temperatureFallbackCount: 0, usePrefillPrompt: true, detectLanguage: false), "en"), // en is the default prompt language
            (DecodingOptions(task: .transcribe, temperatureFallbackCount: 0, usePrefillPrompt: true, detectLanguage: nil), "en"), // en is the default prompt language
            (DecodingOptions(task: .transcribe, temperatureFallbackCount: 0, usePrefillPrompt: false, detectLanguage: true), "ja"), // Unecessary combination, but can be useful if used with low `sampleLength` values to purely detect language and not decode (see above)
            (DecodingOptions(task: .transcribe, temperatureFallbackCount: 0, usePrefillPrompt: false, detectLanguage: false), "ja"), // no prefill, model will detect language naturally
            (DecodingOptions(task: .transcribe, temperatureFallbackCount: 0, usePrefillPrompt: false, detectLanguage: nil), "ja"), // no prefill, model will detect language naturally
        ]

        for (i, option) in optionsPairs.enumerated() {
            let result = try await XCTUnwrapAsync(
                try await transcribe(
                    mlxModelPath: tinyModelPath,
                    options: option.options,
                    audioFile: "ja_test_clip.wav",
                    featureExtractor: MLXFeatureExtractor(),
                    audioEncoder: MLXAudioEncoder(),
                    textDecoder: MLXTextDecoder()
                ),
                "Failed to transcribe"
            )

            let recognizer = NLLanguageRecognizer()
            recognizer.processString(result.text)
            let languageCode = recognizer.dominantLanguage!.rawValue

            XCTAssertEqual(
                languageCode,
                option.language,
                "Text language \"\(languageCode)\" at index \(i) did not match expected language \"\(option.language)\""
            )
            XCTAssertEqual(
                result.first?.language,
                option.language,
                "Result language \"\(String(describing: result.first?.language))\" at index \(i) did not match expected language \"\(option.language)\""
            )
        }
    }

    // MARK: - Utils Tests

    func testStrides() {
        let count = 24
        let arr1 = MLXArray(0..<count, [count]).asType(Int32.self)
        XCTAssertEqual(arr1.asData(access: .noCopy).strides, [1])

        let arr2 = MLXArray(0..<count, [2, count / 2]).asType(Int32.self)
        XCTAssertEqual(arr2.asData(access: .noCopy).strides, [12, 1])

        let arr3 = MLXArray(0..<count, [2, count / 2]).asType(Int32.self).asMLXInput()
        XCTAssertEqual(arr3.asData(access: .noCopy).strides, [12, 1, 12])
    }

    func testArrayConversion() throws {
        let count = 16
        let arr1 = MLXArray(0..<count, [2, count / 2]).asType(Int32.self)
        let arr2 = try arr1.asMLMultiArray().asMLXArray(Int32.self)
        XCTAssertEqual(arr2.asData(access: .noCopy).strides, [8, 1])
        XCTAssertTrue(MLX.allClose(arr1, arr2).item(), "Array conversion failed")

        let arr3 = arr1.asMLXOutput().asMLXInput()
        XCTAssertEqual(arr3.asData(access: .noCopy).strides, [16, 8, 1])
        XCTAssertTrue(MLX.allClose(arr1, arr3).item(), "Input output conversion failed")

        let arr4 = try arr1.asMLXOutput().asMLXInput().asMLMultiArray().asMLXArray(Int32.self)
        XCTAssertEqual(arr4.asData(access: .noCopy).strides, [16, 8, 1])
        XCTAssertTrue(MLX.allClose(arr1, arr4).item(), "Complex conversion failed")

        let arr5 = try arr1.asMLXOutput().asMLMultiArray().asMLXArray(Int32.self).asMLXInput()
        XCTAssertEqual(arr5.asData(access: .noCopy).strides, [16, 8, 1])
        XCTAssertTrue(MLX.allClose(arr1, arr5).item(), "Complex conversion failed")
    }

    func testAsMLMultiArray() throws {
        let count = 24
        let input = (0..<count).map { Float($0) }
        let arr1 = MLXArray(input, [count])
        let multiArray1 = try arr1.asMLMultiArray()

        XCTAssertEqual(arr1.shape, multiArray1.shape.map { $0.intValue })
        for col in 0..<count {
            let v1 = multiArray1[[col] as [NSNumber]].floatValue
            let v2 = arr1[col]
            XCTAssertEqual(v1, v2.item(Float.self), accuracy: accuracy)
        }

        let arr2 = MLXArray(input, [4, 6])
        let multiArray2 = try arr2.asMLMultiArray()

        XCTAssertEqual(arr2.shape, multiArray2.shape.map { $0.intValue })
        for row in 0..<4 {
            for col in 0..<6 {
                let v1 = multiArray2[[row, col] as [NSNumber]].floatValue
                let v2 = arr2[row, col]
                XCTAssertEqual(v1, v2.item(Float.self), accuracy: accuracy)
            }
        }

        let arr3 = MLXArray(input, [4, 3, 2])
        let multiArray3 = try arr3.asMLMultiArray()

        XCTAssertEqual(arr3.shape, multiArray3.shape.map { $0.intValue })
        for dim1 in 0..<4 {
            for dim2 in 0..<3 {
                for dim3 in 0..<2 {
                    let v1 = multiArray3[[dim1, dim2, dim3] as [NSNumber]].floatValue
                    let v2 = arr3[dim1, dim2, dim3]
                    XCTAssertEqual(v1, v2.item(Float.self), accuracy: accuracy)
                }
            }
        }

        let arr4 = MLXArray(input, [2, 3, 2, 2])
        let multiArray4 = try arr4.asMLMultiArray()

        XCTAssertEqual(arr4.shape, multiArray4.shape.map { $0.intValue })
        for dim1 in 0..<2 {
            for dim2 in 0..<3 {
                for dim3 in 0..<2 {
                    for dim4 in 0..<2 {
                        let v1 = multiArray4[[dim1, dim2, dim3, dim4] as [NSNumber]].floatValue
                        let v2 = arr4[dim1, dim2, dim3, dim4]
                        XCTAssertEqual(v1, v2.item(Float.self), accuracy: accuracy)
                    }
                }
            }
        }
    }

    func testSinusoids() {
        let result1 = sinusoids(length: 0, channels: 0).asType(Float.self)
        XCTAssertEqual(result1.shape, [0, 0])
        XCTAssertEqual(result1.count, 0)

        let result2 = sinusoids(length: 2, channels: 4).asType(Float.self)
        XCTAssertEqual(result2.shape, [2, 4])
        XCTAssertEqual(result2.count, 2)
        XCTAssertEqual(result2[0].asArray(Float.self), [0.0, 0.0, 1.0, 1.0], accuracy: accuracy)
        XCTAssertEqual(result2[1].asArray(Float.self), [0.841471, 0.0001, 0.540302, 1.0], accuracy: accuracy)

        let result3 = sinusoids(length: 4, channels: 8).asType(Float.self)
        XCTAssertEqual(result3.shape, [4, 8])
        XCTAssertEqual(result3.count, 4)

        XCTAssertEqual(result3[0].asArray(Float.self), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], accuracy: accuracy)
        XCTAssertEqual(result3[1].asArray(Float.self), [0.841471, 0.0463992, 0.00215443, 0.0001, 0.540302, 0.998923, 0.999998, 1], accuracy: accuracy)
        XCTAssertEqual(result3[2].asArray(Float.self), [0.909297, 0.0926985, 0.00430886, 0.0002, -0.416147, 0.995694, 0.999991, 1.0], accuracy: accuracy)
        XCTAssertEqual(result3[3].asArray(Float.self), [0.14112, 0.138798, 0.00646326, 0.0003, -0.989992, 0.990321, 0.999979, 1.0], accuracy: accuracy)
    }

    func testAdditiveCausalMask() {
        let result1 = additiveCausalMask(0)
        XCTAssertEqual(result1.shape, [0 ,0], "Array shape should be [0, 0]")
        XCTAssertEqual(result1.dtype, .float32, "Array type should be .float32")

        let result2 = additiveCausalMask(3)
        XCTAssertEqual(result2.shape, [3 ,3], "Array shape should be [3, 3]")
        XCTAssertEqual(result2.dtype, .float32, "Array type should be .float32")
        XCTAssertEqual(result2[0].asArray(Float.self), [0.0, -1e9, -1e9], accuracy: accuracy)
        XCTAssertEqual(result2[1].asArray(Float.self), [0.0, 0.0, -1e9], accuracy: accuracy)
        XCTAssertEqual(result2[2].asArray(Float.self), [0.0, 0.0, 0.0], accuracy: accuracy)

        let result3 = additiveCausalMask(4)
        XCTAssertEqual(result3.shape, [4 ,4], "Array shape should be [4, 4]")
        XCTAssertEqual(result3.dtype, .float32, "Array type should be .float32")
        XCTAssertEqual(result3[0].asArray(Float.self), [0.0, -1e9, -1e9, -1e9], accuracy: accuracy)
        XCTAssertEqual(result3[1].asArray(Float.self), [0.0, 0.0, -1e9, -1e9], accuracy: accuracy)
        XCTAssertEqual(result3[2].asArray(Float.self), [0.0, 0.0, 0.0, -1e9], accuracy: accuracy)
        XCTAssertEqual(result3[3].asArray(Float.self), [0.0, 0.0, 0.0, 0.0], accuracy: accuracy)
    }
}
