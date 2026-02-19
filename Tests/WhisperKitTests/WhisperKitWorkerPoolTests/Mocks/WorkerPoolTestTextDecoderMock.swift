//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
@testable import WhisperKit

final class WorkerPoolTestTextDecoderMock: TextDecoding {
    var tokenizer: WhisperTokenizer?
    var prefillData: WhisperMLModel?
    var isModelMultilingual: Bool = false
    var supportsWordTimestamps: Bool = false
    var logitsSize: Int? = 1024
    var logitsFilters: [any LogitsFiltering]?
    var kvCacheEmbedDim: Int? = 1
    var kvCacheMaxSequenceLength: Int? = 1
    var windowSize: Int? = 1
    var embedSize: Int? = 1

    private let runTracker: WorkerPoolRunTracker

    init(runTracker: WorkerPoolRunTracker) {
        self.runTracker = runTracker
    }

    func predictLogits(_ inputs: any TextDecoderInputType) async throws -> TextDecoderOutputType? {
        nil
    }

    func prepareDecoderInputs(withPrompt initialPrompt: [Int]) throws -> any DecodingInputsType {
        try WorkerPoolTestDecodingInputs(initialPrompt: initialPrompt)
    }

    func prefillDecoderInputs(
        _ decoderInputs: any DecodingInputsType,
        withOptions options: DecodingOptions?
    ) async throws -> any DecodingInputsType {
        decoderInputs
    }

    func prefillKVCache(
        withTask task: MLMultiArray,
        andLanguage language: MLMultiArray
    ) async throws -> DecodingCache? {
        nil
    }

    func decodeText(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options decoderOptions: DecodingOptions,
        callback: TranscriptionCallback?
    ) async throws -> DecodingResult {
        await runTracker.beginRun()

        do {
            try await Task.sleep(nanoseconds: 20_000_000)
            let token = decoderOptions.promptTokens?.first ?? 0
            _ = callback?(
                TranscriptionProgress(
                    timings: TranscriptionTimings(),
                    text: "token_\(token)",
                    tokens: [token]
                )
            )
            await runTracker.endRun()
            return DecodingResult(
                language: "en",
                languageProbs: ["en": 1.0],
                tokens: [token],
                tokenLogProbs: [[token: 0.0]],
                text: "token_\(token)",
                avgLogProb: 0.0,
                noSpeechProb: 0.0,
                temperature: 0.0,
                compressionRatio: 1.0,
                cache: nil,
                timings: nil,
                fallback: nil
            )
        } catch {
            await runTracker.endRun()
            throw error
        }
    }

    func detectLanguage(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> DecodingResult {
        DecodingResult(
            language: "en",
            languageProbs: ["en": 1.0],
            tokens: [],
            tokenLogProbs: [],
            text: "",
            avgLogProb: 0.0,
            noSpeechProb: 0.0,
            temperature: 0.0,
            compressionRatio: 1.0,
            cache: nil,
            timings: nil,
            fallback: nil
        )
    }
}
