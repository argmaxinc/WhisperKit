//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import MLX
import MLXNN
import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public final class MLXTextDecoder: TextDecoding {
    public var model: TextDecoderModule?
    public var tokenizer: (any WhisperTokenizer)?
    public var prefillData: (any WhisperMLModel)?
    public var isModelMultilingual: Bool = false
    public let supportsWordTimestamps: Bool = false
    public var logitsSize: Int? {
        model?.nVocab
    }

    public var kvCacheEmbedDim: Int? {
        guard let config else { return nil }
        return config.nTextState * config.nTextLayer
    }

    public var kvCacheMaxSequenceLength: Int? {
        guard let config else { return nil }
        return config.nTextCtx / 2
    }

    public var windowSize: Int? {
        guard let config else { return nil }
        return config.nAudioCtx
    }

    public var embedSize: Int? {
        guard let config else { return nil }
        return config.nTextState
    }

    private var config: MLXModelConfig?
    private var languageLogitsFilter: LanguageLogitsFilter?

    public init() {}

    private static func toKvCache(keyCache: MLMultiArray?, valueCache: MLMultiArray?) -> [KV]? {
        guard let keyCache, let valueCache else {
            return nil
        }
        let keyCacheMlx = keyCache.asMLXArray(FloatType.self)
        let valueCacheMlx = valueCache.asMLXArray(FloatType.self)

        return toKvCache(keyCache: keyCacheMlx, valueCache: valueCacheMlx)
    }

    private static func toKvCache(keyCache: MLXArray?, valueCache: MLXArray?) -> [KV]? {
        guard let keyCache, let valueCache else {
            return nil
        }
        assert(keyCache.shape == valueCache.shape)

        var result = [KV]()
        for index in 0..<keyCache.shape[0] {
            let k = keyCache[index]
            let v = valueCache[index]
            result.append(KV(k: k, v: v))
        }
        return result
    }

    public func prepareDecoderInputs(withPrompt initialPrompt: [Int]) throws -> DecodingInputs {
        let tokenShape = [NSNumber(value: 1), NSNumber(value: initialPrompt.count)]

        // Initialize MLMultiArray for tokens
        let tokenMultiArray = try MLMultiArray(shape: tokenShape, dataType: .int32)

        // Assign token values to the MLMultiArray
        for (index, token) in initialPrompt.enumerated() {
            tokenMultiArray[index] = NSNumber(value: token)
        }

        guard let kvCacheEmbedDim = self.kvCacheEmbedDim else {
            throw WhisperError.prepareDecoderInputsFailed("Unable to determine kvCacheEmbedDim")
        }

        guard let kvCacheMaxSequenceLength = self.kvCacheMaxSequenceLength else {
            throw WhisperError.prepareDecoderInputsFailed("Unable to determine kvCacheMaxSequenceLength")
        }

        guard let encoderOutputDim = self.windowSize else {
            throw WhisperError.prepareDecoderInputsFailed("Unable to determine encoderOutputDim")
        }

        // Initialize each MLMultiArray
        let kvCacheEmbedDimValue = NSNumber(value: kvCacheEmbedDim)
        let kvCacheMaxSequenceLengthValue = NSNumber(value: kvCacheMaxSequenceLength)
        let encoderOutputDimValue = NSNumber(value: encoderOutputDim)

        let inputIds = initMLMultiArray(shape: [1], dataType: .int32, initialValue: Int32(0))
        let cacheLength = initMLMultiArray(shape: [1], dataType: .int32, initialValue: Int32(0))
        let alignmentWeights = initMLMultiArray(shape: [kvCacheMaxSequenceLengthValue, encoderOutputDimValue], dataType: .float16, initialValue: FloatType(0))
        let kvCacheUpdateMask = initMLMultiArray(shape: [1, kvCacheMaxSequenceLengthValue], dataType: .int32, initialValue: Int32(0))
        let decoderKeyPaddingMask = initMLMultiArray(shape: [1, kvCacheMaxSequenceLengthValue], dataType: .float16, initialValue: FloatType(-10000))
        let prefillKeyCache = try! MLMultiArray(shape: [1, kvCacheEmbedDimValue, 1, kvCacheMaxSequenceLengthValue], dataType: .float16)
        let prefillValueCache = try! MLMultiArray(shape: [1, kvCacheEmbedDimValue, 1, kvCacheMaxSequenceLengthValue], dataType: .float16)
        let decoderInputs = DecodingInputs(
            initialPrompt: initialPrompt,
            inputIds: inputIds,
            cacheLength: cacheLength,
            keyCache: nil,
            valueCache: nil,
            alignmentWeights: alignmentWeights,
            kvCacheUpdateMask: kvCacheUpdateMask,
            decoderKeyPaddingMask: decoderKeyPaddingMask,
            prefillKeyCache: prefillKeyCache,
            prefillValueCache: prefillValueCache
        )
        return decoderInputs
    }

    public func predictLogits(
        inputIds: MLMultiArray,
        cacheLength: MLMultiArray,
        keyCache: MLMultiArray?,
        valueCache: MLMultiArray?,
        kvCacheUpdateMask: MLMultiArray,
        encoderOutputEmbeds: MLMultiArray,
        decoderKeyPaddingMask: MLMultiArray
    ) async throws -> (logits: MLMultiArray?, cache: DecodingCache?)? {
        let result = try await predictLogits(
            inputIds: inputIds,
            cacheLength: cacheLength,
            keyCache: keyCache?.asMLXArray(FloatType.self),
            valueCache: valueCache?.asMLXArray(FloatType.self),
            kvCacheUpdateMask: kvCacheUpdateMask,
            encoderOutputEmbeds: encoderOutputEmbeds.asMLXArray(FloatType.self).asMLXInput(),
            decoderKeyPaddingMask: decoderKeyPaddingMask
        )

        guard let result = result,
              let keyCacheResult = result.cache?.kvCache.map(\.k),
              let valueCacheResult = result.cache?.kvCache.map(\.v)
        else { return nil }

        let keyCache = try? MLX.stacked(keyCacheResult).asMLMultiArray()
        let valueCache = try? MLX.stacked(valueCacheResult).asMLMultiArray()
        let decodingCache = DecodingCache(
            keyCache: keyCache,
            valueCache: valueCache,
            alignmentWeights: nil
        )

        let logits = try? result.logits?.asMLMultiArray()

        return (logits, decodingCache)
    }

    public func predictLogits(
        inputIds: MLMultiArray,
        cacheLength: MLMultiArray,
        keyCache: MLXArray?,
        valueCache: MLXArray?,
        kvCacheUpdateMask: MLMultiArray,
        encoderOutputEmbeds: MLXArray,
        decoderKeyPaddingMask: MLMultiArray
    ) async throws -> (logits: MLXArray?, cache: MLXDecodingCache?)? {
        guard let model else {
            return nil
        }
        let tokens = inputIds.asMLXArray(Int32.self)
        let result = model(
            tokens,
            xa: encoderOutputEmbeds,
            kvCache: Self.toKvCache(keyCache: keyCache, valueCache: valueCache)
        )

        let decodingCache = MLXDecodingCache(
            kvCache: result.kvCache,
            alignmentWeights: result.alignmentWeights
        )

        return (result.logits, decodingCache)
    }

    public func decodeText(
        from encoderOutput: MLMultiArray,
        using decoderInputs: DecodingInputs,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        callback: TranscriptionCallback = nil
    ) async throws -> DecodingResult {
        guard let tokenizer else {
            // Tokenizer required for decoding
            throw WhisperError.tokenizerUnavailable()
        }

        let tokenSampler = MLXGreedyTokenSampler(temperature: Float(options.temperature), eotToken: tokenizer.specialTokens.endToken, decodingOptions: options)

        // Single loop variables
        var timings = TranscriptionTimings()
        let prefilledIndex = decoderInputs.cacheLength[0].intValue
        let intialPromptIndex = decoderInputs.initialPrompt.count
        var currentTokens: [Int] = decoderInputs.initialPrompt
        var nextToken: Int = decoderInputs.initialPrompt.last!
        var logProbs: [Float] = Array(repeating: 0, count: currentTokens.count)

        // Logits filters
        var logitsFilters: [any LogitsFiltering] = []
        if options.suppressBlank {
            logitsFilters.append(
                SuppressBlankFilter(
                    specialTokens: tokenizer.specialTokens,
                    sampleBegin: prefilledIndex
                )
            )
        }

        if !options.supressTokens.isEmpty {
            logitsFilters.append(SuppressTokensFilter(suppressTokens: options.supressTokens))
        }

        if !options.withoutTimestamps {
            let maxInitialTimestampIndex: Int? =
                if let maxInitialTimestamp = options.maxInitialTimestamp {
                    Int(maxInitialTimestamp / WhisperKit.secondsPerTimeToken)
                } else {
                    nil
                }
            logitsFilters.append(
                TimestampRulesFilter(
                    specialTokens: tokenizer.specialTokens,
                    sampleBegin: intialPromptIndex,
                    maxInitialTimestampIndex: maxInitialTimestampIndex,
                    isModelMultilingual: isModelMultilingual
                )
            )
        }

        // MARK: Main loop

        let loopCount = min(options.sampleLength, Constants.maxTokenContext - 1)
        Logging.debug("Running main loop for a maximum of \(loopCount) iterations, starting at index \(prefilledIndex)")
        var hasAlignment = false
        var isFirstTokenLogProbTooLow = false
        var keyCache = decoderInputs.keyCache?.asMLXArray(FloatType.self)
        var valueCache = decoderInputs.valueCache?.asMLXArray(FloatType.self)
        let encoderOutput = encoderOutput.asMLXArray(FloatType.self).asMLXInput()
        for tokenIndex in prefilledIndex..<loopCount {
            let loopStart = Date()

            let isPrefill = tokenIndex < intialPromptIndex - 1 // Prefill stops at the last token of the initial prompt
            let isFirstToken = tokenIndex == prefilledIndex

            // Check if current index is part of the initial prompt
            if tokenIndex < intialPromptIndex {
                nextToken = currentTokens[tokenIndex]
                Logging.debug("Forcing prompt tokenIndex: \(tokenIndex), token: \(nextToken), text: \(tokenizer.decode(tokens: [nextToken]))")
            }

            // Set the current token as model input
            decoderInputs.inputIds[0] = NSNumber(value: nextToken)
            decoderInputs.cacheLength[0] = NSNumber(value: tokenIndex)

            if tokenIndex <= prefilledIndex + 3 {
                debugCaches(decoderInputs: decoderInputs, tokenIndex: tokenIndex, prefillSize: prefilledIndex)
            }

            // MARK: Decoding Inference

            // Predict next token
            let inferenceTime = Date()

            let predictedLogits = try await self.predictLogits(
                inputIds: decoderInputs.inputIds,
                cacheLength: decoderInputs.cacheLength,
                keyCache: keyCache,
                valueCache: valueCache,
                kvCacheUpdateMask: decoderInputs.kvCacheUpdateMask,
                encoderOutputEmbeds: encoderOutput,
                decoderKeyPaddingMask: decoderInputs.decoderKeyPaddingMask
            )

            guard let decoderOutput = predictedLogits else {
                throw WhisperError.decodingLogitsFailed("Unable to decode logits")
            }

            var logits = try decoderOutput.logits!.asMLMultiArray()

            let decodingInferenceTime = Date().timeIntervalSince(inferenceTime)
            timings.decodingPredictions += decodingInferenceTime

            let kvStartTime = Date()

            keyCache = try MLX.stacked(decoderOutput.cache!.kvCache.map { $0.k })
            valueCache = try MLX.stacked(decoderOutput.cache!.kvCache.map { $0.v })

            let kvTime = Date().timeIntervalSince(kvStartTime)
            timings.decodingKvCaching += kvTime
            timings.totalKVUpdateRuns += 1

            // MARK: Non-inference

            let nonInferenceStartTime = Date()

            // Update predicted token as current
            for filter in logitsFilters {
                logits = filter.filterLogits(logits, withTokens: currentTokens)
            }

            let filteringTime = Date().timeIntervalSince(nonInferenceStartTime)
            timings.decodingFiltering += filteringTime

            // MARK: Sampling

            let samplingStartTime = Date()
            let sampleResult = tokenSampler.update(tokens: currentTokens, logits: logits.asMLXArray(FloatType.self), logProbs: logProbs)

            nextToken = sampleResult.tokens.last!
            let nextTokenLogProb = sampleResult.logProbs.last!

            let samplingTime = Date().timeIntervalSince(samplingStartTime)
            timings.decodingSampling += samplingTime

            isFirstTokenLogProbTooLow =
                if isFirstToken, let firstTokenLogProbThreshold = options.firstTokenLogProbThreshold, nextTokenLogProb < firstTokenLogProbThreshold {
                    true
                } else {
                    false
                }
            let isSegmentCompleted =
                sampleResult.completed ||
                currentTokens.count >= Constants.maxTokenContext - 1 ||
                isFirstTokenLogProbTooLow

            if isSegmentCompleted {
                // Completed segment, stop the loop
                timings.decodingNonPrediction += Date().timeIntervalSince(nonInferenceStartTime)
                timings.decodingLoop += Date().timeIntervalSince(loopStart)
                timings.totalDecodingLoops += 1
                break
            } else {
                // MARK: KV Caching

                if !isPrefill {
                    // Found the next token, store it
                    currentTokens.append(nextToken)
                    logProbs.append(nextTokenLogProb)
                }

                decoderInputs.decoderKeyPaddingMask[tokenIndex + 1] = 0
                decoderInputs.kvCacheUpdateMask[tokenIndex] = 0
                decoderInputs.kvCacheUpdateMask[tokenIndex + 1] = 1

                // Update alignment weights for token if present
                // TODO: use alignment heads
//                if let newAlignmentWeights = try decoderOutput.cache?.alignmentWeights {
//                    hasAlignment = true
//                    for column in 0..<decoderInputs.alignmentWeights.shape[1].intValue {
//                        let alignmentWeightIndex = [tokenIndex + 1, column] as [NSNumber] // +1 to account for SOT
//                        let weightValue = newAlignmentWeights[[0, column] as [NSNumber]].doubleValue
//                        decoderInputs.alignmentWeights[alignmentWeightIndex] = NSNumber(value: weightValue)
//                    }
//                }

                // Prepare results
                let wordTokens = currentTokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                let slicedTextTokens = options.skipSpecialTokens ? wordTokens : currentTokens
                let currentTranscript = tokenizer.decode(tokens: slicedTextTokens)
                let averageLogProb = logProbs.reduce(0, +) / Float(logProbs.count)
                let compressionRatio = compressionRatio(of: currentTokens)

                let result = TranscriptionProgress(
                    timings: timings,
                    text: currentTranscript,
                    tokens: currentTokens,
                    temperature: nil,
                    avgLogprob: averageLogProb,
                    compressionRatio: compressionRatio
                )
                Logging.debug("Predicted next tokenIndex: \(tokenIndex + 1), token: \(nextToken), text: \(tokenizer.decode(tokens: [nextToken]))")

                // Call the callback if it is provided
                if let shouldContinue = callback?(result) {
                    if !shouldContinue && !isPrefill {
                        Logging.debug("Early stopping")
                        break
                    }
                }
            }

            timings.decodingNonPrediction += Date().timeIntervalSince(nonInferenceStartTime)
            timings.decodingLoop += Date().timeIntervalSince(loopStart)
            timings.totalDecodingLoops += 1

            if tokenIndex == prefilledIndex {
                timings.firstTokenTime = CFAbsoluteTimeGetCurrent()
            }
        }

        let cache = DecodingCache(
            keyCache: decoderInputs.keyCache,
            valueCache: decoderInputs.valueCache,
            alignmentWeights: hasAlignment ? decoderInputs.alignmentWeights : nil
        )

        // NOTE:
        // While `currentTokens` and `logProbs` are usually the same length
        // `currentTokens` does not always contain an end of text token at the end (it is added by this finalize function),
        let finalSamplingResult = tokenSampler.finalize(tokens: currentTokens, logProbs: logProbs)
        let segmentTokens = finalSamplingResult.tokens
        let segmentLogProbs = finalSamplingResult.logProbs

        let startIndex = segmentTokens.firstIndex(of: tokenizer.specialTokens.startOfTranscriptToken) ?? 0
        let endIndex = segmentTokens.firstIndex(of: tokenizer.specialTokens.endToken) ?? segmentTokens.count
        let filteredTokens = Array(segmentTokens[startIndex...endIndex])
        let filteredLogProbs = Array(segmentLogProbs[startIndex...endIndex])

        let sumLogProbs = filteredLogProbs.reduce(0, +)
        let avgLogProbs = sumLogProbs / Float(filteredLogProbs.count)

        var tokenProbs = [[Int: Float]]()
        for (index, token) in filteredTokens.enumerated() {
            tokenProbs.append([token: filteredLogProbs[index]])
        }

        let wordTokens = filteredTokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        let compressionRatio = compressionRatio(of: wordTokens)

        var temperature = options.temperature
        if let sampler = tokenSampler as? MLXGreedyTokenSampler {
            // Convert Float16 temperature to Float with 3 decimal places
            temperature = Float(sampler.temperature).rounded(3)
        }

        let noSpeechProb: Float = 0 // TODO: implement no speech prob

        // If language is still nil here, check language can be inferred from tokens
        var language = options.language ?? Constants.defaultLanguageCode
        var languageProbs = [String: Float]()
        if options.language == nil {
            // Find the first token that is a recognized language token
            if let predictedLanguageIndex = filteredTokens.firstIndex(where: { tokenizer.allLanguageTokens.contains($0) }),
               predictedLanguageIndex < tokenProbs.count
            {
                let predictedLanguageToken = filteredTokens[predictedLanguageIndex]
                // Decode the predicted language token to get the language
                language = tokenizer.decode(tokens: [predictedLanguageToken]).trimmingSpecialTokenCharacters()

                // Fetch the corresponding probability for the predicted language
                let probsDict = tokenProbs[predictedLanguageIndex]
                languageProbs[language] = probsDict[predictedLanguageToken] ?? 0.0
            } else {
                // Set default values if no language token is found
                languageProbs[language] = 0.0
            }
        } else {
            // If language is provided, set the logprob to 0.0
            languageProbs[language] = 0.0
        }

        let transcript = tokenizer.decode(tokens: filteredTokens)

        let decodingFallback = DecodingFallback(
            options: options,
            isFirstTokenLogProbTooLow: isFirstTokenLogProbTooLow,
            noSpeechProb: noSpeechProb,
            compressionRatio: compressionRatio,
            avgLogProb: avgLogProbs
        )

        let decodingResult = DecodingResult(
            language: language,
            languageProbs: languageProbs,
            tokens: filteredTokens,
            tokenLogProbs: tokenProbs,
            text: transcript,
            avgLogProb: avgLogProbs,
            noSpeechProb: noSpeechProb,
            temperature: temperature,
            compressionRatio: compressionRatio,
            cache: cache,
            timings: timings,
            fallback: decodingFallback
        )
        return decodingResult
    }

    public func detectLanguage(
        from encoderOutput: MLMultiArray,
        using decoderInputs: DecodingInputs,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> DecodingResult {
        guard let tokenizer else {
            throw WhisperError.tokenizerUnavailable()
        }
        guard let logitsSize else {
            throw WhisperError.modelsUnavailable("Failed to read logits size from model")
        }
        let languageLogitsFilter = self.languageLogitsFilter ?? LanguageLogitsFilter(
            allLanguageTokens: tokenizer.allLanguageTokens,
            logitsDim: logitsSize,
            sampleBegin: 0
        )
        self.languageLogitsFilter = languageLogitsFilter
        return try await MLXTextDecoder.detectLanguage(
            textDecoder: self,
            languageLogitsFilter: languageLogitsFilter,
            from: encoderOutput,
            using: decoderInputs,
            sampler: tokenSampler,
            options: options,
            temperature: temperature
        )
    }
}

extension MLXTextDecoder: WhisperMLXModel {
    public func loadModel(at modelPath: URL, configPath: URL?) async throws {
        let parameters = try loadParameters(at: modelPath)
        let config = try loadConfig(at: configPath)
        let decoder = TextDecoderModule(
            nVocab: config.nVocab,
            nCtx: config.nTextCtx,
            nState: config.nTextState,
            nHead: config.nTextHead,
            nLayer: config.nTextLayer,
            dtype: .float16
        )
        let loadedDecoder = try decoder.update(parameters: parameters, verify: [.noUnusedKeys])
        MLX.eval(loadedDecoder)
        self.model = loadedDecoder
        self.config = config
    }

    public func unloadModel() {
        model = nil
        config = nil
        prefillData = nil
        languageLogitsFilter = nil
    }
}

public class TextDecoderModule: Module {
    let nVocab: Int
    let nCtx: Int
    let nState: Int
    let nHead: Int
    let nLayer: Int
    let dtype: MLX.DType

    @ModuleInfo(key: "token_embedding") private var tokenEmbedding: Embedding
    @ModuleInfo(key: "positional_embedding") private var positionalEmbedding: MLXArray
    @ModuleInfo(key: "blocks") private var blocks: [ResidualAttentionBlock]
    @ModuleInfo(key: "ln") private var ln: LayerNorm
    private let _mask: MLXArray

    init(
        nVocab: Int,
        nCtx: Int,
        nState: Int,
        nHead: Int,
        nLayer: Int,
        dtype: MLX.DType = .float16
    ) {
        self.nVocab = nVocab
        self.nCtx = nCtx
        self.nState = nState
        self.nHead = nHead
        self.nLayer = nLayer
        self.dtype = dtype

        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: nVocab, dimensions: nState)
        self._positionalEmbedding.wrappedValue = MLX.zeros([nCtx, nState])
        self._blocks.wrappedValue = (0..<nLayer).map { _ in
            ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: true)
        }
        self._ln.wrappedValue = LayerNorm(dimensions: nState)
        self._mask = additiveCausalMask(nCtx).asType(dtype)
    }

    func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray,
        kvCache: [KV?]?
    ) -> TextDecoderResult {
        let offset = kvCache?.first??.k.shape[1] ?? 0
        var x = x[.newAxis, .ellipsis]
        x = tokenEmbedding(x) + positionalEmbedding[offset..<offset + x.shape[x.shape.count - 1]]
        var kvCache: [KV?] = kvCache ?? Array(repeating: nil, count: blocks.count)
        var crossQK: [MLXArray?] = Array(repeating: nil, count: blocks.count)

        for (index, block) in blocks.enumerated() {
            let result = block(x, xa: xa, mask: _mask, kvCache: kvCache[index])
            x = result.x
            kvCache[index] = result.kv
            crossQK[index] = result.crossQk
        }
        x = ln(x)
        return TextDecoderResult(
            logits: tokenEmbedding.asLinear(x),
            kvCache: kvCache.compactMap { $0 },
            alignmentWeights: crossQK.compactMap { $0 }
        )
    }
}
