//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import CoreML
import Tokenizers

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
public protocol TextDecoding {
    var tokenizer: Tokenizer? { get set }
    var prefillData: WhisperMLModel? { get set }
    var logitsSize: Int? { get }
    var kvCacheEmbedDim: Int? { get }
    var kvCacheMaxSequenceLength: Int? { get }
    var windowSize: Int? { get }
    var embedSize: Int? { get }

    func predictLogits(
        inputIds: MLMultiArray,
        cacheLength: MLMultiArray,
        keyCache: MLMultiArray,
        valueCache: MLMultiArray,
        kvCacheUpdateMask: MLMultiArray,
        encoderOutputEmbeds: MLMultiArray,
        decoderKeyPaddingMask: MLMultiArray
    ) async throws -> (logits: MLMultiArray?, keyCache: MLMultiArray?, valueCache: MLMultiArray?)?

    func prefillKVCache(withTask task: MLMultiArray, andLanguage language: MLMultiArray) async throws -> (keyCache: MLMultiArray?, valueCache: MLMultiArray?)?
    func decodeText(
        from encoderOutput: MLMultiArray,
        using decoderInputs: DecodingInputs,
        sampler tokenSampler: TokenSampling,
        options decoderOptions: DecodingOptions,
        callback: ((TranscriptionProgress) -> Bool?)?
    ) async throws -> [DecodingResult]

    static func updateKVCache(
        keyTensor: MLMultiArray,
        keySlice: MLMultiArray,
        valueTensor: MLMultiArray,
        valueSlice: MLMultiArray,
        insertAtIndex index: Int
    )
}

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
public extension TextDecoding {
    func prepareDecoderInputs(withPrompt initialPrompt: [Int]) -> DecodingInputs? {
        let tokenShape = [NSNumber(value: 1), NSNumber(value: initialPrompt.count)]

        // Initialize MLMultiArray for tokens
        guard let tokenMultiArray = try? MLMultiArray(shape: tokenShape, dataType: .int32) else {
            fatalError("Error creating MLMultiArray for tokens")
        }

        // Assign token values to the MLMultiArray
        for (index, token) in initialPrompt.enumerated() {
            tokenMultiArray[index] = NSNumber(value: token)
        }

        guard let kvCacheEmbedDim = self.kvCacheEmbedDim else {
            Logging.error("Unable to determine kvCacheEmbedDim")
            return nil
        }

        guard let kvCacheMaxSequenceLength = self.kvCacheMaxSequenceLength else {
            Logging.error("Unable to determine kvCacheMaxSequenceLength")
            return nil
        }

        // Initialize each MLMultiArray
        let kvCacheEmbedDimValue = NSNumber(value: kvCacheEmbedDim)
        let kvCacheMaxSequenceLengthValue = NSNumber(value: kvCacheMaxSequenceLength)

        let inputIds = initMLMultiArray(shape: [1], dataType: .int32, initialValue: Int32(0))
        let cacheLength = initMLMultiArray(shape: [1], dataType: .int32, initialValue: Int32(0))
        let keyCache = initMLMultiArray(shape: [1, kvCacheEmbedDimValue, 1, kvCacheMaxSequenceLengthValue], dataType: .float16, initialValue: FloatType(0))
        let valueCache = initMLMultiArray(shape: [1, kvCacheEmbedDimValue, 1, kvCacheMaxSequenceLengthValue], dataType: .float16, initialValue: FloatType(0))
        let kvCacheUpdateMask = initMLMultiArray(shape: [1, kvCacheMaxSequenceLengthValue], dataType: .int32, initialValue: Int32(0))
        let decoderKeyPaddingMask = initMLMultiArray(shape: [1, kvCacheMaxSequenceLengthValue], dataType: .float16, initialValue: FloatType(-10000))
        let prefillKeyCache = try! MLMultiArray(shape: [1, kvCacheEmbedDimValue, 1, kvCacheMaxSequenceLengthValue], dataType: .float16)
        let prefillValueCache = try! MLMultiArray(shape: [1, kvCacheEmbedDimValue, 1, kvCacheMaxSequenceLengthValue], dataType: .float16)

        let decoderInputs = DecodingInputs(
            initialPrompt: initialPrompt,
            inputIds: inputIds,
            cacheLength: cacheLength,
            keyCache: keyCache,
            valueCache: valueCache,
            kvCacheUpdateMask: kvCacheUpdateMask,
            decoderKeyPaddingMask: decoderKeyPaddingMask,
            prefillKeyCache: prefillKeyCache,
            prefillValueCache: prefillValueCache
        )

        return decoderInputs
    }

    func prefillDecoderInputs(_ decoderInputs: DecodingInputs, withOptions options: DecodingOptions?, multilingual: Bool) async throws -> DecodingInputs {
        guard let tokenizer = tokenizer else {
            // Tokenizer required for prefill
            throw WhisperError.tokenizerUnavailable()
        }

        var prefilledDecoderInputs = decoderInputs

        // Setup prefill tokens based on task and language
        var prefillTokens: [Int] = [tokenizer.startOfTranscriptToken] // SOT

        var languageToken: Int = tokenizer.englishToken
        var taskToken: Int = tokenizer.transcribeToken
        if let options = options,
           multilingual // Multilingual models require language and task tokens
        {
            // Set languageToken
            let languageTokenString = "<|\(options.language ?? "en")|>"
            languageToken = tokenizer.convertTokenToId(languageTokenString) ?? tokenizer.englishToken
            prefillTokens.append(languageToken)

            // Set taskToken
            let taskTokenString = "<|\(options.task)|>"
            taskToken = tokenizer.convertTokenToId(taskTokenString) ?? tokenizer.transcribeToken
            prefillTokens.append(taskToken)
        }

        let timestampsToken = options?.withoutTimestamps ?? false ? tokenizer.noTimestampsToken : tokenizer.timeTokenBegin // withoutTimestamps must be non-nil and true in order to disable it
        prefillTokens.append(timestampsToken)

        prefilledDecoderInputs.initialPrompt = prefillTokens

        if let options = options,
           options.usePrefillCache,
           prefillData != nil
        {
            // Prefilling kv cache data requires non-nil task and language tokens, set defaults if not provided
            // Task tokens are remapped to 0->transcribe and 1->translate for the prefill lookup table
            let task = MLMultiArray.from([taskToken == tokenizer.transcribeToken ? 0 : 1])
            let lang = MLMultiArray.from([languageToken])
            guard let prefillOutput = try await self.prefillKVCache(withTask: task, andLanguage: lang) else {
                Logging.error("Unable to prefill cache")
                return prefilledDecoderInputs
            }

            // Prefill kv cache
            prefilledDecoderInputs.prefillKeyCache = prefillOutput.keyCache!
            prefilledDecoderInputs.prefillValueCache = prefillOutput.valueCache!

            TextDecoder.updateKVCache(keyTensor: prefilledDecoderInputs.keyCache,
                                      keySlice: prefilledDecoderInputs.prefillKeyCache,
                                      valueTensor: prefilledDecoderInputs.valueCache,
                                      valueSlice: prefilledDecoderInputs.prefillValueCache,
                                      insertAtIndex: 0)
            prefilledDecoderInputs.cacheLength[0] = prefilledDecoderInputs.prefillKeyCache.shape[3]
        }

        return prefilledDecoderInputs
    }

    func prefillKVCache(withTask task: MLMultiArray, andLanguage language: MLMultiArray) async throws -> (keyCache: MLMultiArray?, valueCache: MLMultiArray?)? {
        let modelInputs = TextDecoderCachePrefillInput(
            task: task,
            language: language
        )

        guard let prefillModel = prefillData?.model else {
            return nil
        }

        try Task.checkCancellation()

        let outputFeatures = try await prefillModel.prediction(from: modelInputs, options: MLPredictionOptions())

        let output = TextDecoderCachePrefillOutput(features: outputFeatures)

        let keyCache = output.key_cache_prefill
        let valueCache = output.value_cache_prefill

        return (keyCache, valueCache)
    }

    static func updateKVCache(keyTensor: MLMultiArray, keySlice: MLMultiArray,
                              valueTensor: MLMultiArray, valueSlice: MLMultiArray,
                              insertAtIndex index: Int)
    {
        let tensorShape = keyTensor.shape.map { $0.intValue }
        let sliceShape = keySlice.shape.map { $0.intValue }
        let sliceStrides = keySlice.strides.map { $0.intValue } // same for val
        let bytesPerSample = MemoryLayout<FloatType>.size

        keyTensor.withUnsafeMutableBytes { keyTensorPointer, keyTargetStrides in
            keySlice.withUnsafeBytes { keySlicePointer in
                valueTensor.withUnsafeMutableBytes { valueTensorPointer, valueTargetStrides in
                    valueSlice.withUnsafeBytes { valueSlicePointer in
                        // Assuming batch size is always 1
                        DispatchQueue.concurrentPerform(iterations: tensorShape[1]) { j in
                            // Slice size is 3 for prefill and 1 for decode loops
                            for k in 0..<sliceShape[3] {
                                /// Equivalent to:
                                /// `tensor[0, j, 0, k + index] = slice[0, j, 0, k + index]`
                                let keyDestIndex = j * keyTargetStrides[1] + (index + k) * keyTargetStrides[3]
                                let keyDest = keyTensorPointer.baseAddress! + keyDestIndex * bytesPerSample

                                let keySliceIndex = j * sliceStrides[1] + k * sliceStrides[3]
                                let keySlice = keySlicePointer.baseAddress! + keySliceIndex * bytesPerSample
                                memcpy(keyDest, keySlice, bytesPerSample)

                                let valDestIndex = j * valueTargetStrides[1] + (index + k) * valueTargetStrides[3]
                                let valDest = valueTensorPointer.baseAddress! + valDestIndex * bytesPerSample

                                let valSliceIndex = j * sliceStrides[1] + k * sliceStrides[3]
                                let valSlice = valueSlicePointer.baseAddress! + valSliceIndex * bytesPerSample
                                memcpy(valDest, valSlice, bytesPerSample)
                            }
                        }
                    }
                }
            }
        }
    }
}

public class TextDecoderContextPrefill: WhisperMLModel {
    public var model: MLModel?
}

@available(macOS 14, iOS 17, watchOS 10, visionOS 1, *)
public class TextDecoder: TextDecoding, WhisperMLModel {
    public var model: MLModel?
    public var tokenizer: Tokenizer?
    public var prefillData: WhisperMLModel?

    public var logitsSize: Int? {
        return getModelOutputDimention(model, named: "logits", position: 2)
    }

    public var kvCacheEmbedDim: Int? {
        return getModelInputDimention(model, named: "key_cache", position: 1)
    }

    public var kvCacheMaxSequenceLength: Int? {
        return getModelInputDimention(model, named: "key_cache", position: 3)
    }

    public var windowSize: Int? {
        return getModelInputDimention(model, named: "encoder_output_embeds", position: 3)
    }

    public var embedSize: Int? {
        return getModelInputDimention(model, named: "encoder_output_embeds", position: 1)
    }

    /// Override default so we an unload the prefill data as well
    public func unloadModel() {
        model = nil
        prefillData = nil
    }

    public func predictLogits(
        inputIds: MLMultiArray,
        cacheLength: MLMultiArray,
        keyCache: MLMultiArray,
        valueCache: MLMultiArray,
        kvCacheUpdateMask: MLMultiArray,
        encoderOutputEmbeds: MLMultiArray,
        decoderKeyPaddingMask: MLMultiArray
    ) async throws -> (logits: MLMultiArray?, keyCache: MLMultiArray?, valueCache: MLMultiArray?)? {
        let modelInputs = TextDecoderInput(
            input_ids: inputIds,
            cache_length: cacheLength,
            key_cache: keyCache,
            value_cache: valueCache,
            kv_cache_update_mask: kvCacheUpdateMask,
            encoder_output_embeds: encoderOutputEmbeds,
            decoder_key_padding_mask: decoderKeyPaddingMask
        )

        guard let model = model else {
            return nil
        }

        try Task.checkCancellation()

        let outputFeatures = try await model.prediction(from: modelInputs, options: MLPredictionOptions())

        let output = TextDecoderOutput(features: outputFeatures)

        let logits = output.logits
        let keyCache = output.key_cache_updates
        let valueCache = output.value_cache_updates

        return (logits, keyCache, valueCache)
    }

    public func detectLanguage(
        from encoderOutput: MLMultiArray,
        using decoderInputs: DecodingInputs,
        options decoderOptions: DecodingOptions,
        temperature: FloatType
    ) -> [DecodingResult] {
        // TODO: implement
        // predict logits for 1 iteration with sot
        // 1. LanguageLogitsFilter for only language tokens
        // 2. GreedyTokenSampler for most likely language
        return []
    }

    public func decodeText(
        from encoderOutput: MLMultiArray,
        using decoderInputs: DecodingInputs,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        callback: TranscriptionCallback = nil
    ) async throws -> [DecodingResult] {
        // Decode text loop timings
        var timings = TranscriptionTimings()
        let prefilledIndex = decoderInputs.cacheLength[0].intValue
        let intialPromptIndex = decoderInputs.initialPrompt.count - 1
        var currentTokens: [Int] = decoderInputs.initialPrompt
        var nextToken: Int = decoderInputs.initialPrompt.last!

        guard let tokenizer = tokenizer else {
            // Tokenizer required for decoding
            throw WhisperError.tokenizerUnavailable()
        }

        // Sampling
        var logProbs: [Float] = []

        var logitsFilters: [any LogitsFiltering] = []
        if options.suppressBlank {
            logitsFilters.append(
                SuppressBlankFilter(
                    suppressBlankTokens: [tokenizer.whitespaceToken, tokenizer.endToken],
                    sampleBegin: prefilledIndex
                )
            )
        }

        if !options.supressTokens.isEmpty {
            logitsFilters.append(SuppressTokensFilter(suppressTokens: options.supressTokens))
        }

        if !options.withoutTimestamps {
            // TODO: implement
            // logitsFilters.append(TimestampRulesFilter(tokenizer: tokenizer, sampleBegin: prefilledIndex))
        }

        // MARK: Main loop

        let loopCount = min(options.sampleLength, WhisperKit.maxTokenContext)
        Logging.debug("Running main loop for a maximum of \(loopCount) iterations, starting at index \(prefilledIndex)")
        var isPrefill: Bool
        for tokenIndex in prefilledIndex..<loopCount {
            let loopStart = Date()

            // Check if current index is part of the initial prompt
            isPrefill = false
            if tokenIndex <= intialPromptIndex {
                isPrefill = tokenIndex < intialPromptIndex // Prefill stops at the last token of the initial prompt
                let prefillToken = currentTokens[tokenIndex]
                nextToken = prefillToken
                Logging.debug("Forcing token \(nextToken) at index \(tokenIndex) from initial prompt")
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
                keyCache: decoderInputs.keyCache,
                valueCache: decoderInputs.valueCache,
                kvCacheUpdateMask: decoderInputs.kvCacheUpdateMask,
                encoderOutputEmbeds: encoderOutput,
                decoderKeyPaddingMask: decoderInputs.decoderKeyPaddingMask
            )

            guard let decoderOutput = predictedLogits else {
                Logging.error("Unable to decode logits")
                throw WhisperError.decodingLogitsFailed()
            }

            let decodingInferenceTime = Date().timeIntervalSince(inferenceTime)
            timings.decodingPredictions += decodingInferenceTime

            // MARK: Non-inference

            let nonInferenceStartTime = Date()

            // Update predicted token as current
            var logits = decoderOutput.logits!
            for filter in logitsFilters {
                logits = filter.filterLogits(logits, withTokens: currentTokens)
            }

            // MARK: Sampling

            let samplingStartTime = Date()

            let sampleResult = tokenSampler.update(tokens: currentTokens, logits: logits, logProbs: logProbs)

            nextToken = sampleResult.tokens.last!
            logProbs = sampleResult.logProbs

            let samplingTime = Date().timeIntervalSince(samplingStartTime)
            timings.decodingSampling += samplingTime

            if sampleResult.completed || currentTokens.count >= WhisperKit.maxTokenContext {
                // Completed segment, stop the loop
                break
            } else {
                // MARK: KV Caching

                if !isPrefill {
                    // Found the next token, store it
                    currentTokens.append(nextToken)
                }

                // Update KV cache for this token
                guard let newKeyCache = decoderOutput.keyCache,
                      let newValueCache = decoderOutput.valueCache
                else {
                    fatalError("Invalid model output")
                }

                // tensor: [1, kvCacheEmbedDim, 1, kvCacheMaxSequenceLength], slice: [1, kvCacheEmbedDim, 1, 1]
                let kvStartTime = Date()
                TextDecoder.updateKVCache(keyTensor: decoderInputs.keyCache,
                                          keySlice: newKeyCache,
                                          valueTensor: decoderInputs.valueCache,
                                          valueSlice: newValueCache,
                                          insertAtIndex: tokenIndex)
                let kvTime = Date().timeIntervalSince(kvStartTime)
                timings.decodingKvCaching += kvTime
                timings.totalKVUpdateRuns += 1

                decoderInputs.decoderKeyPaddingMask[tokenIndex + 1] = 0

                decoderInputs.kvCacheUpdateMask[tokenIndex] = 0
                decoderInputs.kvCacheUpdateMask[tokenIndex + 1] = 1

                let currentTranscript = tokenizer.decode(tokens: currentTokens)
                let averageLogProb = logProbs.reduce(0, +) / Float(logProbs.count)
                let compressionRatio = compressionRatio(of: currentTokens)

                let result = TranscriptionProgress(timings: timings, text: currentTranscript, tokens: currentTokens, avgLogprob: averageLogProb, compressionRatio: compressionRatio)
                Logging.debug("tokenIndex: \(tokenIndex), token: \(nextToken), word: \(tokenizer.decode(tokens: [nextToken]))")

                // Call the callback if it is provided
                if let shouldContinue = callback?(result) {
                    if !shouldContinue {
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

        let (segmentTokens, sumLogProbs) = tokenSampler.finalize(tokens: currentTokens, logProbs: logProbs)
        let avgLogProbs = sumLogProbs / Float(logProbs.count)

        let wordTokens = segmentTokens.filter { $0 < tokenizer.specialTokenBegin }
        let compressionRatio = compressionRatio(of: wordTokens)

        var temperature = options.temperature
        if let sampler = tokenSampler as? GreedyTokenSampler {
            // Convert Float16 temperature to Float with 3 decimal places
            temperature = round(Float(sampler.temperature) * 1000) / 1000
        }

        let transcript = tokenizer.decode(tokens: segmentTokens)

        let decodingResult = DecodingResult(
            language: options.language ?? "en",
            languageProbs: [options.language ?? "en": 1.0],
            tokens: segmentTokens,
            text: transcript,
            avgLogProb: avgLogProbs,
            noSpeechProb: 0, // TODO: implement no speech prob
            temperature: temperature,
            compressionRatio: compressionRatio,
            timings: timings
        )

        return [decodingResult]
    }

    func debugCaches(decoderInputs: DecodingInputs, tokenIndex: Int, prefillSize: Int) {
        Logging.debug("--------------- DECODER INPUTS DEBUG ---------------")
        Logging.debug(
            String(
                format: "Cache Length: %2.0f Input Token: %4.0f",
                decoderInputs.cacheLength[0].floatValue,
                decoderInputs.inputIds[0].floatValue
            )
        )
        Logging.debug("Key Cache | Val Cache | Update Mask | Decoder Mask | Position")

        for i in 0..<prefillSize + 4 {
            let formattedString = String(format: "%9.6f | %9.6f | %11.0f | %12.0f | %d",
                                         decoderInputs.keyCache[i].floatValue,
                                         decoderInputs.valueCache[i].floatValue,
                                         decoderInputs.kvCacheUpdateMask[i].floatValue,
                                         decoderInputs.decoderKeyPaddingMask[i].floatValue,
                                         i)
            Logging.debug(formattedString)
        }
    }
}
