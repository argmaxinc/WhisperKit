import CoreML
import Foundation
@testable import WhisperKit
import XCTest

enum TestError: Error {
    case missingFile(String)
    case missingDirectory(String)
}

@discardableResult
func XCTUnwrapAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async throws -> T {
    let evaluated = try? await expression()
    return try XCTUnwrap(evaluated, message(), file: file, line: line)
}

@discardableResult
func XCTUnwrapAsync<T>(
    _ expression: @autoclosure () async throws -> T?,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async throws -> T {
    let evaluated = try? await expression()
    return try XCTUnwrap(evaluated, message(), file: file, line: line)
}

func XCTAssertNoThrowAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await expression()
    } catch {
        XCTFail(message(), file: file, line: line)
    }
}

func XCTAssertNoThrowAsync<T>(
    _ expression: @autoclosure () async throws -> T?,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await expression()
    } catch {
        XCTFail(message(), file: file, line: line)
    }
}

func XCTAssertNoThrowAsync(
    _ expression: @autoclosure () async throws -> Void,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        try await expression()
    } catch {
        XCTFail(message(), file: file, line: line)
    }
}

// MARK: Helpers

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
    func transcribe(
        with variant: ModelVariant,
        options: DecodingOptions,
        audioFile: String = "jfk.wav",
        file: StaticString = #file,
        line: UInt = #line
    ) async throws -> TranscriptionResult? {
        let modelPath: String
        switch variant {
            case .largev3:
                modelPath = try largev3ModelPath()
            default:
                modelPath = try tinyModelPath()
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
            throw TestError.missingFile("Missing audio file")
        }
        return try await whisperKit.transcribe(audioPath: audioFileURL, decodeOptions: options)
    }

    func tinyModelPath() throws -> String {
        let modelDir = "whisperkit-coreml/openai_whisper-tiny"
        guard let modelPath = Bundle.module.urls(forResourcesWithExtension: "mlmodelc", subdirectory: modelDir)?.first?.deletingLastPathComponent().path else {
            throw TestError.missingFile("Failed to load model, ensure \"Models/\(modelDir)\" exists via Makefile command: `make download-models`")
        }
        return modelPath
    }

    func largev3ModelPath() throws -> String {
        let modelDir = "whisperkit-coreml/openai_whisper-large-v3" // use faster to compile model for tests
        guard let modelPath = Bundle.module.urls(forResourcesWithExtension: "mlmodelc", subdirectory: modelDir)?.first?.deletingLastPathComponent().path else {
            throw TestError.missingFile("Failed to load model, ensure \"Models/\(modelDir)\" exists via Makefile command: `make download-models`")
        }
        return modelPath
    }

    func largev3TurboModelPath() throws -> String {
        let modelDir = "whisperkit-coreml/openai_whisper-large-v3_turbo"
        guard let modelPath = Bundle.module.urls(forResourcesWithExtension: "mlmodelc", subdirectory: modelDir)?.first?.deletingLastPathComponent().path else {
            throw TestError.missingFile("Failed to load model, ensure \"Models/\(modelDir)\" exists via Makefile command: `make download-models`")
        }
        return modelPath
    }

    func allModelPaths() throws -> [String] {
        let fileManager = FileManager.default
        var modelPaths: [String] = []
        let directory = "whisperkit-coreml"
        let resourceKeys: [URLResourceKey] = [.isDirectoryKey]
        guard let baseurl = Bundle.module.resourceURL?.appendingPathComponent(directory) else {
            throw TestError.missingDirectory("Base URL for directory \(directory) not found.")
        }
        let directoryContents = try fileManager.contentsOfDirectory(at: baseurl, includingPropertiesForKeys: resourceKeys, options: .skipsHiddenFiles)
        for folderURL in directoryContents {
            let resourceValues = try folderURL.resourceValues(forKeys: Set(resourceKeys))
            if resourceValues.isDirectory == true {
                // Check if the directory contains actual data files, or if it contains pointer files.
                // As a proxy, use the MelSpectrogramc.mlmodel/coredata.bin file.
                let proxyFileToCheck = folderURL.appendingPathComponent("MelSpectrogram.mlmodelc/coremldata.bin")
                if try isGitLFSPointerFile(url: proxyFileToCheck) {
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
        return modelPaths
    }
    
    // Function to check if the beginning of the file matches a Git LFS pointer pattern
    func isGitLFSPointerFile(url: URL) throws -> Bool {
        let fileHandle = try FileHandle(forReadingFrom: url)
        // Read the first few bytes of the file to get enough for the Git LFS pointer signature
        let data = fileHandle.readData(ofLength: 512) // Read first 512 bytes
        fileHandle.closeFile()
        if let string = String(data: data, encoding: .utf8),
           string.starts(with: "version https://git-lfs.github.com/") {
            return true
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

extension SpecialTokens {
    static func `default`(
        endToken: Int = 0,
        englishToken: Int = 0,
        noSpeechToken: Int = 0,
        noTimestampsToken: Int = 0,
        specialTokenBegin: Int = 0,
        startOfTranscriptToken: Int = 0,
        timeTokenBegin: Int = 0,
        transcribeToken: Int = 0,
        translateToken: Int = 0,
        whitespaceToken: Int = 0
    ) -> SpecialTokens {
        SpecialTokens(
            endToken: endToken,
            englishToken: englishToken,
            noSpeechToken: noSpeechToken,
            noTimestampsToken: noTimestampsToken,
            specialTokenBegin: specialTokenBegin,
            startOfTranscriptToken: startOfTranscriptToken,
            timeTokenBegin: timeTokenBegin,
            transcribeToken: transcribeToken,
            translateToken: translateToken,
            whitespaceToken: whitespaceToken
        )
    }
}
