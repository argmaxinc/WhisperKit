//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import os
@testable import WhisperKit
import XCTest

final class WhisperKitTests: XCTestCase {
    private func makeUnloadedWhisperKit() async throws -> WhisperKit {
        let config = WhisperKitConfig(
            verbose: false,
            logLevel: .error,
            load: false,
            download: false
        )
        return try await WhisperKit(config)
    }

    func testTokenizerSetterSynchronizesTextDecoderTokenizer() async throws {
        let whisperKit = try await makeUnloadedWhisperKit()
        let tokenizer = WhisperTokenizerMock(specialTokenBegin: 1000)

        whisperKit.tokenizer = tokenizer

        let decoderTokenizer = try XCTUnwrap(whisperKit.textDecoder.tokenizer as? WhisperTokenizerMock)
        XCTAssertTrue(decoderTokenizer === tokenizer, "Setting tokenizer should also update textDecoder.tokenizer")
    }

    func testTextDecoderSetterPreservesExistingTokenizer() async throws {
        let whisperKit = try await makeUnloadedWhisperKit()
        let tokenizer = WhisperTokenizerMock(specialTokenBegin: 1000)
        whisperKit.tokenizer = tokenizer

        let replacementDecoder = TextDecoder()
        whisperKit.textDecoder = replacementDecoder

        let replacementTokenizer = try XCTUnwrap(replacementDecoder.tokenizer as? WhisperTokenizerMock)
        XCTAssertTrue(replacementTokenizer === tokenizer, "Replacing textDecoder should preserve existing tokenizer")
    }

    func testModelStateCallbackTransitionOrderDuringUnload() async throws {
        let whisperKit = try await makeUnloadedWhisperKit()
        let callbackExpectation = expectation(description: "Model state callback fired for unload transition")
        callbackExpectation.expectedFulfillmentCount = 2

        let transitions = LockedStore<[(old: String, new: String)]>([])
        whisperKit.modelStateCallback = { oldState, newState in
            let oldDescription = oldState?.description ?? "nil"
            let newDescription = newState.description
            transitions.withValue {
                $0.append((old: oldDescription, new: newDescription))
            }
            callbackExpectation.fulfill()
        }

        await whisperKit.unloadModels()
        await fulfillment(of: [callbackExpectation], timeout: 1)

        let snapshot = transitions.withValue { $0 }
        XCTAssertEqual(snapshot.count, 2)
        XCTAssertEqual(snapshot[0].old, ModelState.unloaded.description)
        XCTAssertEqual(snapshot[0].new, ModelState.unloading.description)
        XCTAssertEqual(snapshot[1].old, ModelState.unloading.description)
        XCTAssertEqual(snapshot[1].new, ModelState.unloaded.description)
    }

    func testConcurrentWhisperKitPropertyAccess() async throws {
        let whisperKit = try await makeUnloadedWhisperKit()
        let iterations = 200

        await withTaskGroup(of: Void.self) { taskGroup in
            for index in 0..<iterations {
                taskGroup.addTask {
                    switch index % 4 {
                        case 0:
                            whisperKit.modelCompute = ModelComputeOptions(
                                melCompute: .cpuOnly,
                                audioEncoderCompute: .cpuOnly,
                                textDecoderCompute: .cpuOnly,
                                prefillCompute: .cpuOnly
                            )
                        case 1:
                            whisperKit.audioInputConfig = AudioInputConfig(channelMode: .sumChannels([0]))
                            whisperKit.modelFolder = URL(fileURLWithPath: "/tmp/model-\(index)")
                        case 2:
                            whisperKit.segmentDiscoveryCallback = { _ in }
                            whisperKit.transcriptionStateCallback = { _ in }
                        default:
                            _ = whisperKit.modelCompute
                            _ = whisperKit.audioInputConfig
                            _ = whisperKit.segmentDiscoveryCallback
                            _ = whisperKit.modelFolder
                    }
                }
            }
        }

        XCTAssertNotNil(whisperKit.modelFolder)
        XCTAssertNotNil(whisperKit.segmentDiscoveryCallback)
    }
}
