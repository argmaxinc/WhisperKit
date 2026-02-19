//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
@testable import WhisperKit
import XCTest

final class WhisperKitWorkerPoolTests: XCTestCase {
    func testTranscribeWithOptionsWorkerPoolPreservesInputOrder() async throws {
        let runTracker = WorkerPoolRunTracker()
        let whisperKit = try await WorkerPoolTestWhisperKitSUT(
            runTracker: runTracker,
            installTokenizerOnLoad: true
        )

        let audioArrays: [[Float]] = [
            Array(repeating: 0.1, count: 20_100),
            Array(repeating: 0.2, count: 20_200),
            Array(repeating: 0.3, count: 20_300),
            Array(repeating: 0.4, count: 20_400),
        ]
        let decodeOptionsArray = audioArrays.enumerated().map { index, _ in
            DecodingOptions(
                promptTokens: [100 + index],
                concurrentWorkerCount: 2
            )
        }

        let results = await whisperKit.transcribeWithOptions(
            audioArrays: audioArrays,
            decodeOptionsArray: decodeOptionsArray
        )

        XCTAssertEqual(results.count, audioArrays.count)
        let maxConcurrentRunsObserved = await runTracker.maxConcurrentRunsObserved
        XCTAssertGreaterThanOrEqual(maxConcurrentRunsObserved, 2, "Expected worker pool to run concurrently")

        for (index, result) in results.enumerated() {
            let output = try result.get()
            XCTAssertEqual(output.count, 1)
            XCTAssertEqual(output[0].text, "\(100 + index)")
        }
    }

    func testTranscribeWithOptionsFallsBackWhenWorkersCannotBeCreated() async throws {
        let whisperKit = try await WorkerPoolTestWhisperKitSUT(
            runTracker: WorkerPoolRunTracker(),
            installTokenizerOnLoad: false
        )

        let audioArrays: [[Float]] = [
            Array(repeating: 0.1, count: 20_100),
            Array(repeating: 0.2, count: 20_200),
            Array(repeating: 0.3, count: 20_300),
        ]
        let decodeOptionsArray = Array(repeating: DecodingOptions(concurrentWorkerCount: 3), count: audioArrays.count)

        let results = await whisperKit.transcribeWithOptions(
            audioArrays: audioArrays,
            decodeOptionsArray: decodeOptionsArray
        )

        XCTAssertEqual(results.count, audioArrays.count)
        XCTAssertTrue(
            results.allSatisfy {
                if case .failure = $0 {
                    return true
                }
                return false
            },
            "Expected failures from missing tokenizer after safe sequential fallback"
        )
    }

    func testTranscribeWithOptionsSequentialPathWhenWorkerCountIsOne() async throws {
        let runTracker = WorkerPoolRunTracker()
        let whisperKit = try await WorkerPoolTestWhisperKitSUT(
            runTracker: runTracker,
            installTokenizerOnLoad: true
        )

        let audioArrays: [[Float]] = [
            Array(repeating: 0.1, count: 20_100),
            Array(repeating: 0.2, count: 20_200),
            Array(repeating: 0.3, count: 20_300),
        ]
        let decodeOptionsArray = audioArrays.enumerated().map { index, _ in
            DecodingOptions(
                promptTokens: [200 + index],
                concurrentWorkerCount: 1
            )
        }

        let results = await whisperKit.transcribeWithOptions(
            audioArrays: audioArrays,
            decodeOptionsArray: decodeOptionsArray
        )

        XCTAssertEqual(results.count, audioArrays.count)
        let maxConcurrentRunsObserved = await runTracker.maxConcurrentRunsObserved
        XCTAssertEqual(maxConcurrentRunsObserved, 1, "Expected sequential path when worker count is 1")

        for (index, result) in results.enumerated() {
            let output = try result.get()
            XCTAssertEqual(output.count, 1)
            XCTAssertEqual(output[0].text, "\(200 + index)")
        }
    }

    func testTranscribeWithOptionsWorkerPoolAppliesSeekOffsetsToSegmentCallback() async throws {
        let runTracker = WorkerPoolRunTracker()
        let whisperKit = try await WorkerPoolTestWhisperKitSUT(
            runTracker: runTracker,
            installTokenizerOnLoad: true
        )
        let seekCollector = IntValueCollector()
        whisperKit.segmentDiscoveryCallback = { segments in
            if let first = segments.first {
                Task {
                    await seekCollector.append(first.seek)
                }
            }
        }

        let audioArrays: [[Float]] = [
            Array(repeating: 0.1, count: 20_100),
            Array(repeating: 0.2, count: 20_200),
        ]
        let decodeOptionsArray = audioArrays.enumerated().map { index, _ in
            DecodingOptions(
                promptTokens: [300 + index],
                concurrentWorkerCount: 2
            )
        }
        let seekOffsets = [600, 900]

        let results = await whisperKit.transcribeWithOptions(
            audioArrays: audioArrays,
            decodeOptionsArray: decodeOptionsArray,
            seekOffsets: seekOffsets
        )

        XCTAssertEqual(results.count, audioArrays.count)
        XCTAssertTrue(results.allSatisfy { if case .success = $0 { return true }; return false })
        await waitForCollectorValues(seekCollector, expectedCount: seekOffsets.count)
        let collectedSeeks = await seekCollector.values
        XCTAssertEqual(Set(collectedSeeks), Set(seekOffsets))
    }

    func testTranscribeWithOptionsWorkerPoolReportsIndexedWindowIds() async throws {
        let runTracker = WorkerPoolRunTracker()
        let whisperKit = try await WorkerPoolTestWhisperKitSUT(
            runTracker: runTracker,
            installTokenizerOnLoad: true
        )
        let windowIdCollector = IntValueCollector()

        let audioArrays: [[Float]] = [
            Array(repeating: 0.1, count: 20_100),
            Array(repeating: 0.2, count: 20_200),
            Array(repeating: 0.3, count: 20_300),
            Array(repeating: 0.4, count: 20_400),
        ]
        let decodeOptionsArray = audioArrays.enumerated().map { index, _ in
            DecodingOptions(
                promptTokens: [400 + index],
                concurrentWorkerCount: 2
            )
        }
        let callback: TranscriptionCallback = { progress in
            Task {
                await windowIdCollector.append(progress.windowId)
            }
            return true
        }

        let results = await whisperKit.transcribeWithOptions(
            audioArrays: audioArrays,
            decodeOptionsArray: decodeOptionsArray,
            callback: callback
        )

        XCTAssertEqual(results.count, audioArrays.count)
        XCTAssertTrue(results.allSatisfy { if case .success = $0 { return true }; return false })
        await waitForCollectorValues(windowIdCollector, expectedCount: audioArrays.count)
        let collectedWindowIds = await windowIdCollector.values
        XCTAssertEqual(Set(collectedWindowIds), Set(0..<audioArrays.count))
    }

    private func waitForCollectorValues(
        _ collector: IntValueCollector,
        expectedCount: Int,
        maxAttempts: Int = 100
    ) async {
        for _ in 0..<maxAttempts {
            if await collector.count >= expectedCount {
                return
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
    }
}




