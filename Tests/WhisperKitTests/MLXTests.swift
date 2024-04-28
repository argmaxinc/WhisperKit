//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import XCTest
@testable import WhisperKit
@testable import WhisperKitMLX

final class MLXTests: XCTestCase {

    // MARK: - Feature Extractor Tests

    func testLogmelOutput() async throws {
        let audioSamples = [Float](repeating: 0.0, count: 16000)
        let paddedSamples = try XCTUnwrap(
            AudioProcessor.padOrTrimAudio(fromArray: audioSamples, startAt: 0, toLength: 480_000),
            "Failed to pad audio samples"
        )
        let featureExtractor = MLXFeatureExtractor()
        let melSpectrogram = try await XCTUnwrapAsync(
            await featureExtractor.logMelSpectrogram(fromAudio: paddedSamples),
            "Failed to produce Mel spectrogram from audio samples"
        )
        let expectedShape: [NSNumber] = [3000, 80]
        XCTAssertNotNil(melSpectrogram, "Failed to produce Mel spectrogram from audio samples")
        XCTAssertEqual(melSpectrogram.shape, expectedShape, "Mel spectrogram shape is not as expected")
    }
}
