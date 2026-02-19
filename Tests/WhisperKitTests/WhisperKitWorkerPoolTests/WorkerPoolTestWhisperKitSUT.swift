//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
@testable import WhisperKit

final class WorkerPoolTestWhisperKitSUT: WhisperKit {
    private let installTokenizerOnLoad: Bool

    init(
        runTracker: WorkerPoolRunTracker,
        installTokenizerOnLoad: Bool
    ) async throws {
        self.installTokenizerOnLoad = installTokenizerOnLoad

        let config = WhisperKitConfig(
            featureExtractor: WorkerPoolTestFeatureExtractorMock(),
            audioEncoder: WorkerPoolTestAudioEncoderMock(),
            textDecoder: WorkerPoolTestTextDecoderMock(runTracker: runTracker),
            segmentSeeker: WorkerPoolTestSegmentSeekerMock(),
            verbose: false,
            logLevel: .none,
            prewarm: false,
            load: false,
            download: false
        )
        try await super.init(config)
    }

    override func loadModels(prewarmMode: Bool = false) async throws {
        guard installTokenizerOnLoad else {
            return
        }
        if tokenizer == nil {
            tokenizer = WorkerPoolTestTokenizerMock()
        }
    }
}
