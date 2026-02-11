//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

// MARK: - TTSGenerating

/// Protocol for a single-chunk speech synthesis task.
///
/// `TTSKit` creates tasks via `makeTask(progress:)` and calls `run(...)` on each one.
/// Each task instance is independent - fresh KV caches, own RNG - so multiple tasks
/// can run concurrently without data races.
///
/// The interface uses plain `String` for `voice` and `language` to stay model-agnostic.
/// Each implementation maps these to its internal representation (e.g. token IDs).
///
/// **Reference implementation:** `TTSGenerateTask` in `Sources/TTSKit/Qwen3TTS/Qwen3TTSGenerateTask.swift`.
public protocol TTSGenerating: Sendable {
    /// Generate speech for a single text chunk, delivering per-step audio via `callback`.
    ///
    /// - Parameters:
    ///   - text: The text chunk to synthesise.
    ///   - voice: Voice identifier (e.g. `Qwen3Speaker.rawValue` in the Qwen3 backend).
    ///   - language: Language identifier (e.g. `Qwen3Language.rawValue`).
    ///   - options: Generation options (temperature, top-k, chunking, concurrency, etc.)
    ///   - callback: Called on every decoded step with audio samples and running timings.
    ///               `TTSProgress.stepTime` is non-nil only on the first step, allowing
    ///               adaptive playback buffer configuration before audio starts.
    ///               Return `false` to cancel; `nil` or `true` to continue.
    func run(
        text: String,
        voice: String,
        language: String,
        options: TTSGenerationOptions,
        callback: TTSCallback
    ) async throws -> TTSResult
}
