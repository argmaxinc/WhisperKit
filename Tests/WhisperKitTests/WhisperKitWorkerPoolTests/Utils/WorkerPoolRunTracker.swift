//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

actor WorkerPoolRunTracker {
    private var activeRuns: Int = 0
    private var maxConcurrentRuns: Int = 0

    var maxConcurrentRunsObserved: Int {
        maxConcurrentRuns
    }

    func beginRun() {
        activeRuns += 1
        maxConcurrentRuns = max(maxConcurrentRuns, activeRuns)
    }

    func endRun() {
        activeRuns -= 1
    }
}
