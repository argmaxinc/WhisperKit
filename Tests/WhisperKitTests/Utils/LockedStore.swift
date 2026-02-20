//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import os

final class LockedStore<Value: Sendable>: @unchecked Sendable {
    private let lock: OSAllocatedUnfairLock<Value>

    init(_ initialValue: Value) {
        self.lock = OSAllocatedUnfairLock(initialState: initialValue)
    }

    func withValue<T: Sendable>(_ body: @Sendable (inout Value) -> T) -> T {
        lock.withLock(body)
    }
}
