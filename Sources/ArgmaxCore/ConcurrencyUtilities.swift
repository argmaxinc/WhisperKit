//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import os.lock

// MARK: - Concurrency Utilities

public struct ConcurrencyUtilities {
    private init() {}

    /// Number of active processors on this device.
    public static var activeProcessorCount: Int {
        ProcessInfo.processInfo.activeProcessorCount
    }
}

// MARK: - Unfair Lock

/// Thin wrapper around `os_unfair_lock` that exposes a Swift-friendly
/// `withLock` helper. This lock is non-reentrant and optimized for low
/// contention, matching the semantics of Core Foundation's unfair lock.
public final class UnfairLock: @unchecked Sendable {
    var lock = os_unfair_lock()

    public init() {}

    public func withLock<T>(_ body: () throws -> T) rethrows -> T {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return try body()
    }
}

// MARK: - Property Lock

/// Serializes access to a value with an `os_unfair_lock` so mutation stays
/// thread-safe. Useful for properties on types marked `@unchecked Sendable`.
@propertyWrapper
public struct PropertyLock<Value: Codable & Sendable>: Sendable, Codable {
    private let lock: UnfairLock
    private var value: Value

    public init(wrappedValue: Value) {
        self.lock = UnfairLock()
        self.value = wrappedValue
    }

    public init(from decoder: Swift.Decoder) throws {
        self.lock = UnfairLock()
        self.value = try Value(from: decoder)
    }

    public func encode(to encoder: Encoder) throws {
        try lock.withLock {
            try value.encode(to: encoder)
        }
    }

    public var wrappedValue: Value {
        get {
            lock.withLock {
                return value
            }
        }
        set {
            lock.withLock {
                value = newValue
            }
        }
    }
}

// MARK: - Early Stop Actor

/// An actor that provides thread-safe early stopping functionality using UUIDs as keys.
public actor EarlyStopActor {
    private var shouldStop = [UUID: Bool]()

    public init() {}

    /// Sets the stop flag for a given UUID
    public func set(_ value: Bool, for uuid: UUID) {
        shouldStop[uuid] = value
    }

    /// Gets the stop flag for a given UUID
    public func get(for uuid: UUID) -> Bool {
        return shouldStop[uuid] ?? false
    }

    /// Removes and returns the stop flag for a given UUID
    @discardableResult
    public func remove(for uuid: UUID) -> Bool? {
        return shouldStop.removeValue(forKey: uuid)
    }
}
