//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import os.lock

/// An actor that provides thread-safe early stopping functionality using UUIDs as keys
public actor EarlyStopActor {
    private var shouldStop = [UUID: Bool]()

    public init() {}

    /// Sets the stop flag for a given UUID
    /// - Parameters:
    ///   - value: The boolean value to set
    ///   - uuid: The UUID key
    public func set(_ value: Bool, for uuid: UUID) {
        shouldStop[uuid] = value
    }

    /// Gets the stop flag for a given UUID
    /// - Parameter uuid: The UUID key
    /// - Returns: The current stop flag value, or false if not set
    public func get(for uuid: UUID) -> Bool {
        return shouldStop[uuid] ?? false
    }

    /// Removes and returns the stop flag for a given UUID
    /// - Parameter uuid: The UUID key
    /// - Returns: The removed stop flag value, if it existed
    public func remove(for uuid: UUID) -> Bool? {
        return shouldStop.removeValue(forKey: uuid)
    }
}

/// Serializes access to a value with an `os_unfair_lock` so mutation stays
/// thread-safe. The wrapper is used by `TranscriptionResult`, which is marked
/// `@unchecked Sendable`; guarding each property with this lock helps keep the
/// result instance safe when shared across concurrent contexts.
@propertyWrapper
public struct TranscriptionPropertyLock<Value: Codable & Sendable>: Sendable, Codable {
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

/// Thin wrapper around `os_unfair_lock` that exposes a Swift-friendly
/// `withLock` helper. This lock is non-reentrant and optimized for low
/// contention, matching the semantics of Core Foundation’s unfair lock.
@usableFromInline
final class UnfairLock: @unchecked Sendable {
    @usableFromInline
    var lock = os_unfair_lock()

    @inlinable
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return try body()
    }
}
