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

/// Serializes whole-value reads and writes with an `os_unfair_lock`.
/// Useful for properties on types marked `@unchecked Sendable`.
///
/// **Known limitation — only whole-value replacement is safe:**
///
/// ```swift
/// holder.ref = otherRef        // safe: single locked write
/// holder.count = newCount      // safe: single locked write
///
/// holder.ref.count += 1        // NOT safe: gets the reference (locked),
///                              // then mutates .count with no lock held
/// holder.count += 1            // NOT safe: get and set are two separate
///                              // lock acquisitions with a gap between them
/// ```
///
/// For read-modify-write safety, callers must replace the entire value
/// atomically or use their own external synchronisation.
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

// MARK: - Mutex

@dynamicMemberLookup
public final class Mutex<Value: Sendable>: Sendable {
    private let lock: OSAllocatedUnfairLock<Value>

    public init(_ value: Value) {
        lock = .init(initialState: value)
    }

    public var value: Value {
        lock.withLock { $0 }
    }

    @discardableResult
    public func mutate<Result: Sendable>(_ mutation: @Sendable (inout Value) throws -> Result) rethrows -> Result {
        try lock.withLock { try mutation(&$0) }
    }

    /// Set to the new value and return the old value.
    @discardableResult
    public func set(_ newValue: Value) -> Value {
        lock.withLock { value in
            let oldValue = value
            value = newValue
            return oldValue
        }
    }

    /// Set property to the new value and return the old value.
    @discardableResult
    public func set<T: Sendable>(_ keyPath: WritableKeyPath<Value, T>, to newValue: T) -> T {
        lock.withLock { value in
            let oldValue = value[keyPath: keyPath]
            value[keyPath: keyPath] = newValue
            return oldValue
        }
    }

    public subscript<T: Sendable>(dynamicMember keyPath: WritableKeyPath<Value, T>) -> T {
        get { value[keyPath: keyPath] }
        set { lock.withLock { value in value[keyPath: keyPath] = newValue } }
    }

    public subscript<T>(dynamicMember keyPath: KeyPath<Value, T>) -> T {
        value[keyPath: keyPath]
    }
}

// MARK: - KeyPath + @unchecked @retroactive Sendable

extension KeyPath: @unchecked @retroactive Sendable where Root: Sendable, Value: Sendable { }

