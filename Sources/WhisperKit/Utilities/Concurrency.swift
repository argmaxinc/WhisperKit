//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// An actor that provides thread-safe early stopping functionality using UUIDs as keys
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
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
