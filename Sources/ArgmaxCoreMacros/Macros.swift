//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftSyntaxMacros

@attached(member, names: named(_internalState), named(_InternalState), named(inLock))
@attached(memberAttribute)
public macro ThreadSafe() = #externalMacro(module: "ArgmaxCoreMacroPlugin", type: "ThreadSafeMacro")

@attached(accessor)
public macro ThreadSafeProperty() = #externalMacro(module: "ArgmaxCoreMacroPlugin", type: "ThreadSafePropertyMacro")

@attached(body)
public macro ThreadSafeInitializer(_ params: [String: Any]) = #externalMacro(
  module: "ArgmaxCoreMacroPlugin",
  type: "ThreadSafeInitializerMacro")

// MARK: - TypeErased

/// Type-erased wrapper to allow storing values of any type in a homogeneous collection.
/// This is used for storing property values in the thread-safe wrapper without exposing their types.
public struct TypeErased<T> {
  let value: T?

  public init(value: T? = nil) {
    self.value = value
  }
}
