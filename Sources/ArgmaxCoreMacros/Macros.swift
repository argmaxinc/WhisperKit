//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import SwiftCompilerPlugin
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

public struct TypeErased<T> {
  let value: T?

  public init(value: T? = nil) {
    self.value = value
  }
}
