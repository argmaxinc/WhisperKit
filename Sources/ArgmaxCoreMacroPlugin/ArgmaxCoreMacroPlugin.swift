//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct ArgmaxCoreMacroPlugin: CompilerPlugin {
  let providingMacros: [Macro.Type] = [
    ThreadSafeMacro.self,
    ThreadSafeInitializerMacro.self,
    ThreadSafePropertyMacro.self,
  ]
}
