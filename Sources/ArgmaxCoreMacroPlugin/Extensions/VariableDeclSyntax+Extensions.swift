//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import SwiftSyntax

extension VariableDeclSyntax {
  var isMutable: Bool {
    guard
      bindingSpecifier.text == "var",
      attributes.isEmpty,
      bindings.count == 1,
      let binding = bindings.first,
      binding.accessorBlock == nil,
      let _ = binding.pattern.as(IdentifierPatternSyntax.self)
    else {
      return false
    }
    return true
  }
}
