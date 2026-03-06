//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftSyntax

extension TypeSyntax {
  var defaultValueForOptional: String? {
    if self.as(OptionalTypeSyntax.self) != nil {
      return "nil"
    }
    return nil
  }
}
