//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftSyntaxMacros
import SwiftSyntax

public struct ThreadSafePropertyMacro: AccessorMacro {
    private enum Constants {
        static let internalStateName = "_internalState"
    }

    /// Adds a getter pointing to the internal state for each property marked with `@ThreadSafeProperty`.
    public static func expansion(
        of _: AttributeSyntax,
        providingAccessorsOf declaration: some DeclSyntaxProtocol,
        in _: some MacroExpansionContext)
    throws -> [AccessorDeclSyntax]
    {
        guard
            let property = declaration.as(VariableDeclSyntax.self),
            let identifier = property.bindings.first?.pattern.as(IdentifierPatternSyntax.self)?.identifier.text
        else {
            return []
        }

        return [
            AccessorDeclSyntax(stringLiteral: "get { \(Constants.internalStateName).value.\(identifier) }"),
            AccessorDeclSyntax(stringLiteral: "set { _ = \(Constants.internalStateName).set(\\.\(identifier), to: newValue) }"),
        ]
    }
}
