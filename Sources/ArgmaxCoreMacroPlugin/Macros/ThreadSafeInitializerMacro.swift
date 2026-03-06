//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import RegexBuilder
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

public struct ThreadSafeInitializerMacro: BodyMacro {
    /// Rewrites the initializer to initialize the internal state with the stored properties.
    public static func expansion(
        of syntax: AttributeSyntax,
        providingBodyFor declaration: some DeclSyntaxProtocol & WithOptionalCodeBlockSyntax,
        in _: some MacroExpansionContext
    ) throws -> [CodeBlockItemSyntax] {
        guard
            let decl = declaration.body,
            let argument = syntax.arguments?.as(LabeledExprListSyntax.self)?.first
        else {
            return []
        }

        // Parse arguments
        guard let dictExpr = argument.expression.as(DictionaryExprSyntax.self)
        else {
            // Not a dictionary => do nothing
            return decl.statements.compactMap { CodeBlockItemSyntax($0) }
        }

        let elements = dictExpr.content.as(DictionaryElementListSyntax.self) ?? DictionaryElementListSyntax()

        let storedVariables: [(key: String, type: String, defaultValue: String?)] = elements.compactMap { element in
            // Parse the key
            guard
                let stringLiteral = element.key.as(StringLiteralExprSyntax.self),
                let firstSegment = stringLiteral.segments.first?.as(StringSegmentSyntax.self)
            else {
                return nil
            }

            let keyName = firstSegment.content.text

            // Parse the type
            guard
                let callExpr = element.value.as(FunctionCallExprSyntax.self),
                let genericType = callExpr.calledExpression.as(GenericSpecializationExprSyntax.self),
                let typeName = genericType.genericArgumentClause.arguments.first?.argument.trimmedDescription
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            else {
                return nil
            }

            // Parse the optional default value
            var defaultValue: String? = nil
            for arg in callExpr.arguments {
                if arg.label?.text == "value" {
                    defaultValue = arg.expression.trimmedDescription
                }
            }
            if defaultValue == nil, typeName.hasSuffix("?") { defaultValue = "nil" }

            return (key: keyName, type: typeName, defaultValue: defaultValue)
        }

        // Find when the last stored variable is set
        let lastVariableSetAt = decl.statements.enumerated().compactMap { offset, statement in
            for (key, _, _) in storedVariables.filter({ $0.defaultValue == nil }) {
                let trimmedStatement = statement.trimmedDescription.trimmingCharacters(in: .whitespacesAndNewlines)
                if
                    trimmedStatement.contains(selfDotPropertyEqual(key, isAtStart: true)) ||
                        trimmedStatement.contains(propertyEqual(key, isAtStart: true))
                {
                    return (offset: offset, element: key)
                }
            }
            return nil
        }.last?.offset ?? -1

        // Replace foo = ... by _foo = ... for all stored properties
        var mutatedProperties = Set<String>()
        var statements: [CodeBlockItemSyntax?] = decl.statements.enumerated().flatMap { offset, statement -> [CodeBlockItemSyntax?] in
            if offset > lastVariableSetAt {
                return [CodeBlockItemSyntax(statement)]
            }
            let trimmedStatement = statement.trimmedDescription
            for (key, _, _) in storedVariables {
                if trimmedStatement.contains(selfDotPropertyEqual(key, isAtStart: true)) {
                    mutatedProperties.insert(key)
                    return [
                        CodeBlockItemSyntax(stringLiteral: statement.trimmedDescription.replacing(
                            selfDotPropertyEqual(key),
                            with: "_\(key) =")),
                    ]
                }
                if trimmedStatement.contains(propertyEqual(key, isAtStart: true)) {
                    mutatedProperties.insert(key)
                    return [
                        CodeBlockItemSyntax(stringLiteral: statement.trimmedDescription.replacing(propertyEqual(key), with: "_\(key) =")),
                    ]
                }
            }
            return [CodeBlockItemSyntax(statement)]
        }

        // Set _internalState once the required properties have been set
        let addedStatement = CodeBlockItemSyntax(
            stringLiteral: "self._internalState = Mutex<_InternalState>(_InternalState(\(storedVariables.map { "\($0.key): _\($0.key)" }.joined(separator: ", "))))")
        statements.insert(addedStatement, at: lastVariableSetAt + 1)

        // Add variables to hold the properties while they are created
        for (key, type, defaultValue) in storedVariables.reversed() {
            let isMutated = mutatedProperties.contains(key)
            if let defaultValue {
                statements.insert(
                    CodeBlockItemSyntax(stringLiteral: "\(isMutated ? "var" : "let") _\(key): \(type) = \(defaultValue)"),
                    at: 0)
            } else {
                statements.insert(CodeBlockItemSyntax(stringLiteral: "\(isMutated ? "var" : "let") _\(key): \(type)"), at: 0)
            }
        }

        return statements.compactMap(\.self)
    }

    /// Regex to match "self.key = " with flexible whitespace
    /// Handles standard spacing, multiple spaces, tabs, etc.
    private static func selfDotPropertyEqual(
        _ propertyName: String,
        isAtStart: Bool = false
    ) -> Regex<Regex<Substring>.RegexOutput>
    {
        if isAtStart {
            Regex {
                Anchor.startOfLine
                "self."
                propertyName
                OneOrMore(.whitespace)
                "="
            }
        } else {
            Regex {
                "self."
                propertyName
                OneOrMore(.whitespace)
                "="
            }
        }
    }

    /// Regex to match "key = " with flexible whitespace
    /// Handles standard spacing, multiple spaces, tabs, etc.
    private static func propertyEqual(_ propertyName: String, isAtStart: Bool = false) -> Regex<Regex<Substring>.RegexOutput> {
        if isAtStart {
            Regex {
                Anchor.startOfLine
                propertyName
                OneOrMore(.whitespace)
                "="
            }
        } else {
            Regex {
                propertyName
                OneOrMore(.whitespace)
                "="
            }
        }
    }
}
