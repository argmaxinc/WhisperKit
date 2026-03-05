//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import SwiftCompilerPlugin
import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros
import RegexBuilder

extension ClassDeclSyntax {
    private static let trailingCallSuffixRegex = Regex {
        "("
        ZeroOrMore {
            CharacterClass.anyOf(")").inverted
        }
        ")"
        Anchor.endOfLine
    }

    private static let integerLiteralRegex = Regex {
        Anchor.startOfLine
        Optionally { "-" }
        OneOrMore(.digit)
        Anchor.endOfLine
    }

    private static let doubleLiteralRegex = Regex {
        Anchor.startOfLine
        Optionally { "-" }
        OneOrMore(.digit)
        Optionally { "." }
        ZeroOrMore(.digit)
        Anchor.endOfLine
    }

    private static let quotedStringRegex = Regex {
        Anchor.startOfLine
        "\""
        ZeroOrMore {
            CharacterClass.anyOf("\n").inverted
        }
        "\""
        Anchor.endOfLine
    }

    /// Returns the list of mutable stored properties in the class.
    var storedVariables: [(name: String, type: String, defaultValue: String?)] {
        var storedVars = [(String, String, String?)]()

        for member in memberBlock.members {
            guard
                let varDecl = member.decl.as(VariableDeclSyntax.self),
                varDecl.isMutable
            else { continue }

            for binding in varDecl.bindings {
                if
                    binding.accessorBlock == nil,
                    let pattern = binding.pattern.as(IdentifierPatternSyntax.self)
                {
                    let name = pattern.identifier.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    let defaultValue = binding.initializer?.value.trimmedDescription
                        .trimmingCharacters(in: .whitespacesAndNewlines) ?? binding
                        .typeAnnotation?.type.defaultValueForOptional

                    if let typeAnnotation = binding.typeAnnotation {
                        let type = typeAnnotation.type.trimmedDescription.trimmingCharacters(in: .whitespacesAndNewlines)
                        storedVars.append((name, type, defaultValue))
                    } else if let defaultValue {
                        // Heuristically tries to infer the type from the default value
                        let value = stripTrailingCallSuffix(from: defaultValue)
                            .trimmingCharacters(in: .whitespacesAndNewlines)

                        let type: String =
                        if value == "true" || value == "false" {
                            "Bool"
                        } else if value.wholeMatch(of: Self.integerLiteralRegex) != nil {
                            "Int"
                        } else if value.wholeMatch(of: Self.doubleLiteralRegex) != nil {
                            "Double"
                        } else if value.wholeMatch(of: Self.quotedStringRegex) != nil {
                            "String"
                        } else {
                            value
                        }
                        storedVars.append((name, type, defaultValue))
                    }
                }
            }
        }
        return storedVars
    }

    private func stripTrailingCallSuffix(from value: String) -> String {
        guard let match = value.firstMatch(of: Self.trailingCallSuffixRegex) else {
            return value
        }
        return String(value[..<match.range.lowerBound])
    }
}
