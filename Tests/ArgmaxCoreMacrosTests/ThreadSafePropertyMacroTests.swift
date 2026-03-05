//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCoreMacroPlugin
import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

final class ThreadSafePropertyMacroTests: XCTestCase {
    private let macros: [String: Macro.Type] = [
        "ThreadSafeProperty": ThreadSafePropertyMacro.self,
    ]

    func testExpandsToGetterAndSetterUsingInternalState() {
        assertMacroExpansion(
            """
            class Holder {
                @ThreadSafeProperty
                var name: String = "default"
            }
            """,
            expandedSource: """
            class Holder {
                var name: String {
                    get {
                        _internalState.value.name
                    }
                    set {
                        _ = _internalState.set(\\.name, to: newValue)
                    }
                }
            }
            """,
            macros: macros
        )
    }

    func testSetterUsesKeyPathMutationAPI() {
        assertMacroExpansion(
            """
            class Counter {
                @ThreadSafeProperty
                var count: Int = 0
            }
            """,
            expandedSource: """
            class Counter {
                var count: Int {
                    get {
                        _internalState.value.count
                    }
                    set {
                        _ = _internalState.set(\\.count, to: newValue)
                    }
                }
            }
            """,
            macros: macros
        )
    }

    func testNonIdentifierPatternProducesNoAccessors() {
        assertMacroExpansion(
            """
            class PairHolder {
                @ThreadSafeProperty
                var (x, y) = (0, 0)
            }
            """,
            expandedSource: """
            class PairHolder {
                var (x, y) = (0, 0)
            }
            """,
            macros: macros
        )
    }
}
