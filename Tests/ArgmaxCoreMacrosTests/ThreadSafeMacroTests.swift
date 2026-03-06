//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCoreMacroPlugin
import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

final class ThreadSafeMacroTests: XCTestCase {
    private let macros: [String: Macro.Type] = [
        "ThreadSafe": ThreadSafeMacro.self,
    ]

    func testClassWithoutInitializerAddsInitializedInternalStateStructAndInLock() {
        assertMacroExpansion(
            """
            @ThreadSafe
            class Counter {
                var count: Int = 0
            }
            """,
            expandedSource: """
            class Counter {
                @ThreadSafeProperty
                var count: Int = 0

                private let _internalState = Mutex<_InternalState>(_InternalState(count: 0))

                private struct _InternalState: Sendable {
                    var count: Int
                }

                @discardableResult
                    private func inLock<Result: Sendable>(_ mutation: @Sendable (inout _InternalState) -> Result) -> Result {
                        _internalState.mutate(mutation)
                    }
            }
            """,
            macros: macros
        )
    }

    func testClassWithInitializerAddsDeferredInternalStateAndNoAutoInit() {
        assertMacroExpansion(
            """
            @ThreadSafe
            class Counter {
                init() {}
            }
            """,
            expandedSource: """
            class Counter {
                @ThreadSafeInitializer([:])
                init() {}

                private let _internalState: Mutex<_InternalState>

                private struct _InternalState: Sendable {
                }

                @discardableResult
                    private func inLock<Result: Sendable>(_ mutation: @Sendable (inout _InternalState) -> Result) -> Result {
                        _internalState.mutate(mutation)
                    }
            }
            """,
            macros: macros
        )
    }

    func testAddsThreadSafePropertyOnlyToEligibleStoredVars() {
        assertMacroExpansion(
            """
            @ThreadSafe
            class Sample {
                var tracked: Int = 1
                let constant: Int = 2
                @available(*, deprecated)
                var attributed: Int = 3
                var computed: Int { 4 }
                var first = 5, second = 6
            }
            """,
            expandedSource: """
            class Sample {
                @ThreadSafeProperty
                var tracked: Int = 1
                let constant: Int = 2
                @available(*, deprecated)
                var attributed: Int = 3
                var computed: Int { 4 }
                var first = 5, second = 6

                private let _internalState = Mutex<_InternalState>(_InternalState(tracked: 1))

                private struct _InternalState: Sendable {
                    var tracked: Int
                }

                @discardableResult
                    private func inLock<Result: Sendable>(_ mutation: @Sendable (inout _InternalState) -> Result) -> Result {
                        _internalState.mutate(mutation)
                    }
            }
            """,
            macros: macros
        )
    }

    func testDoesNotDuplicateExistingThreadSafePropertyAttribute() {
        assertMacroExpansion(
            """
            @ThreadSafe
            class AlreadyWrapped {
                @ThreadSafeProperty
                var count: Int = 0
            }
            """,
            expandedSource: """
            class AlreadyWrapped {
                @ThreadSafeProperty
                var count: Int = 0

                private let _internalState = Mutex<_InternalState>(_InternalState())

                private struct _InternalState: Sendable {
                }

                @discardableResult
                    private func inLock<Result: Sendable>(_ mutation: @Sendable (inout _InternalState) -> Result) -> Result {
                        _internalState.mutate(mutation)
                    }
            }
            """,
            macros: macros
        )
    }

    func testAddsThreadSafeInitializerOnlyToNonConvenienceInitializers() {
        assertMacroExpansion(
            """
            @ThreadSafe
            class InitKinds {
                init() {}

                convenience init(flag: Bool) {
                    self.init()
                }
            }
            """,
            expandedSource: """
            class InitKinds {
                @ThreadSafeInitializer([:])
                init() {}

                convenience init(flag: Bool) {
                    self.init()
                }

                private let _internalState: Mutex<_InternalState>

                private struct _InternalState: Sendable {
                }

                @discardableResult
                    private func inLock<Result: Sendable>(_ mutation: @Sendable (inout _InternalState) -> Result) -> Result {
                        _internalState.mutate(mutation)
                    }
            }
            """,
            macros: macros
        )
    }

    func testMissingDefaultWithoutInitializerEmitsDiagnostic() {
        assertMacroExpansion(
            """
            @ThreadSafe
            class MissingDefault {
                var count: Int
            }
            """,
            expandedSource: """
            class MissingDefault {
                @ThreadSafeProperty
                var count: Int
            }
            """,
            diagnostics: [
                DiagnosticSpec(
                    message: "Property 'count' must have a default value or the class must define an initializer.",
                    line: 1,
                    column: 1,
                    severity: .error
                ),
            ],
            macros: macros
        )
    }

    func testNonClassDeclarationProducesNoExpansion() {
        assertMacroExpansion(
            """
            @ThreadSafe
            enum State {
                case idle
            }
            """,
            expandedSource: """
            enum State {
                case idle
            }
            """,
            macros: macros
        )
    }
}
