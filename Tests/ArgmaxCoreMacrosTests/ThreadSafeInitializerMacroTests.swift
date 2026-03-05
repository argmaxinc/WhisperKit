//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCoreMacroPlugin
import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

final class ThreadSafeInitializerMacroTests: XCTestCase {
    private let macros: [String: Macro.Type] = [
        "ThreadSafeInitializer": ThreadSafeInitializerMacro.self,
    ]

    func testRewritesSelfDotAssignmentsToTemporaryBackingVars() {
        assertMacroExpansion(
            """
            class Counter {
                @ThreadSafeInitializer(["count": TypeErased<Int>()])
                init(count: Int) {
                    self.count = count
                }
            }
            """,
            expandedSource: """
            class Counter {
                init(count: Int) {
                    var _count: Int
                    _count = count
                    self._internalState = Mutex<_InternalState>(_InternalState(count: _count))
                }
            }
            """,
            macros: macros
        )
    }

    func testRewritesBareAssignmentsToTemporaryBackingVars() {
        assertMacroExpansion(
            """
            class NameHolder {
                @ThreadSafeInitializer(["name": TypeErased<String>()])
                init(raw: String) {
                    name = raw
                }
            }
            """,
            expandedSource: """
            class NameHolder {
                init(raw: String) {
                    var _name: String
                    _name = raw
                    self._internalState = Mutex<_InternalState>(_InternalState(name: _name))
                }
            }
            """,
            macros: macros
        )
    }

    func testInsertsInternalStateAssignmentAfterLastRequiredPropertySet() {
        assertMacroExpansion(
            """
            class SequencedInit {
                @ThreadSafeInitializer([
                    "a": TypeErased<Int>(),
                    "b": TypeErased<String>()
                ])
                init(a: Int, b: String) {
                    self.a = a
                    log()
                    b = normalize(b)
                    finish()
                }
            }
            """,
            expandedSource: """
            class SequencedInit {
                init(a: Int, b: String) {
                    var _a: Int
                    var _b: String
                    _a = a
                    log()
                    _b = normalize(b)
                    self._internalState = Mutex<_InternalState>(_InternalState(a: _a, b: _b))
                    finish()
                }
            }
            """,
            macros: macros
        )
    }

    func testCreatesVarTemporariesForMutatedPropertiesAndLetForUnmutated() {
        assertMacroExpansion(
            """
            class MixedTemporaries {
                @ThreadSafeInitializer([
                    "count": TypeErased<Int>(),
                    "label": TypeErased<String>(value: "idle")
                ])
                init(count: Int) {
                    self.count = count
                }
            }
            """,
            expandedSource: """
            class MixedTemporaries {
                init(count: Int) {
                    var _count: Int
                    let _label: String = "idle"
                    _count = count
                    self._internalState = Mutex<_InternalState>(_InternalState(count: _count, label: _label))
                }
            }
            """,
            macros: macros
        )
    }

    func testUsesValueLabelDefaultsFromTypeErasedArguments() {
        assertMacroExpansion(
            """
            class DefaultFromValueLabel {
                @ThreadSafeInitializer(["enabled": TypeErased<Bool>(value: true)])
                init() {
                    trace()
                }
            }
            """,
            expandedSource: """
            class DefaultFromValueLabel {
                init() {
                    let _enabled: Bool = true
                    self._internalState = Mutex<_InternalState>(_InternalState(enabled: _enabled))
                    trace()
                }
            }
            """,
            macros: macros
        )
    }

    func testOptionalTypeWithoutValueDefaultsToNil() {
        assertMacroExpansion(
            """
            class OptionalDefault {
                @ThreadSafeInitializer(["nickname": TypeErased<String?>()])
                init() {
                    trace()
                }
            }
            """,
            expandedSource: """
            class OptionalDefault {
                init() {
                    let _nickname: String? = nil
                    self._internalState = Mutex<_InternalState>(_InternalState(nickname: _nickname))
                    trace()
                }
            }
            """,
            macros: macros
        )
    }

    func testNonDictionaryMacroArgumentLeavesBodyUnchanged() {
        assertMacroExpansion(
            """
            class NonDictionaryArgument {
                @ThreadSafeInitializer(1)
                init() {
                    trace()
                }
            }
            """,
            expandedSource: """
            class NonDictionaryArgument {
                init() {
                    trace()
                }
            }
            """,
            macros: macros
        )
    }

}
