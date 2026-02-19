//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

actor IntValueCollector {
    private var storage: [Int] = []

    var values: [Int] {
        storage
    }

    var count: Int {
        storage.count
    }

    func append(_ value: Int) {
        storage.append(value)
    }
}
