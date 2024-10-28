//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// Compute the last row of the edit distance dynamic programming matrix
/// between s1 and s2.
func computeLastRow(_ s1Chars: [Unicode.Scalar], _ s2Chars: [Unicode.Scalar]) -> [Int] {
    var prevRow = Array(0...s2Chars.endIndex)

    for i in 1...s1Chars.endIndex {
        var currentRow = [Int](repeating: 0, count: s2Chars.endIndex + 1)
        currentRow[0] = i

        for j in 1...s2Chars.endIndex {
            let cost = s1Chars[i - 1] == s2Chars[j - 1] ? 0 : 1
            currentRow[j] = min(
                prevRow[j] + 1, // Deletion
                currentRow[j - 1] + 1, // Insertion
                prevRow[j - 1] + cost // Substitution
            )
        }
        prevRow = currentRow
    }

    return prevRow
}

func needlemanWunsch(_ xArray: [Unicode.Scalar], _ yArray: [Unicode.Scalar]) -> [EditOp] {
    let m = xArray.count
    let n = yArray.count

    var dp = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)
    for i in 1...m {
        dp[i][0] = i
    }
    for j in 1...n {
        dp[0][j] = j
    }

    for i in 1...m {
        for j in 1...n {
            let cost = xArray[i - 1] == yArray[j - 1] ? 0 : 1
            dp[i][j] = min(
                dp[i - 1][j] + 1, // Deletion
                dp[i][j - 1] + 1, // Insertion
                dp[i - 1][j - 1] + cost // Substitution
            )
        }
    }

    var i = m
    var j = n
    var ops = [EditOp]()

    while i > 0, j > 0 {
        if dp[i][j] == dp[i - 1][j - 1], xArray[i - 1] == yArray[j - 1] {
            // Match operation is omitted
            i -= 1
            j -= 1
        } else if dp[i][j] == dp[i - 1][j - 1] + 1 {
            ops.append(EditOp.replace) // Substitution
            i -= 1
            j -= 1
        } else if dp[i][j] == dp[i][j - 1] + 1 {
            ops.append(EditOp.insert) // Insertion
            j -= 1
        } else {
            ops.append(EditOp.delete) // Deletion
            i -= 1
        }
    }

    while i > 0 {
        ops.append(EditOp.delete)
        i -= 1
    }
    while j > 0 {
        ops.append(EditOp.insert)
        j -= 1
    }

    return ops.reversed()
}

func hirschberg(_ reference: [Unicode.Scalar], _ s2: [Unicode.Scalar]) -> [EditOp] {
    func hirschbergRec(_ x: [Unicode.Scalar], _ y: [Unicode.Scalar]) -> [EditOp] {
        let m = x.endIndex
        let n = y.endIndex

        if m == 0 {
            let result = y.map { _ in EditOp.insert }
            return result
        }
        if n == 0 {
            let result = x.map { _ in EditOp.delete }
            return result
        }
        if m == 1 || n == 1 {
            let result = needlemanWunsch(x, y)
            return result
        }

        let i = m / 2
        let xPrefix = Array(x[x.startIndex..<i])
        let xSuffix = Array(x[i...])
        let scoreL = computeLastRow(xPrefix, y)
        let scoreR = computeLastRow(Array(xSuffix.reversed()), Array(y.reversed()))

        var k = 0
        var minCost = Int.max
        for j in 0..<scoreL.count {
            let cost = scoreL[j] + scoreR[scoreR.count - 1 - j]
            if cost < minCost {
                minCost = cost
                k = j
            }
        }

        let result = hirschbergRec(Array(x[..<i]), Array(y[..<k])) +
            hirschbergRec(Array(x[i...]), Array(y[k...]))

        return result
    }

    return hirschbergRec(reference, s2)
}

/// Calculates the Levenshtein distance between two strings and returns the edit operations.
///
/// - Parameters:
///   - sourceText: The first input string as an array of Unicode.Scalar.
///   - targetText: The second input string as an array of Unicode.Scalar.
/// - Returns: An array of EditOp representing the sequence of edit operations.
func levenshtein(_ sourceText: [Unicode.Scalar], _ targetText: [Unicode.Scalar]) -> [EditOp] {
    let n = sourceText.count
    let m = targetText.count
    let maxD = n + m
    let vSize = 2 * maxD + 1
    var v = [Int](repeating: 0, count: vSize)
    var trace = [[Int]]()

    let offset = maxD

    for d in 0...maxD {
        let vSnapshot = v
        for k in stride(from: -d, through: d, by: 2) {
            let kIndex = k + offset
            var x: Int
            if k == -d || (k != d && v[kIndex - 1] < v[kIndex + 1]) {
                x = v[kIndex + 1]
            } else {
                x = v[kIndex - 1] + 1
            }
            var y = x - k
            while x < n, y < m, sourceText[x] == targetText[y] {
                x += 1
                y += 1
            }
            v[kIndex] = x
            if x >= n, y >= m {
                trace.append(vSnapshot)
                return backtrack(trace: trace, sourceText: sourceText, targetText: targetText)
            }
        }
        trace.append(vSnapshot)
    }
    return []
}

func backtrack(trace: [[Int]], sourceText: [Unicode.Scalar], targetText: [Unicode.Scalar]) -> [EditOp] {
    var editOps = [EditOp]()
    let n = sourceText.count
    let m = targetText.count
    let offset = trace[0].count / 2
    var x = n
    var y = m

    for d in stride(from: trace.count - 1, through: 0, by: -1) {
        let v = trace[d]
        let k = x - y
        let kIndex = k + offset

        var prevK: Int
        if k == -d || (k != d && v[kIndex - 1] < v[kIndex + 1]) {
            prevK = k + 1
        } else {
            prevK = k - 1
        }
        let prevX = v[prevK + offset]
        let prevY = prevX - prevK

        while x > prevX, y > prevY {
            // Match or Replace
            if sourceText[x - 1] == targetText[y - 1] {
                editOps.append(.blank)
            } else {
                editOps.append(.replace)
            }
            x -= 1
            y -= 1
        }

        if d > 0 {
            if x == prevX {
                // Insertion
                editOps.append(.insert)
                y -= 1
            } else {
                // Deletion
                editOps.append(.delete)
                x -= 1
            }
        }
    }

    return editOps.reversed()
}
