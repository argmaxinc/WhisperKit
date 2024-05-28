import Foundation

//Compute the last row of the edit distance dynamic programming matrix
//between s1 and s2.
func computeLastRow(_ s1Chars: Array<Unicode.Scalar>, _ s2Chars: Array<Unicode.Scalar>) -> [Int] {
    
    var prevRow = Array(0...s2Chars.endIndex)
    
    for i in 1...s1Chars.endIndex {
        var currentRow = [Int](repeating: 0, count: s2Chars.endIndex + 1)
        currentRow[0] = i
        
        for j in 1...s2Chars.endIndex {
            let cost = s1Chars[i - 1] == s2Chars[j - 1] ? 0 : 1
            currentRow[j] = min(
                prevRow[j] + 1,     // Deletion
                currentRow[j - 1] + 1, // Insertion
                prevRow[j - 1] + cost  // Substitution
            )
        }
        prevRow = currentRow
    }
    
    return prevRow
}

func needlemanWunsch(_ xArray: Array<Unicode.Scalar>, _ yArray: Array<Unicode.Scalar>) -> [EditOp] {
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
                dp[i - 1][j] + 1,      // Deletion
                dp[i][j - 1] + 1,      // Insertion
                dp[i - 1][j - 1] + cost // Substitution
            )
        }
    }

    var i = m
    var j = n
    var ops = [EditOp]()

    while i > 0 && j > 0 {
        if dp[i][j] == dp[i - 1][j - 1] && xArray[i - 1] == yArray[j - 1] {
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


func hirschberg(_ s1: Array<Unicode.Scalar>, _ s2: Array<Unicode.Scalar>) -> [EditOp] {

    func hirschbergRec(_ x: Array<Unicode.Scalar>, _ y: Array<Unicode.Scalar>) -> [EditOp] {

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

    return hirschbergRec(s1, s2)
}
