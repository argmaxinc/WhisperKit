// Simple Fraction implementation for the normalization code.
// Doesn't do everything the python module fractions can do.
import Foundation

struct Fraction{
    var numerator: Int
    var denominator: Int
    
    var description: String {
        "\(numerator)/\(denominator)"
    }
    
    init?(numerator: Int, denominator: Int){
        guard denominator != 0 else { return nil }
        guard numerator > Int.min, denominator > Int.min else { return nil }
        
        self.numerator = numerator
        self.denominator = denominator
        if denominator < 0{
            self.numerator = -1 * self.numerator
            self.denominator = -1 * self.denominator
        }
        self.simplify()
    }
    
    init?(_ value: Double){
        if value == Double.infinity || value == Double.nan{
            return nil
        }
        if value == 0.0{
            self.init(numerator: 0, denominator: 1)
        }
        else if let (n,d) = Double.toIntegerNumberRatio(value: value){
            self.init(numerator: n, denominator: d)
        }else{
            return nil
        }
    }
    
    init?(_ value: Float){
        self.init(Double(value))
    }
    
    init?(_ value: String){
        let rationalFormatPattern = """
            \\A\\s*
            (?<sign>[-+]?)?
            (?=\\d|\\.\\d)
            (?<num>\\d*|\\d+(_\\d+)*)?
            (?:\\.(?<decimal>\\d+(_\\d+)*))?
            (?:\\s*/\\s*(?<denom>\\d+(_\\d+)*))?
            (?:E(?<exp>[-+]?\\d+(_\\d+)*))?
            \\s*\\Z
        """
        
        let regex = try? NSRegularExpression(pattern: rationalFormatPattern, options: [.allowCommentsAndWhitespace, .caseInsensitive])
        guard let regex = regex else { return nil}
        let range = NSRange(location: 0, length: value.utf16.count)
        var matches : [String:String] = [:]
        if let match = regex.firstMatch(in: value, options: [], range: range) {
            let groups = ["sign", "num", "denom", "decimal", "exp"]
            for group in groups {
                if let range = Range(match.range(withName: group), in: value) {
                    matches[group] = String(value[range])
                }
            }
        }
        if matches.count == 0{ return nil}
        // catch overflow if matches[num] will exceed size of Int64
        if matches["num"]?.count ?? 0 > 19{ return nil}
        var numerator = Int(matches["num"] ?? "0")!
        var denominator: Int
        
        if let denom = matches["denom"]{
            denominator = Int(denom)!
        }
        else{
            denominator = 1
            if var decimal = matches["decimal"]{
                decimal = decimal.replacingOccurrences(of: "_", with: "")
                let scale = Int(pow(Double(10), Double(decimal.count))) //10**len(decimal)
                guard let d = Int(decimal) else {return nil}
                numerator = numerator * scale + d
                denominator *= scale
            }
            
            if matches["exp"] != nil, let exponent = Int(matches["exp"]!){
                if exponent >= 0{
                    numerator *= Int(pow(Double(10), Double(exponent)))
                }else{
                    denominator *= Int(pow(Double(10), Double(-exponent)))
                }
            }
        }
        if matches["sign"] == "-"{
            numerator = -numerator
        }
        
        self.init(numerator: numerator, denominator: denominator)
    }
    
    static func gcd(lhs:Int,rhs:Int) -> Int{
        var a = lhs
        var b = rhs
        while b != 0 { (a, b) = (b, a % b) }
        return a
    }
    
    static func lcm(lhs:Int,rhs:Int) -> Int{
        return (lhs * rhs / gcd(lhs:lhs, rhs:rhs))
    }
    
    mutating func simplify(){
        var divisor = Fraction.gcd(lhs: numerator, rhs: denominator)
        if divisor < 0 { divisor *= -1 }
        self.numerator = Int(numerator / divisor)
        self.denominator = Int(denominator / divisor)
    }
    
    static func +(lhs: Fraction, rhs: Fraction) -> Fraction?{
        let na = lhs.numerator
        let nb = rhs.numerator
        let da = lhs.denominator
        let db = rhs.denominator
        let g = Fraction.gcd(lhs: da, rhs: db)
        if g == 1{
            return Fraction(numerator: na * db + da * nb, denominator: da * db)
        }
        let s = da / g
        let t = na * (db / g) + nb * s
        let g2 = Fraction.gcd(lhs: t, rhs: g)
        if g2 == 1{
            return Fraction(numerator: t, denominator: s * db)
        }
        return Fraction(numerator: t / g2, denominator: s * (db / g2))
    }
    
    static func -(lhs: Fraction, rhs: Fraction) -> Fraction?{
        let na = lhs.numerator
        let nb = rhs.numerator
        let da = lhs.denominator
        let db = rhs.denominator
        let g = Fraction.gcd(lhs: da, rhs: db)
        if g == 1{
            return Fraction(numerator: na * db - da * nb, denominator: da * db)
        }
        let s = da / g
        let t = na * (db / g) - nb * s
        let g2 = Fraction.gcd(lhs: t, rhs: g)
        if g2 == 1{
            return Fraction(numerator: t, denominator: s * db)
        }
        return Fraction(numerator: t / g2, denominator: s * (db / g2))
    }
    
    static func *(lhs: Fraction, rhs: Fraction) -> Fraction?{
        return Fraction(numerator:lhs.numerator * rhs.numerator, denominator:lhs.denominator * rhs.denominator)
    }
    
    static func /(lhs: Fraction, rhs: Fraction) -> Fraction?{
        return Fraction(numerator:lhs.numerator * rhs.denominator, denominator:lhs.denominator * rhs.numerator)
    }
    
}

extension Fraction: Equatable{
    static func == (lhs: Fraction, rhs: Fraction) -> Bool{
        if lhs.numerator == rhs.numerator, lhs.denominator == rhs.denominator{
            return true
        }
        return false
    }
}

// MARK: Fraction operations with Int's
extension Fraction{
    static func +(lhs: Int, rhs: Fraction) -> Fraction?{
        guard let lhsFraction = Fraction(numerator: lhs, denominator: 1) else {return rhs}
        return lhsFraction + rhs
    }
    
    static func +(lhs: Fraction, rhs: Int) -> Fraction?{
        guard let rhsFraction = Fraction(numerator: rhs, denominator: 1) else {return lhs}
        return  lhs + rhsFraction
    }
    
    static func -(lhs: Int, rhs: Fraction) -> Fraction?{
        guard let lhsFraction = Fraction(numerator: lhs, denominator: 1) else {return rhs}
        return lhsFraction - rhs
    }
    
    static func -(lhs: Fraction, rhs: Int) -> Fraction?{
        guard let rhsFraction = Fraction(numerator: rhs, denominator: 1) else {return lhs}
        return  lhs - rhsFraction
    }
    
    static func *(lhs: Fraction, rhs: Int) -> Fraction?{
        guard let rhsFraction = Fraction(numerator: rhs, denominator: 1) else {return lhs}
        return  lhs * rhsFraction
    }
    
    static func *(lhs: Int, rhs: Fraction) -> Fraction?{
        guard let lhsFraction = Fraction(numerator: lhs, denominator: 1) else {return rhs}
        return  lhsFraction * rhs
    }
    
    static func /(lhs: Fraction, rhs: Int) -> Fraction?{
        guard let rhsFraction = Fraction(numerator: rhs, denominator: 1) else {return lhs}
        return  lhs / rhsFraction
    }
    
    static func /(lhs: Int, rhs: Fraction) -> Fraction?{
        guard let lhsFraction = Fraction(numerator: lhs, denominator: 1) else {return rhs}
        return  lhsFraction / rhs
    }
}

extension Double{
    static func toIntegerNumberRatio(value: Double) -> (Int,Int)?{
        var floatPart: Double = value.significand
        var exponent: Int = value.exponent
        var numerator: Int
        var denominator: Int
        
        for _ in 0..<300 where floatPart != floatPart.rounded(.down){
            floatPart *= 2.0
            exponent -= 1
        }
        
        if floatPart == Double.infinity || floatPart == Double.nan{
            return nil
        }
        
        numerator = Int(floatPart.rounded(.down))
        denominator = 1
        
        if exponent > 0{
            numerator <<= exponent
        }
        else{
            denominator <<= -exponent
        }
        return (numerator, denominator)
    }
}



