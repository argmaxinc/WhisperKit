import Foundation
import WhisperKit

// MARK: RegressionStats

class RegressionStats: JSONCodable {
    let testInfo: TestInfo
    let memoryStats: MemoryStats
    let latencyStats: LatencyStats

    init(testInfo: TestInfo, memoryStats: MemoryStats, latencyStats: LatencyStats) {
        self.testInfo = testInfo
        self.memoryStats = memoryStats
        self.latencyStats = latencyStats
    }

    func jsonData() throws -> Data {
        return try JSONEncoder().encode(self)
    }
}

// MARK: TestInfo

class TestInfo: JSONCodable {
    let device, audioFile: String
    let model: String
    let date: String
    let timeElapsedInSeconds: TimeInterval
    let timings: TranscriptionTimings?
    let transcript: String?

    init(device: String, audioFile: String, model: String, date: String, timeElapsedInSeconds: TimeInterval, timings: TranscriptionTimings?, transcript: String?) {
        self.device = device
        self.audioFile = audioFile
        self.model = model
        self.date = date
        self.timeElapsedInSeconds = timeElapsedInSeconds
        self.timings = timings
        self.transcript = transcript
    }
}

// MARK: TestReport

struct TestReport: JSONCodable {
    let device: String
    let modelsTested: [String]
    let failureInfo: [String: String]

    init(device: String, modelsTested: [String], failureInfo: [String: String]) {
        self.device = device
        self.modelsTested = modelsTested
        self.failureInfo = failureInfo
    }
}

// MARK: Stats

class Stats: JSONCodable {
    var measurements: [Measurement]
    let units: String
    var totalNumberOfMeasurements: Int

    init(measurements: [Measurement], units: String, totalNumberOfMeasurements: Int) {
        self.measurements = measurements
        self.units = units
        self.totalNumberOfMeasurements = totalNumberOfMeasurements
    }

    func measure(from values: [Float], timeElapsed: TimeInterval) {
        var measurement: Measurement
        if let min = values.min(), let max = values.max() {
            measurement = Measurement(
                min: min,
                max: max,
                average: values.reduce(0,+) / Float(values.count),
                numberOfMeasurements: values.count,
                timeElapsed: timeElapsed
            )
            self.measurements.append(measurement)
            self.totalNumberOfMeasurements += values.count
        }
    }
}

// MARK: LatencyStats

class LatencyStats: Stats {
    override init(measurements: [Measurement] = [], units: String, totalNumberOfMeasurements: Int = 0) {
        super.init(measurements: measurements, units: units, totalNumberOfMeasurements: totalNumberOfMeasurements)
    }

    required init(from decoder: any Decoder) throws {
        fatalError("init(from:) has not been implemented")
    }

    func calculate(from total: Double, runs: Int) -> Double {
        return runs > 0 ? total / Double(runs) : -1
    }
}

class MemoryStats: Stats {
    var preTranscribeMemory: Float
    var postTranscribeMemory: Float

    init(measurements: [Measurement] = [], units: String, totalNumberOfMeasurements: Int = 0, preTranscribeMemory: Float, postTranscribeMemory: Float) {
        self.preTranscribeMemory = preTranscribeMemory
        self.postTranscribeMemory = postTranscribeMemory
        super.init(measurements: measurements, units: units, totalNumberOfMeasurements: totalNumberOfMeasurements)
    }

    required init(from decoder: any Decoder) throws {
        fatalError("init(from:) has not been implemented")
    }

    /// Implement the encode(to:) method
    override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try super.encode(to: encoder)
        try container.encode(preTranscribeMemory, forKey: .preTranscribeMemory)
        try container.encode(postTranscribeMemory, forKey: .postTranscribeMemory)
    }

    /// Coding keys for MemoryStats properties
    enum CodingKeys: String, CodingKey {
        case preTranscribeMemory
        case postTranscribeMemory
    }
}

struct Measurement: JSONCodable {
    let min, max, average: Float
    let numberOfMeasurements: Int
    let timeElapsed: TimeInterval
}

protocol JSONCodable: Codable {}

extension JSONCodable {
    func jsonData() throws -> Data {
        return try JSONEncoder().encode(self)
    }
}

extension Data {
    var prettyPrintedJSONString: NSString? { // NSString gives us a nice sanitized debugDescription
        guard let object = try? JSONSerialization.jsonObject(with: self, options: []),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
              let prettyPrintedString = NSString(data: data, encoding: String.Encoding.utf8.rawValue) else { return nil }

        return prettyPrintedString
    }
}

// MARK: - SystemMemoryChecker

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
class SystemMemoryChecker: NSObject {
    static func getMemoryUsed() -> UInt64 {
        // The `TASK_VM_INFO_COUNT` and `TASK_VM_INFO_REV1_COUNT` macros are too
        // complex for the Swift C importer, so we have to define them ourselves.
        let TASK_VM_INFO_COUNT = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        guard let offset = MemoryLayout.offset(of: \task_vm_info_data_t.min_address) else { return 0 }
        let TASK_VM_INFO_REV1_COUNT = mach_msg_type_number_t(offset / MemoryLayout<integer_t>.size)
        var info = task_vm_info_data_t()
        var count = TASK_VM_INFO_COUNT
        let kr = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard
            kr == KERN_SUCCESS,
            count >= TASK_VM_INFO_REV1_COUNT
        else { return 0 }

        let usedBytes = Float(info.phys_footprint)
        let usedBytesInt = UInt64(usedBytes)
        let usedMB = usedBytesInt / 1024 / 1024
        return usedMB
    }
}
