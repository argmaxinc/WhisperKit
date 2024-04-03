import Foundation

// MARK: - RegressionStats
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

// MARK: - TestInfo
class TestInfo: JSONCodable {
    let device, audioFile: String
    let model: String
    let date: String
    let timeElapsedInSeconds: TimeInterval
    
    init(device: String, audioFile: String, model: String, date: String, timeElapsedInSeconds: TimeInterval) {
        self.device = device
        self.audioFile = audioFile
        self.model = model
        self.date = date
        self.timeElapsedInSeconds = timeElapsedInSeconds
    }
}


// MARK: - Stats
class Stats: JSONCodable {
    var measurements: [Measurement]
    let units: String
    var totalNumberOfMeasurements: Int
    
    init(measurements: [Measurement], units: String, totalNumberOfMeasurements: Int) {
        self.measurements = measurements
        self.units = units
        self.totalNumberOfMeasurements = totalNumberOfMeasurements
    }
    
    func measure(from values: [Float], timeElapsed: TimeInterval){
        var measurement: Measurement
        if let min = values.min(),let max = values.max(){
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

// MARK: - Stats
class LatencyStats: Stats{
    override init(measurements: [Measurement] = [], units: String, totalNumberOfMeasurements: Int = 0) {
        super.init(measurements: measurements, units: units, totalNumberOfMeasurements: totalNumberOfMeasurements)
    }
    
    required init(from decoder: any Decoder) throws {
        fatalError("init(from:) has not been implemented")
    }
    
}

class MemoryStats: Stats{
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
}

struct Measurement: JSONCodable{
    let min, max, average: Float
    let numberOfMeasurements: Int
    let timeElapsed: TimeInterval
}

protocol JSONCodable: Codable {
}
extension JSONCodable{
    func jsonData() throws -> Data {
        return try JSONEncoder().encode(self)
    }
}

extension Data {
    var prettyPrintedJSONString: NSString? { /// NSString gives us a nice sanitized debugDescription
        guard let object = try? JSONSerialization.jsonObject(with: self, options: []),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
              let prettyPrintedString = NSString(data: data, encoding: String.Encoding.utf8.rawValue) else { return nil }

        return prettyPrintedString
    }
}

// MARK: - Memory Footprint
func memoryFootprint() -> UInt64 {
    // The `TASK_VM_INFO_COUNT` and `TASK_VM_INFO_REV1_COUNT` macros are too
    // complex for the Swift C importer, so we have to define them ourselves.
    let TASK_VM_INFO_COUNT = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    guard let offset = MemoryLayout.offset(of: \task_vm_info_data_t.min_address) else {return 0}
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
    else { return 0}
    
    let usedBytes = Float(info.phys_footprint)
    let usedBytesInt: UInt64 = UInt64(usedBytes)
    let usedMB = usedBytesInt / 1024 / 1024
    return usedMB
}

func writeToFile(text: String, append: Bool = false, fileName: String) throws {
    // File URL where you want to write the output
    var fileURL = URL(fileURLWithPath: "./\(fileName)")
    var dataToWrite = text.data(using: .utf8)
    if FileManager.default.fileExists(atPath: fileURL.path) {
        if let fileHandle = try? FileHandle(forWritingTo: fileURL) {
            if append{fileHandle.seekToEndOfFile()}
            if let data = dataToWrite{
                fileHandle.write(data)
            }
            fileHandle.closeFile()
        }
    }else {
        try dataToWrite?.write(to: fileURL)
    }
}
