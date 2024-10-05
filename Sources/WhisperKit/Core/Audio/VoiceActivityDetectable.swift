//
//  VoiceActivityDetectable.swift
//  whisperkit
//
//  Created by Norikazu Muramoto on 2024/10/03.
//

/// Protocol defining the interface for Voice Activity Detection (VAD)
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public protocol VoiceActivityDetectable: Sendable {
    var sampleRate: Int { get }
    var frameLengthSamples: Int { get }
    var frameOverlapSamples: Int { get }
    
    func voiceActivity(in waveform: [Float]) -> [Bool]
    func calculateActiveChunks(in waveform: [Float]) -> [(startIndex: Int, endIndex: Int)]
    func voiceActivityIndexToAudioSampleIndex(_ index: Int) -> Int
    func voiceActivityIndexToSeconds(_ index: Int) -> Float
    func findLongestSilence(in vadResult: [Bool]) -> (startIndex: Int, endIndex: Int)?
    func voiceActivityClipTimestamps(in waveform: [Float]) -> [Float]
    func calculateNonSilentSeekClips(in waveform: [Float]) -> [(start: Int, end: Int)]
    func calculateSeekTimestamps(in waveform: [Float]) -> [(startTime: Float, endTime: Float)]
}

extension VoiceActivityDetectable {
    
    public func calculateActiveChunks(in waveform: [Float]) -> [(startIndex: Int, endIndex: Int)] {
        let vad = voiceActivity(in: waveform)
        var result = [(startIndex: Int, endIndex: Int)]()
        var currentStartIndex: Int?

        for (index, vadChunk) in vad.enumerated() {
            if vadChunk {
                let chunkStart = index * frameLengthSamples
                let chunkEnd = min(chunkStart + frameLengthSamples, waveform.count)

                if currentStartIndex != nil {
                    result[result.count - 1].endIndex = chunkEnd
                } else {
                    currentStartIndex = chunkStart
                    result.append((startIndex: chunkStart, endIndex: chunkEnd))
                }
            } else {
                currentStartIndex = nil
            }
        }

        return result
    }
    
    public func voiceActivityIndexToAudioSampleIndex(_ index: Int) -> Int {
        return index * frameLengthSamples
    }

    public func voiceActivityIndexToSeconds(_ index: Int) -> Float {
        return Float(voiceActivityIndexToAudioSampleIndex(index)) / Float(sampleRate)
    }

    public func findLongestSilence(in vadResult: [Bool]) -> (startIndex: Int, endIndex: Int)? {
        var longestStartIndex: Int?
        var longestEndIndex: Int?
        var longestCount = 0
        var index = 0
        while index < vadResult.count {
            if vadResult[index] {
                index += 1
            } else {
                var endIndex = index
                while endIndex < vadResult.count, !vadResult[endIndex] {
                    endIndex += 1
                }
                let count = endIndex - index
                if count > longestCount {
                    longestCount = count
                    longestStartIndex = index
                    longestEndIndex = endIndex
                }
                index = endIndex
            }
        }
        if let longestStartIndex, let longestEndIndex {
            return (startIndex: longestStartIndex, endIndex: longestEndIndex)
        } else {
            return nil
        }
    }
    
    // MARK - Utility

    public func voiceActivityClipTimestamps(in waveform: [Float]) -> [Float] {
        let nonSilentChunks = calculateActiveChunks(in: waveform)
        var clipTimestamps = [Float]()

        for chunk in nonSilentChunks {
            let startTimestamp = Float(chunk.startIndex) / Float(sampleRate)
            let endTimestamp = Float(chunk.endIndex) / Float(sampleRate)

            clipTimestamps.append(contentsOf: [startTimestamp, endTimestamp])
        }

        return clipTimestamps
    }

    public func calculateNonSilentSeekClips(in waveform: [Float]) -> [(start: Int, end: Int)] {
        let clipTimestamps = voiceActivityClipTimestamps(in: waveform)
        let options = DecodingOptions(clipTimestamps: clipTimestamps)
        let seekClips = prepareSeekClips(contentFrames: waveform.count, decodeOptions: options)
        return seekClips
    }

    public func calculateSeekTimestamps(in waveform: [Float]) -> [(startTime: Float, endTime: Float)] {
        let nonSilentChunks = calculateActiveChunks(in: waveform)
        var seekTimestamps = [(startTime: Float, endTime: Float)]()

        for chunk in nonSilentChunks {
            let startTimestamp = Float(chunk.startIndex) / Float(sampleRate)
            let endTimestamp = Float(chunk.endIndex) / Float(sampleRate)

            seekTimestamps.append(contentsOf: [(startTime: startTimestamp, endTime: endTimestamp)])
        }

        return seekTimestamps
    }
}
