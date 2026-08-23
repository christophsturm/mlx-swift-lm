// Copyright © 2026 Apple Inc.

import Foundation
import MLX

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

#if canImport(AVFoundation)
extension UserInput.AudioFormat {
    public var asAudioFormatID: AudioFormatID {
        switch self {
        case .linearPCM:
            return kAudioFormatLinearPCM
        }
    }
}
#endif

extension UserInput.Audio {

    public func asMLXArray(processing: UserInput.AudioProcessing = .init()) async throws -> MLXArray
    {
        switch self {
        case .url(let url):
            #if canImport(AVFoundation)
            let asset = AVURLAsset(url: url)

            guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
                throw UserInputError.noAudioData(url)
            }

            let settings: [String: Any] = [
                AVFormatIDKey: processing.audioFormat.asAudioFormatID,
                AVSampleRateKey: processing.sampleRate,
                AVNumberOfChannelsKey: processing.channels,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsNonInterleaved: false,
            ]

            let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
            let reader = try AVAssetReader(asset: asset)
            reader.add(output)
            reader.startReading()

            var samples: [Float] = []

            while let sampleBuffer = output.copyNextSampleBuffer() {
                guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { continue }

                let byteCount = CMBlockBufferGetDataLength(blockBuffer)
                var chunk = [Float](repeating: 0, count: byteCount / MemoryLayout<Float>.size)
                CMBlockBufferCopyDataBytes(
                    blockBuffer, atOffset: 0, dataLength: byteCount, destination: &chunk)
                samples.append(contentsOf: chunk)
            }

            guard reader.status == .completed else {
                throw reader.error ?? UserInputError.unableToLoad(url)
            }

            return MLXArray(samples)
            #else
            fatalError("Audio processing is not supported on this platform.")
            #endif
        case .array(let array, let sourceSampleRate):
            guard let sourceSampleRate, sourceSampleRate != processing.sampleRate else {
                return array
            }
            guard sourceSampleRate > 0, processing.sampleRate > 0 else {
                throw AudioArrayConversionError.invalidSampleRate(
                    source: sourceSampleRate,
                    target: processing.sampleRate
                )
            }
            #if canImport(AVFoundation)
            return try Self.resample(
                array,
                from: sourceSampleRate,
                to: processing.sampleRate
            )
            #else
            throw AudioArrayConversionError.unsupportedPlatform
            #endif
        }
    }

    #if canImport(AVFoundation)
    /// Resamples in-memory mono PCM before a model-specific processor consumes it.
    private static func resample(
        _ array: MLXArray,
        from sourceSampleRate: Double,
        to targetSampleRate: Double
    ) throws -> MLXArray {
        guard
            let sourceFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sourceSampleRate,
                channels: 1,
                interleaved: false
            ),
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: targetSampleRate,
                channels: 1,
                interleaved: false
            ),
            let converter = AVAudioConverter(from: sourceFormat, to: targetFormat)
        else {
            throw AudioArrayConversionError.invalidSampleRate(
                source: sourceSampleRate,
                target: targetSampleRate
            )
        }

        let samples = array.asType(.float32).flattened().asArray(Float.self)
        let sourceBuffer = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: AVAudioFrameCount(samples.count)
        )!
        sourceBuffer.frameLength = AVAudioFrameCount(samples.count)
        guard let sourceChannel = sourceBuffer.floatChannelData?.pointee else {
            throw AudioArrayConversionError.conversionFailed
        }
        samples.withUnsafeBufferPointer { pointer in
            if let baseAddress = pointer.baseAddress {
                sourceChannel.update(from: baseAddress, count: samples.count)
            }
        }

        let estimatedFrames = Int(
            ceil(Double(samples.count) * targetSampleRate / sourceSampleRate)
        )
        let targetBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: AVAudioFrameCount(estimatedFrames + 1_024)
        )!
        let state = AudioArrayConversionState()
        var conversionError: NSError?
        let status = converter.convert(to: targetBuffer, error: &conversionError) {
            _, outputStatus in
            if state.didProvideBuffer {
                outputStatus.pointee = .endOfStream
                return nil
            }
            state.didProvideBuffer = true
            outputStatus.pointee = .haveData
            return sourceBuffer
        }
        guard conversionError == nil, status != .error,
            let targetChannel = targetBuffer.floatChannelData?.pointee
        else {
            throw AudioArrayConversionError.conversionFailed
        }
        return MLXArray(
            Array(
                UnsafeBufferPointer(
                    start: targetChannel,
                    count: Int(targetBuffer.frameLength)
                )
            )
        )
    }
    #endif
}

#if canImport(AVFoundation)
private final class AudioArrayConversionState: @unchecked Sendable {
    var didProvideBuffer = false
}
#endif

private enum AudioArrayConversionError: LocalizedError {
    case invalidSampleRate(source: Double, target: Double)
    case conversionFailed
    case unsupportedPlatform

    var errorDescription: String? {
        switch self {
        case .invalidSampleRate(let source, let target):
            "Invalid audio sample-rate conversion from \(source) Hz to \(target) Hz."
        case .conversionFailed:
            "Audio sample-rate conversion failed."
        case .unsupportedPlatform:
            "In-memory audio resampling is not supported on this platform."
        }
    }
}
