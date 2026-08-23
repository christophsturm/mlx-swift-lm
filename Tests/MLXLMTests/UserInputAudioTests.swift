import MLX
import MLXLMCommon
import Testing

@Suite("User input audio")
struct UserInputAudioTests {
    @Test("in-memory PCM is resampled from its source rate")
    func inMemoryPCMIsResampledFromItsSourceRate() async throws {
        var processing = UserInput.AudioProcessing()
        processing.sampleRate = 16_000
        let source = UserInput.Audio.array(
            MLXArray([Float](repeating: 0.25, count: 8_000)),
            sampleRate: 8_000
        )

        let samples = try await source.asMLXArray(processing: processing)

        #expect(samples.size == 16_000)
    }
}
