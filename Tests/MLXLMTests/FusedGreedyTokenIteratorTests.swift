// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite("Token iterator fused greedy decode")
struct FusedGreedyTokenIteratorTests {
    @Test("Greedy decode without log probabilities takes the fused step")
    func greedyDecodeUsesTheFusedStep() throws {
        let model = FusedDecodeRecordingModel()
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
            model: model,
            parameters: .init(maxTokens: 3, temperature: 0))

        #expect(iterator.next() == FusedDecodeRecordingModel.regularToken)
        #expect(iterator.next() == FusedDecodeRecordingModel.fusedToken)
        #expect(model.fusedDecodeCallCount > 0)
    }

    @Test("Requested log probabilities keep greedy decode on the logits step")
    func logProbabilitiesBypassTheFusedStep() throws {
        let model = FusedDecodeRecordingModel()
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
            model: model,
            parameters: .init(maxTokens: 3, temperature: 0, logProbabilities: 2))

        #expect(iterator.next() == FusedDecodeRecordingModel.regularToken)
        let token = iterator.next()
        let probabilities = try #require(iterator.lastLogProbabilities).materialize()

        #expect(token == FusedDecodeRecordingModel.regularToken)
        #expect(probabilities.chosen.token == token)
        #expect(probabilities.topLogProbabilities.count == 2)
        #expect(model.fusedDecodeCallCount == 0)
    }
}

/// Logits always select ``regularToken``; the fused path always returns ``fusedToken``, so a
/// generated token shows which step produced it.
private final class FusedDecodeRecordingModel: Module, LanguageModel, FusedGreedyDecodeModel,
    KVCacheDimensionProvider
{
    static let regularToken = 48
    static let fusedToken = 49
    static let vocabularySize = 64

    var kvHeads: [Int] { [] }
    private(set) var fusedDecodeCallCount = 0

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        .logits(LMOutput(logits: Self.logits()))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        Self.logits()
    }

    func fusedGreedyDecode(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?,
        tokenMask: MLXArray?
    ) -> FusedGreedyDecodeOutput? {
        fusedDecodeCallCount += 1
        return FusedGreedyDecodeOutput(
            tokens: MLXArray([Int32(Self.fusedToken)]).reshaped(1, 1), state: state)
    }

    private static func logits() -> MLXArray {
        var values = [Float](repeating: -100, count: vocabularySize)
        values[regularToken] = 100
        values[fusedToken] = 50
        return MLXArray(values).reshaped(1, 1, vocabularySize)
    }
}
