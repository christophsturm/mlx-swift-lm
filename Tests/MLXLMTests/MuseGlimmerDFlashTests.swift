// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite("Muse Glimmer DFlash target")
struct MuseGlimmerDFlashTargetTests {
    @Test("Muse emits requested post-layer residuals in capture order")
    func capturesTargetLayers() throws {
        let model = try MuseGlimmerForwardTests.model()
        let text = model.config.textConfiguration
        let width = text.hiddenSize
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])

        let everyLayer = try #require(
            emitDrafterState(model, tokens, layers: Array(0 ..< text.hiddenLayers))
                .state?[mtpLastHiddenStatesKey])
        let output = emitDrafterState(model, tokens, layers: [1, 3, 4])
        let captured = try #require(output.state?[mtpLastHiddenStatesKey])
        func layer(_ index: Int) -> MLXArray {
            everyLayer[.ellipsis, (index * width) ..< ((index + 1) * width)]
        }

        #expect(captured.shape == [1, 3, 3 * width])
        #expect(
            arrayEqual(captured, concatenated([layer(1), layer(3), layer(4)], axis: -1))
                .item(Bool.self))

        // The last layer's residual is what the final norm and the scaled, capped head read.
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let norm = try #require(weights["language_model.model.norm.weight"])
        let head = try #require(weights["language_model.lm_head.weight"])
        let normalized = MLXFast.rmsNorm(layer(4), weight: norm, eps: text.rmsNormEps)
        let scaled = matmul(normalized, head.T) * text.outputMultiplier
        let cap = text.finalLogitSoftcapping
        #expect(allClose(output.logits, tanh(scaled / cap) * cap, atol: 1e-5).item(Bool.self))
    }

    /// The reference binds the raw lookup table, without Muse's `embed_norm`, and the raw
    /// `lm_head`, without the output multiplier or the soft cap.
    @Test("Muse hands a drafter its raw token embeddings and raw output head")
    func rawEmbeddingsAndHead() throws {
        let model = try MuseGlimmerForwardTests.model()
        let text = model.config.textConfiguration
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let table = try #require(weights["language_model.model.embed_tokens.weight"])
        let head = try #require(weights["language_model.lm_head.weight"])
        let tokens = MLXArray([Int32(5), 9, 63]).reshaped([1, 3])
        let hidden = MLXRandom.normal([1, 3, text.hiddenSize])

        #expect(model.dflashHiddenSize == text.hiddenSize)
        #expect(model.dflashTargetLayerCount == text.hiddenLayers)
        #expect(model.dflashVocabularySize == text.vocabularySize)
        #expect(arrayEqual(model.dflashTokenEmbeddings(tokens), table[tokens]).item(Bool.self))
        #expect(
            allClose(model.dflashLogits(hidden), matmul(hidden, head.T), atol: 1e-5)
                .item(Bool.self))
    }
}

/// Runs one uncached forward pass that asks the target for drafter state.
private func emitDrafterState(
    _ model: MuseGlimmer, _ tokens: MLXArray, layers: [Int]
) -> LMOutput {
    var state = LMOutput.State()
    state[mtpEmitFlagKey] = true
    state[mtpTargetLayerIdsKey] = layers
    return model(LMInput.Text(tokens: tokens), cache: nil, state: state)
}
