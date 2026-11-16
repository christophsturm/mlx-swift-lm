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
        let (model, _) = try seededMuseDFlashPair()
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
        let (model, _) = try seededMuseDFlashPair()
        let text = model.config.textConfiguration
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let table = try #require(weights["language_model.model.embed_tokens.weight"])
        let head = try #require(weights["language_model.lm_head.weight"])
        let tokens = MLXArray([Int32(5), 9, 63]).reshaped([1, 3])
        let hidden = MLXRandom.normal([1, 3, text.hiddenSize], key: MLXRandom.key(5))

        #expect(model.dflashHiddenSize == text.hiddenSize)
        #expect(model.dflashTargetLayerCount == text.hiddenLayers)
        #expect(model.dflashVocabularySize == text.vocabularySize)
        #expect(arrayEqual(model.dflashTokenEmbeddings(tokens), table[tokens]).item(Bool.self))
        #expect(
            allClose(model.dflashLogits(hidden), matmul(hidden, head.T), atol: 1e-5)
                .item(Bool.self))
    }
}

/// Prompts and replies in this suite outgrow the target's sliding window of 8 and the
/// assistant's window of 4, so both rotating caches wrap.
@Suite("Muse Glimmer DFlash speculative decoding")
struct MuseGlimmerDFlashGenerationTests {
    /// With unrelated random weights the assistant's proposals are almost always rejected,
    /// and every target token depends on the whole cached context.
    @Test("Greedy speculative output equals target-only greedy output after rejected blocks")
    func greedySpeculationMatchesTargetOnly() throws {
        let (target, drafter) = try seededMuseDFlashPair()
        let prompt = MLXArray([Int32(3), 17, 42, 5, 29, 11])
        let parameters = GenerateParameters(maxTokens: 24, temperature: 0)

        let expected = try targetOnlyTokens(target, prompt: prompt, parameters: parameters)
        var speculative = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: prompt), mainModel: target, drafter: drafter,
            parameters: parameters, blockSize: 4)
        var tokens = [Int]()
        while let token = speculative.next() { tokens.append(token) }

        #expect(tokens == expected)
        #expect(tokens.count == 24)
        #expect(speculative.passthroughReason == nil)
        #expect(speculative.proposedCount > 0)
        #expect(speculative.acceptedCount < speculative.proposedCount)
    }

    /// With a head that two tokens dominate, the assistant's proposals are often accepted, so
    /// accepted drafts are committed into the target's rotating caches.
    @Test("Greedy speculative output equals target-only greedy output with accepted drafts")
    func greedySpeculationWithAcceptedDrafts() throws {
        let (target, drafter) = try seededMuseDFlashPair(vocabularySize: 16)
        try concentrateHead(of: target)
        let prompt = MLXArray([Int32(3), 7, 12, 5, 9, 11])
        let parameters = GenerateParameters(maxTokens: 24, temperature: 0)

        let expected = try targetOnlyTokens(target, prompt: prompt, parameters: parameters)
        var speculative = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: prompt), mainModel: target, drafter: drafter,
            parameters: parameters, blockSize: 4)
        var tokens = [Int]()
        while let token = speculative.next() { tokens.append(token) }

        #expect(tokens == expected)
        #expect(tokens.count == 24)
        #expect(speculative.passthroughReason == nil)
        #expect(speculative.acceptedCount > 0)
        #expect(speculative.acceptedCount < speculative.proposedCount)
    }
}

/// Tokens a plain target-only iterator generates.
func targetOnlyTokens(
    _ target: MuseGlimmer, prompt: MLXArray, parameters: GenerateParameters
) throws -> [Int] {
    var iterator = try TokenIterator(
        input: LMInput(tokens: prompt), model: target, parameters: parameters)
    var tokens = [Int]()
    while let token = iterator.next() { tokens.append(token) }
    return tokens
}

/// A tiny Muse target and assistant with fixed weights. The target is the one in
/// `MuseGlimmerForwardTests` with an output multiplier of 1, so its logits are not flattened.
func seededMuseDFlashPair(vocabularySize: Int = 64) throws
    -> (MuseGlimmer, MuseGlimmerAssistantDraftModel)
{
    let configuration = try JSONDecoder().decode(
        MuseGlimmerConfiguration.self,
        from: Data(
            """
            {
              "model_type": "muse_glimmer",
              "image_token_id": 7,
              "video_token_id": 8,
              "out_hidden_size": 64,
              "projector_hidden_size": 32,
              "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 5,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "intermediate_size": 64,
                "sliding_window": 8,
                "vocab_size": \(vocabularySize),
                "output_multiplier": 1.0,
                "rope_parameters": { "rope_theta": 500000.0 }
              },
              "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "patch_size": 14,
                "patch_temporal": 2,
                "merge_size": 2,
                "pos_emb_height": 32,
                "pos_emb_width": 32
              }
            }
            """.utf8))
    let assistantConfiguration = try tinyAssistantConfiguration(vocabularySize: vocabularySize)
    let target = withRandomState(MLXRandom.RandomState(seed: 23)) {
        MuseGlimmer(configuration)
    }
    let drafter = withRandomState(MLXRandom.RandomState(seed: 29)) {
        MuseGlimmerAssistantDraftModel(assistantConfiguration)
    }
    eval(target, drafter)
    return (target, drafter)
}

/// Rewrites the shared output head so tokens 4 and 9 take opposite signs of residual
/// coordinate 0 and every other row keeps a tenth of its weight. The target then prefers one
/// of the two by context, and the assistant, which reads the same head, proposes them too.
func concentrateHead(of target: MuseGlimmer) throws {
    let weights = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
    let head = try #require(weights["language_model.lm_head.weight"])
    let direction = MLXArray(
        (0 ..< head.dim(1)).map { $0 == 0 ? Float(3) : Float(0) })
    let rows = (0 ..< head.dim(0)).map { row in
        switch row {
        case 4: direction
        case 9: -direction
        default: head[row] * 0.1
        }
    }
    try target.update(
        parameters: ModuleParameters.unflattened([
            "language_model.lm_head.weight": stacked(rows).asType(head.dtype)
        ]),
        verify: .noUnusedKeys)
    eval(target)
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
