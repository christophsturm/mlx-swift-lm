// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite("Speculative drafter registrations")
struct MTPDrafterRegistrationTests {
    private typealias Registration =
        @Sendable (ModelTypeRegistry<any MTPDrafterModel>) async -> Void

    /// Both registrations claim the `qwen3` model type, so their matchers must be disjoint.
    @Test(
        "Each drafter checkpoint loads its own drafter in either registration order",
        arguments: [false, true])
    func registrationsAreDisjoint(reversed: Bool) async throws {
        let registrations: [Registration] = [
            { await LFM2DSparkRegistration.register(into: $0) },
            { await Qwen3DFlashRegistration.register(into: [$0]) },
        ]
        let registry = ModelTypeRegistry<any MTPDrafterModel>()
        for registration in reversed ? registrations.reversed() : registrations {
            await registration(registry)
        }

        let lfm2DSpark = try await registry.createModel(
            configuration: Data(lfm2DSparkJSON.utf8), modelType: "qwen3")
        let qwenDFlash = try await registry.createModel(
            configuration: Data(qwenDFlashJSON.utf8), modelType: "qwen3")

        #expect(lfm2DSpark is LFM2DSparkDraftModel)
        #expect(qwenDFlash is Qwen3DFlashDraftModel)
    }
}

private let lfm2DSparkJSON = """
    {
      "architectures": ["Lfm2DSparkDraftModel"],
      "model_type": "qwen3",
      "hidden_size": 8,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 4,
      "intermediate_size": 16,
      "hidden_act": "silu",
      "rms_norm_eps": 1e-6,
      "vocab_size": 32,
      "rope_theta": 1000000.0,
      "max_position_embeddings": 128,
      "layer_types": ["full_attention", "full_attention"],
      "block_size": 3,
      "dflash_config": {
        "mask_token_id": 31,
        "target_layer_ids": [0, 1],
        "num_target_layers": 2
      },
      "markov_rank": 4,
      "rope_is_neox_style": false,
      "enable_confidence_head": true,
      "markov_head_type": "vanilla"
    }
    """

private let qwenDFlashJSON = """
    {
      "architectures": ["DFlashDraftModel"],
      "model_type": "qwen3",
      "hidden_size": 16,
      "intermediate_size": 32,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 8,
      "hidden_act": "silu",
      "vocab_size": 32,
      "max_position_embeddings": 64,
      "rms_norm_eps": 0.000001,
      "block_size": 4,
      "num_target_layers": 2,
      "layer_types": ["full_attention", "full_attention"],
      "dflash_config": {
        "mask_token_id": 31,
        "target_layer_ids": [0]
      }
    }
    """
