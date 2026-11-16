// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite("Muse Glimmer assistant drafter")
struct MuseGlimmerAssistantTests {
    @Test("The published assistant config decodes with the reference defaults")
    func decodesPublishedConfiguration() throws {
        let configuration = try JSONDecoder().decode(
            MuseGlimmerAssistantConfiguration.self,
            from: Data(publishedAssistantConfigJSON.utf8))

        #expect(configuration.modelType == "muse_glimmer_assistant")
        #expect(configuration.hiddenSize == 6656)
        #expect(configuration.intermediateSize == 19968)
        #expect(configuration.hiddenLayers == 5)
        #expect(configuration.attentionHeads == 32)
        #expect(configuration.keyValueHeads == 8)
        #expect(configuration.headDimension == 128)
        #expect(configuration.rmsNormEpsilon == 1e-5)
        #expect(configuration.maximumPositionEmbeddings == 131072)
        #expect(configuration.ropeTheta == 500_000)
        #expect(configuration.layerTypes == Array(repeating: "sliding_attention", count: 5))
        #expect(configuration.slidingWindow == 2048)
        #expect(configuration.blockSize == 16)
        #expect(configuration.maskTokenID == 201818)
        #expect(configuration.targetLayerIDs == [1, 13, 25, 37, 49])
        // The published file omits these; the reference supplies Muse Glimmer 30B's values.
        #expect(configuration.targetLayerCount == 52)
        #expect(configuration.vocabularySize == 202048)
    }

    @Test(
        "Configs the reference rejects do not decode",
        arguments: [
            #""model_type": "muse_glimmer_assistant""#: #""model_type": "qwen3""#,
            #""mask_token_id": 201818"#: #""mask_token_id": 202048"#,
            #""target_layer_ids": ["#: #""target_layer_ids": [49, "#,
            #""sliding_window": 2048"#: #""sliding_window": 0"#,
        ])
    func rejectsInvalidConfiguration(original: String, replacement: String) throws {
        let json = publishedAssistantConfigJSON.replacingOccurrences(
            of: original, with: replacement)
        #expect(json != publishedAssistantConfigJSON)
        #expect(throws: MuseGlimmerAssistantConfigurationError.self) {
            try JSONDecoder().decode(
                MuseGlimmerAssistantConfiguration.self, from: Data(json.utf8))
        }
    }

    /// The names are those of the published `model.safetensors`; the shapes are scaled down.
    @Test("Weights load under the published checkpoint names")
    func loadsPublishedWeightNames() throws {
        let configuration = try tinyAssistantConfiguration()
        let hidden = configuration.hiddenSize
        let queries = configuration.attentionHeads * configuration.headDimension
        let keys = configuration.keyValueHeads * configuration.headDimension
        let intermediate = configuration.intermediateSize
        var shapes: [String: [Int]] = [
            "encoder.fc.weight": [hidden, configuration.targetLayerIDs.count * hidden],
            "encoder.output_norm_enc.weight": [hidden],
            "norm.weight": [hidden],
        ]
        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "layers.\(layer)."
            shapes[prefix + "input_layernorm.weight"] = [hidden]
            shapes[prefix + "post_attention_layernorm.weight"] = [hidden]
            shapes[prefix + "self_attn.q_proj.weight"] = [queries, hidden]
            shapes[prefix + "self_attn.k_proj.weight"] = [keys, hidden]
            shapes[prefix + "self_attn.v_proj.weight"] = [keys, hidden]
            shapes[prefix + "self_attn.o_proj.weight"] = [hidden, queries]
            shapes[prefix + "self_attn.q_norm.weight"] = [configuration.headDimension]
            shapes[prefix + "self_attn.k_norm.weight"] = [configuration.headDimension]
            shapes[prefix + "mlp.gate_proj.weight"] = [intermediate, hidden]
            shapes[prefix + "mlp.up_proj.weight"] = [intermediate, hidden]
            shapes[prefix + "mlp.down_proj.weight"] = [hidden, intermediate]
        }
        let weights = shapes.mapValues { MLXArray.ones($0) }
        let drafter = withRandomState(MLXRandom.RandomState(seed: 31)) {
            MuseGlimmerAssistantDraftModel(configuration)
        }

        try drafter.update(
            parameters: ModuleParameters.unflattened(drafter.sanitize(weights: weights)),
            verify: .all)

        let loaded = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())
        #expect(loaded.keys.sorted() == shapes.keys.sorted())
        #expect(loaded.mapValues(\.shape) == shapes)
    }

    /// Context rows 0-3 fill a window of 4, then single rows 4 and 5 wrap the ring buffer.
    /// The proposals sit at positions 7-9; position 2 is more than the window away from all
    /// of them and reaches only the anchor at 6, whose logits are not proposals. One layer
    /// keeps the anchor from carrying it further.
    @Test("Proposals attend only to context within the sliding window, across the ring wrap")
    func proposalsRespectTheSlidingWindow() throws {
        let (target, _) = try seededMuseDFlashPair()
        let configuration = try tinyAssistantConfiguration(hiddenLayers: 1, slidingWindow: 4)
        let drafter = withRandomState(MLXRandom.RandomState(seed: 37)) {
            MuseGlimmerAssistantDraftModel(configuration)
        }
        eval(drafter)
        let width = target.dflashHiddenSize
        let context = MLXRandom.normal([1, 6, width], key: MLXRandom.key(7))
        let replacement = MLXRandom.normal([1, 1, width], key: MLXRandom.key(8))

        func proposalLogits(replacingPosition position: Int?) -> MLXArray {
            var rows = context
            if let position {
                rows = concatenated(
                    [
                        context[0..., ..<position, 0...], replacement,
                        context[0..., (position + 1)..., 0...],
                    ], axis: 1)
            }
            var state = drafter.makeState(parameters: nil)
            let recorder = LogitRecorder()
            for range in [0 ..< 4, 4 ..< 5, 5 ..< 6] {
                _ = drafter.draftBlock(
                    target: target, lastToken: MLXArray([Int32(1)]),
                    lastHidden: rows[0..., range, 0...], sharedKV: [:], positionDeltas: nil,
                    queryOffset: 0, blockSize: 4, state: &state, sampler: recorder)
            }
            let logits = recorder.logits.last!
            eval(logits)
            return logits
        }

        let reference = proposalLogits(replacingPosition: nil)
        #expect(arrayEqual(reference, proposalLogits(replacingPosition: 2)).item(Bool.self))
        // Position 3 is exactly the window away from the first proposal, which still sees it.
        #expect(!allClose(reference, proposalLogits(replacingPosition: 3)).item(Bool.self))
    }
}

/// Records the logits a drafter samples from and returns their argmax.
private final class LogitRecorder: LogitSampler {
    private(set) var logits = [MLXArray]()

    func sample(logits: MLXArray) -> MLXArray {
        self.logits.append(logits)
        return argMax(logits, axis: -1)
    }
}

/// A small assistant for the tiny Muse target in `MuseGlimmerForwardTests`.
func tinyAssistantConfiguration(
    hiddenLayers: Int = 2, slidingWindow: Int = 4, vocabularySize: Int = 64
) throws -> MuseGlimmerAssistantConfiguration {
    let layerTypes = Array(repeating: #""sliding_attention""#, count: hiddenLayers)
    let targetLayerIDs = [1, 3, 4].prefix(hiddenLayers).map(String.init)
    return try JSONDecoder().decode(
        MuseGlimmerAssistantConfiguration.self,
        from: Data(
            """
            {
              "model_type": "muse_glimmer_assistant",
              "hidden_size": 32,
              "intermediate_size": 64,
              "num_hidden_layers": \(hiddenLayers),
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 8,
              "rms_norm_eps": 1e-05,
              "max_position_embeddings": 4096,
              "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" },
              "layer_types": [\(layerTypes.joined(separator: ", "))],
              "sliding_window": \(slidingWindow),
              "block_size": 4,
              "mask_token_id": \(vocabularySize - 1),
              "target_layer_ids": [\(targetLayerIDs.joined(separator: ", "))],
              "num_target_layers": 5,
              "vocab_size": \(vocabularySize)
            }
            """.utf8))
}

/// `config.json` of `meta-models/Muse-Glimmer-30B-assistant` at revision
/// `c676cce91658f9568aaf264a416211e2782735b7`, byte for byte.
let publishedAssistantConfigJSON = """
    {
      "architectures": [
        "MuseGlimmerAssistantModel"
      ],
      "attention_dropout": 0,
      "block_size": 16,
      "bos_token_id": 200000,
      "dtype": "bfloat16",
      "eos_token_id": 200001,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 6656,
      "intermediate_size": 19968,
      "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention"
      ],
      "mask_token_id": 201818,
      "max_position_embeddings": 131072,
      "model_type": "muse_glimmer_assistant",
      "num_attention_heads": 32,
      "num_hidden_layers": 5,
      "num_key_value_heads": 8,
      "pad_token_id": 200018,
      "rms_norm_eps": 1e-05,
      "rope_parameters": {
        "rope_theta": 500000.0,
        "rope_type": "default"
      },
      "sliding_window": 2048,
      "target_layer_ids": [
        1,
        13,
        25,
        37,
        49
      ],
      "transformers_version": "5.15.0.dev0"
    }

    """
