import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("Qwen DFlash checkpoint compatibility")
struct Qwen3DFlashTests {
    @Test("DFlash registers for vision-language targets")
    func registersForVisionLanguageTargets() async throws {
        await Qwen3DFlashRegistration.register()

        let model = try await MTPDrafterTypeRegistry.visionLanguage.createModel(
            configuration: Data(dsparkConfigurationJSON().utf8),
            modelType: "qwen3")

        #expect(model is Qwen3DFlashDraftModel)
    }

    @Test("DSpark normalizes proposal count and adaptive runtime width")
    func decodesDSparkCheckpoint() throws {
        let configuration = try JSONDecoder().decode(
            Qwen3DFlashConfiguration.self,
            from: Data(dsparkConfigurationJSON().utf8))

        #expect(configuration.projector == .dspark)
        #expect(configuration.proposalLength == 7)
        #expect(configuration.verificationBlockSize == 8)
        #expect(configuration.runtimeBlockSize == 8)
        #expect(configuration.initialBlockSize == 4)
        #expect(configuration.blockSizePolicy == .adaptive)
        #expect(configuration.targetLayerIDs == [4, 16, 28, 40, 52])
        #expect(configuration.ropeScaling?["rope_type"] == .string("yarn"))
    }

    @Test("DFlash keeps its published verification width and sliding cache policy")
    func decodesDFlashCheckpoint() throws {
        let configuration = try JSONDecoder().decode(
            Qwen3DFlashConfiguration.self,
            from: Data(dflashConfigurationJSON().utf8))
        let state = Qwen3DFlashDraftModel(configuration).makeState(parameters: nil)

        #expect(configuration.projector == .dflash)
        #expect(configuration.proposalLength == 3)
        #expect(configuration.verificationBlockSize == 4)
        #expect(configuration.runtimeBlockSize == 4)
        #expect(configuration.initialBlockSize == 4)
        #expect(configuration.blockSizePolicy == .fixed)
        #expect(configuration.ropeIsNeoXStyle)
        #expect(!configuration.attentionBias)
        #expect(state.cache.map(\.maxSize) == [nil, 7])
    }

    @Test("DFlash2 recognizes its dynamic selector shape and adaptive widths")
    func decodesDFlash2Checkpoint() throws {
        let configuration = try JSONDecoder().decode(
            Qwen3DFlashConfiguration.self,
            from: Data(dflash2ConfigurationJSON(publishedDimensions: true).utf8))
        let drafter = Qwen3DFlashDraftModel(configuration)
        let parameterKeys = Set(drafter.parameters().flattened().map(\.0))

        #expect(configuration.projector == .dflash2)
        #expect(configuration.proposalLength == 7)
        #expect(configuration.verificationBlockSize == 8)
        #expect(configuration.runtimeBlockSize == 5)
        #expect(configuration.initialBlockSize == 3)
        #expect(configuration.blockSizePolicy == .adaptive)
        #expect(configuration.convolutionKernelSize == 2)
        #expect(configuration.convolutionGroupSize == 16)
        #expect(configuration.selectorRank == 256)
        #expect(configuration.selectorTopK == 16)
        #expect(drafter.adaptiveBlockSizeMinimum == 3)
        #expect(parameterKeys.contains("layers.0.attention_conv.base_kernel"))
        #expect(parameterKeys.contains("layers.0.mlp_conv.kernel_projection.weight"))
        #expect(parameterKeys.contains("candidate_selector.predecessor_codebook.weight"))
    }

    @Test("DFlash2 grouped dynamic convolution only reads current and prior positions")
    func dflash2DynamicConvolutionIsCausal() {
        let hidden = MLXArray(Array(1 ... 12)).reshaped(1, 3, 4)
        let dynamic = MLXArray.zeros([1, 3, 2, 2])
        let base = MLXArray([1, 1, 1, 1, 2, 2, 2, 2]).reshaped(2, 4)
        let expected = MLXArray([1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28])
            .reshaped(1, 3, 4)

        let actual = qwen3DFlash2GroupedDynamicConvolution(
            hidden: hidden, dynamic: dynamic, base: base, groupSize: 2)

        eval(actual)
        #expect(arrayEqual(actual, expected).item(Bool.self))
    }

    @Test("DFlash2 selector follows the strongest candidate at every position")
    func dflash2CandidateSelectorUsesCoherentPath() throws {
        let configuration = try JSONDecoder().decode(
            Qwen3DFlashConfiguration.self,
            from: Data(dflash2ConfigurationJSON().utf8))
        let selector = Qwen3DFlash2CandidateSelector(configuration)
        try selector.update(
            parameters: ModuleParameters.unflattened([
                "predecessor_codebook.weight": MLXArray.zeros([32, 4]),
                "successor_codebook.weight": MLXArray.zeros([32, 4]),
                "hidden_projection.weight": MLXArray.zeros([4, 16]),
            ]),
            verify: [])
        let hidden = MLXArray.zeros([1, 2, 16])
        let logits = MLXArray(
            [0, 1, 5, 2] + Array(repeating: 0, count: 28)
                + [0, 7, 2, 3] + Array(repeating: 0, count: 28)
        ).reshaped(1, 2, 32)

        let selected = selector.select(
            hidden: hidden,
            logits: logits,
            anchorTokenIDs: MLXArray([4]),
            sampler: ArgMaxSampler())

        eval(selected)
        #expect(selected.asArray(Int.self) == [2, 1])
    }

    @Test("DFlash2 sanitization normalizes published codebook tensors")
    func dflash2SanitizeNormalizesCodebooks() throws {
        let configuration = try JSONDecoder().decode(
            Qwen3DFlashConfiguration.self,
            from: Data(dflash2ConfigurationJSON().utf8))
        let drafter = Qwen3DFlashDraftModel(configuration)
        let predecessor = MLXArray.zeros([32, 4])
        let successor = MLXArray.ones([32, 4])

        let sanitized = drafter.sanitize(weights: [
            "candidate_selector.predecessor_codebook": predecessor,
            "candidate_selector.successor_codebook": successor,
        ])

        #expect(
            Set(sanitized.keys) == [
                "candidate_selector.predecessor_codebook.weight",
                "candidate_selector.successor_codebook.weight",
            ])
        #expect(
            arrayEqual(
                sanitized["candidate_selector.predecessor_codebook.weight"]!, predecessor
            ).item(Bool.self))
        #expect(
            arrayEqual(
                sanitized["candidate_selector.successor_codebook.weight"]!, successor
            ).item(Bool.self))
    }

    @Test("DFlash2 drafts a complete candidate path against a Qwen target")
    func dflash2DraftBlockRunsItsProductionGraph() throws {
        let draftConfiguration = try JSONDecoder().decode(
            Qwen3DFlashConfiguration.self,
            from: Data(dflash2ConfigurationJSON().utf8))
        let targetConfiguration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(qwen35DFlashTargetConfigurationJSON().utf8))
        let drafter = Qwen3DFlashDraftModel(draftConfiguration)
        let target = Qwen35TextModel(targetConfiguration)
        var state = drafter.makeState(parameters: nil)

        try drafter.requireCompatibility(with: target)
        let tokens = drafter.draftBlock(
            target: target,
            lastToken: MLXArray([4]),
            lastHidden: MLXRandom.normal([1, 4, 16]),
            sharedKV: [:],
            positionDeltas: nil,
            queryOffset: 4,
            blockSize: 3,
            state: &state,
            sampler: ArgMaxSampler())

        eval(tokens)
        #expect(tokens.shape == [1, 2])
        #expect(tokens.asArray(Int.self).allSatisfy { (0 ..< 32).contains($0) })
    }

    @Test("A speculative session reports which target dimension a DFlash checkpoint misses")
    func sessionReportsTheMismatchedTargetDimension() throws {
        let drafter = Qwen3DFlashDraftModel(
            try JSONDecoder().decode(
                Qwen3DFlashConfiguration.self, from: Data(dflash2ConfigurationJSON().utf8)))
        let matching = Qwen35TextModel(
            try JSONDecoder().decode(
                Qwen35TextConfiguration.self,
                from: Data(qwen35DFlashTargetConfigurationJSON().utf8)))
        let deeper = Qwen35TextModel(
            try JSONDecoder().decode(
                Qwen35TextConfiguration.self, from: Data(qwen35CaptureConfigurationJSON().utf8)))

        #expect(drafter.isCompatible(with: matching))
        #expect(!drafter.isCompatible(with: deeper))
        #expect(throws: DFlashTargetCompatibilityError.layerCount(drafter: 2, target: 3)) {
            _ = try MTPSpeculativeSessionState(
                mainModel: deeper, drafter: drafter, parameters: GenerateParameters())
        }
        #expect(throws: DFlashTargetCompatibilityError.layerCount(drafter: 2, target: 3)) {
            _ = try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
                mainModel: deeper,
                drafter: drafter,
                parameters: GenerateParameters(temperature: 0),
                blockSize: 3)
        }
    }

    @Test("A global draft window bounds every DFlash layer cache")
    func appliesDFlashDraftWindow() throws {
        let configuration = try JSONDecoder().decode(
            Qwen3DFlashConfiguration.self,
            from: Data(dflashConfigurationJSON(draftWindowSize: 5).utf8))
        let state = Qwen3DFlashDraftModel(configuration).makeState(parameters: nil)

        #expect(state.cache.map(\.maxSize) == [5, 5])
    }

    @Test("Sliding DFlash layers require an architectural window")
    func rejectsSlidingLayerWithoutWindow() {
        #expect(throws: Qwen3DFlashConfigurationError.self) {
            try JSONDecoder().decode(
                Qwen3DFlashConfiguration.self,
                from: Data(dflashConfigurationJSON(includeSlidingWindow: false).utf8))
        }
    }

    @Test("DFlash rejects projection bias unsupported by the published backbone")
    func rejectsAttentionBias() {
        #expect(throws: Qwen3DFlashConfigurationError.self) {
            try JSONDecoder().decode(
                Qwen3DFlashConfiguration.self,
                from: Data(dflashConfigurationJSON(attentionBias: true).utf8))
        }
    }

    @Test("Target capture layers must fit the target transformer")
    func rejectsOutOfRangeTargetLayer() {
        #expect(throws: Qwen3DFlashConfigurationError.self) {
            try JSONDecoder().decode(
                Qwen3DFlashConfiguration.self,
                from: Data(dsparkConfigurationJSON(lastTargetLayer: 64).utf8))
        }
    }

    @Test("Runtime verification width must contain an anchor and a draft")
    func rejectsSingleTokenRuntimeBlock() {
        #expect(throws: Qwen3DFlashConfigurationError.self) {
            try JSONDecoder().decode(
                Qwen3DFlashConfiguration.self,
                from: Data(dsparkConfigurationJSON(runtimeBlockSize: 1).utf8))
        }
    }

    @Test("Adaptive initial width cannot exceed the runtime ceiling")
    func rejectsInitialBlockAboveRuntimeBlock() {
        #expect(throws: Qwen3DFlashConfigurationError.self) {
            try JSONDecoder().decode(
                Qwen3DFlashConfiguration.self,
                from: Data(
                    dsparkConfigurationJSON(runtimeBlockSize: 4, initialBlockSize: 6).utf8))
        }
    }

    @Test("Qwen emits requested post-layer residuals in capture order")
    func capturesTargetLayers() throws {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(qwen35CaptureConfigurationJSON().utf8))
        let model = Qwen35TextModel(configuration)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpTargetLayerCaptureIDsKey] = [0, 2]

        let tokens = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let output = model(LMInput.Text(tokens: tokens), cache: nil, state: state)
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(hidden)

        #expect(hidden.shape == [1, 3, 32])
    }
}

private func dsparkConfigurationJSON(
    lastTargetLayer: Int = 52,
    runtimeBlockSize: Int? = nil,
    initialBlockSize: Int? = nil
) -> String {
    let runtimeBlock = runtimeBlockSize.map { "\"runtime_block_size\": \($0)," } ?? ""
    let initialBlock = initialBlockSize.map { "\"dflash_initial_block_size\": \($0)," } ?? ""
    return """
        {
          "model_type": "qwen3",
          "hidden_size": 5120,
          "intermediate_size": 10240,
          "num_hidden_layers": 5,
          "num_attention_heads": 40,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "vocab_size": 248320,
          "max_position_embeddings": 262144,
          "rms_norm_eps": 0.000001,
          "block_size": 7,
          "num_target_layers": 64,
          "markov_rank": 256,
          "enable_confidence_head": true,
          "confidence_head_with_markov": true,
          "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 10000000,
            "factor": 4.0,
            "original_max_position_embeddings": 65536
          },
          "dflash_config": {
            "projector_type": "dspark",
            \(runtimeBlock)
            \(initialBlock)
            "mask_token_id": 248077,
            "target_layer_ids": [4, 16, 28, 40, \(lastTargetLayer)]
          }
        }
        """
}

private func dflashConfigurationJSON(
    includeSlidingWindow: Bool = true,
    draftWindowSize: Int? = nil,
    attentionBias: Bool = false
) -> String {
    let slidingWindow = includeSlidingWindow ? "\"sliding_window\": 8," : ""
    let draftWindow = draftWindowSize.map { "\"draft_window_size\": \($0)," } ?? ""
    return """
        {
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
          "rope_is_neox_style": true,
          "attention_bias": \(attentionBias),
          "block_size": 4,
          "num_target_layers": 2,
          "layer_types": ["full_attention", "sliding_attention"],
          \(slidingWindow)
          "dflash_config": {
            "projector_type": "dflash",
            \(draftWindow)
            "mask_token_id": 31,
            "target_layer_ids": [0]
          }
        }
        """
}

private func dflash2ConfigurationJSON(publishedDimensions: Bool = false) -> String {
    let hiddenSize = publishedDimensions ? 5120 : 16
    let intermediateSize = publishedDimensions ? 17_408 : 32
    let hiddenLayers = publishedDimensions ? 5 : 1
    let attentionHeads = publishedDimensions ? 32 : 2
    let keyValueHeads = publishedDimensions ? 8 : 1
    let headDimension = publishedDimensions ? 128 : 8
    let vocabularySize = publishedDimensions ? 248_320 : 32
    let targetLayers = publishedDimensions ? 64 : 2
    let layerTypes =
        publishedDimensions
        ? "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"sliding_attention\"]"
        : "[\"full_attention\"]"
    let slidingWindow = publishedDimensions ? "\"sliding_window\": 2048," : ""
    let blockSize = publishedDimensions ? 8 : 3
    let convolutionGroupSize = publishedDimensions ? 16 : 4
    let selectorRank = publishedDimensions ? 256 : 4
    let selectorTopK = publishedDimensions ? 16 : 4
    let maskTokenID = publishedDimensions ? 248_070 : 31
    let targetLayerIDs = publishedDimensions ? "[5, 19, 33, 47, 61]" : "[0]"
    return """
        {
          "architectures": ["DFlash2DraftModel"],
          "model_type": "qwen3",
          "is_causal": false,
          "hidden_size": \(hiddenSize),
          "intermediate_size": \(intermediateSize),
          "num_hidden_layers": \(hiddenLayers),
          "num_attention_heads": \(attentionHeads),
          "num_key_value_heads": \(keyValueHeads),
          "head_dim": \(headDimension),
          "hidden_act": "silu",
          "rms_norm_eps": 0.000001,
          "vocab_size": \(vocabularySize),
          "max_position_embeddings": 262144,
          "num_target_layers": \(targetLayers),
          "layer_types": \(layerTypes),
          \(slidingWindow)
          "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000
          },
          "dflash_config": {
            "block_size": \(blockSize),
            "conv_group_size": \(convolutionGroupSize),
            "conv_kernel_size": 2,
            "mask_token_id": \(maskTokenID),
            "selector_rank": \(selectorRank),
            "selector_top_k": \(selectorTopK),
            "target_layer_ids": \(targetLayerIDs)
          }
        }
        """
}

private func qwen35CaptureConfigurationJSON() -> String {
    """
    {
      "model_type": "qwen3_5_text",
      "hidden_size": 16,
      "num_hidden_layers": 3,
      "intermediate_size": 32,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 8,
      "linear_num_value_heads": 2,
      "linear_num_key_heads": 1,
      "linear_key_head_dim": 8,
      "linear_value_head_dim": 8,
      "linear_conv_kernel_dim": 2,
      "rms_norm_eps": 0.000001,
      "vocab_size": 16,
      "rope_theta": 100000.0,
      "partial_rotary_factor": 0.25,
      "max_position_embeddings": 64,
      "tie_word_embeddings": true,
      "attention_bias": false,
      "full_attention_interval": 1,
      "mtp_num_hidden_layers": 1,
      "mtp_use_dedicated_embeddings": false,
      "num_experts": 0,
      "num_experts_per_tok": 0,
      "moe_intermediate_size": 16,
      "shared_expert_intermediate_size": 16,
      "rope_parameters": {
        "type": "default",
        "rope_theta": 100000.0,
        "partial_rotary_factor": 0.25
      }
    }
    """
}

private func qwen35DFlashTargetConfigurationJSON() -> String {
    """
    {
      "model_type": "qwen3_5_text",
      "hidden_size": 16,
      "num_hidden_layers": 2,
      "intermediate_size": 32,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 8,
      "linear_num_value_heads": 2,
      "linear_num_key_heads": 2,
      "linear_key_head_dim": 4,
      "linear_value_head_dim": 4,
      "linear_conv_kernel_dim": 4,
      "rms_norm_eps": 0.000001,
      "vocab_size": 32,
      "rope_theta": 10000.0,
      "partial_rotary_factor": 0.25,
      "max_position_embeddings": 128,
      "tie_word_embeddings": true,
      "attention_bias": false,
      "full_attention_interval": 2,
      "mtp_num_hidden_layers": 1,
      "mtp_use_dedicated_embeddings": false,
      "num_experts": 0,
      "num_experts_per_tok": 0,
      "moe_intermediate_size": 16,
      "shared_expert_intermediate_size": 16,
      "rope_parameters": {
        "type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [1, 0, 0]
      }
    }
    """
}
