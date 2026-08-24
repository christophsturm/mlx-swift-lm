import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite("Qwen 3.5 VLM exact speculative verification", .serialized)
struct Qwen35VLMExactSpeculativeVerificationTests {
    @Test("Four-bit hybrid VLM continuation matches sequential target decode exactly")
    func quantizedHybridContinuationMatchesSingletons() throws {
        MLXRandom.seed(51)
        let model = Qwen35(try Self.configuration())
        Self.quantizeForProduction(model.languageModel)

        let blockCache = try model.newCache(parameters: nil)
        let singletonCache = try model.newCache(parameters: nil)
        var initialState = LMOutput.State()
        initialState[mtpEmitFlagKey] = true
        initialState[mtpTargetLayerCaptureIDsKey] = [0, 1]
        let prompt = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4))
        let blockPrompt = model(prompt, cache: blockCache, state: initialState)
        let singletonPrompt = model(prompt, cache: singletonCache, state: initialState)
        let blockPromptCache = blockCache.flatMap(\.state)
        let singletonPromptCache = singletonCache.flatMap(\.state)
        eval(blockPrompt.logits, singletonPrompt.logits, blockPromptCache, singletonPromptCache)
        #expect(arrayEqual(blockPrompt.logits, singletonPrompt.logits).item(Bool.self))
        #expect(
            zip(blockPromptCache, singletonPromptCache).allSatisfy {
                arrayEqual($0, $1).item(Bool.self)
            })

        let verifyTokens = MLXArray([Int32(5), 6, 7, 8]).reshaped(1, 4)
        var blockState = try #require(blockPrompt.state)
        blockState[mtpCacheCheckpointIndexKey] = 1
        let block = model.verifySpeculativeBlock(
            LMInput.Text(tokens: verifyTokens), cache: blockCache, state: blockState)

        var singletonState = try #require(singletonPrompt.state)
        let singletonOutputs = (0 ..< verifyTokens.dim(1)).map { token in
            singletonState[mtpCacheCheckpointIndexKey] = 1
            let output = model(
                LMInput.Text(tokens: verifyTokens[0..., token ..< (token + 1)]),
                cache: singletonCache,
                state: singletonState)
            singletonState = output.state ?? singletonState
            return output
        }
        let singletonLogits = concatenated(singletonOutputs.map(\.logits), axis: 1)
        let blockCaptured = try #require(block.state?[mtpLastHiddenStatesKey])
        let singletonCaptured = concatenated(
            singletonOutputs.map { $0.state![mtpLastHiddenStatesKey]! }, axis: 1)

        eval(block.logits, singletonLogits, blockCaptured, singletonCaptured)
        #expect(arrayEqual(block.logits, singletonLogits).item(Bool.self))
        #expect(arrayEqual(blockCaptured, singletonCaptured).item(Bool.self))
        #expect(blockCache.map(\.offset) == singletonCache.map(\.offset))
        #expect(block.state?[mtpPositionDeltasKey] != nil)
        #expect(blockCaptured.shape == [1, 4, 1024])
    }

    @Test("Production quantized VLM fuses warm greedy continuation without losing M-RoPE state")
    func quantizedVLMFusedGreedyPreservesContinuationState() throws {
        MLXRandom.seed(52)
        let model = Qwen35(try Self.configuration())
        Self.quantizeForProduction(model.languageModel)
        let referenceCache = try model.newCache(parameters: nil)
        let fusedCache = try model.newCache(parameters: nil)
        let prompt = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4))
        let referencePrompt = model(prompt, cache: referenceCache, state: nil)
        let fusedPrompt = model(prompt, cache: fusedCache, state: nil)
        eval(referencePrompt.logits, fusedPrompt.logits)

        let continuation = LMInput.Text(
            tokens: MLXArray([Int32(5), 6, 7, 8]).reshaped(1, 4))
        let reference = model(
            continuation, cache: referenceCache, state: referencePrompt.state)
        let fused = try #require(
            model.fusedGreedyDecode(
                continuation, cache: fusedCache, state: fusedPrompt.state,
                tokenMask: nil))
        let referenceTokens = argMax(reference.logits, axis: -1)
        eval(referenceTokens, fused.tokens)
        #expect(arrayEqual(referenceTokens, fused.tokens).item(Bool.self))

        let next = LMInput.Text(tokens: MLXArray([Int32(9)]).reshaped(1, 1))
        let referenceNext = model(next, cache: referenceCache, state: reference.state)
        let fusedNext = model(next, cache: fusedCache, state: fused.state)
        let referenceCacheState = referenceCache.flatMap(\.state)
        let fusedCacheState = fusedCache.flatMap(\.state)
        eval(referenceNext.logits, fusedNext.logits, referenceCacheState, fusedCacheState)
        #expect(arrayEqual(referenceNext.logits, fusedNext.logits).item(Bool.self))
        #expect(referenceCache.map(\.offset) == fusedCache.map(\.offset))
        #expect(
            zip(referenceCacheState, fusedCacheState).allSatisfy {
                arrayEqual($0, $1).item(Bool.self)
            })
    }

    private static func configuration() throws -> Qwen35Configuration {
        try JSONDecoder().decode(
            Qwen35Configuration.self,
            from: Data(configurationJSON.utf8))
    }

    private static func quantizeForProduction(_ module: Module) {
        module.apply { array in
            array.dtype.isFloatingPoint ? array.asType(.bfloat16) : array
        }
        quantize(model: module, groupSize: 64, bits: 4)
    }

    private static let configurationJSON = """
        {
          "model_type": "qwen3_5",
          "image_token_id": 500,
          "video_token_id": 501,
          "vision_start_token_id": 502,
          "vision_end_token_id": 503,
          "vocab_size": 512,
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 512,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "linear_num_value_heads": 8,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_conv_kernel_dim": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 512,
            "rope_theta": 100000.0,
            "partial_rotary_factor": 1.0,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "full_attention_interval": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "norm_topk_prob": true,
            "rope_parameters": {
              "type": "default",
              "mrope_section": [11, 11, 10],
              "rope_theta": 100000.0,
              "partial_rotary_factor": 1.0
            }
          },
          "vision_config": {
            "model_type": "qwen3_5_vit",
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "out_hidden_size": 512,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "num_position_embeddings": 16
          }
        }
        """
}
