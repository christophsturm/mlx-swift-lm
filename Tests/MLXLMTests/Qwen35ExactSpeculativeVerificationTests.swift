import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("Qwen 3.5 exact speculative verification", .serialized)
struct Qwen35ExactSpeculativeVerificationTests {
    @Test("Four-bit GDN block matches sequential decode exactly")
    func quantizedGDNBlockMatchesSingletons() throws {
        MLXRandom.seed(42)
        let configuration = try Self.configuration()
        let layer = Qwen35GatedDeltaNet(configuration)
        Self.quantizeForProduction(layer)
        let input = MLXRandom.normal([1, 4, 512]).asType(.bfloat16)
        let zero = layer.zeroStates(batch: 1, dtype: .bfloat16)

        let block = layer.forward(
            input, convState: zero.conv, recState: zero.rec, mask: nil,
            checkpointAfter: 1, exactSpeculative: true
        ).output
        let singletonCache = MambaCache()
        let singletons = concatenated(
            (0 ..< input.dim(1)).map { token in
                layer(input[0..., token ..< (token + 1), 0...], cache: singletonCache)
            },
            axis: 1)

        eval(block, singletons)
        #expect(arrayEqual(block, singletons).item(Bool.self))
    }

    @Test("Four-bit attention block matches sequential decode exactly")
    func quantizedAttentionBlockMatchesSingletons() throws {
        MLXRandom.seed(43)
        let layer = Qwen35Attention(try Self.configuration())
        Self.quantizeForProduction(layer)
        let input = MLXRandom.normal([1, 4, 512]).asType(.bfloat16)
        let blockCache = KVCacheSimple()
        let singletonCache = KVCacheSimple()

        let block = layer.exactSpeculativeForward(input, cache: blockCache)
        let singletons = concatenated(
            (0 ..< input.dim(1)).map { token in
                layer(
                    input[0..., token ..< (token + 1), 0...],
                    mask: .none,
                    cache: singletonCache)
            },
            axis: 1)

        eval(block, singletons)
        #expect(arrayEqual(block, singletons).item(Bool.self))
    }

    @Test("Four-bit MoE block matches sequential decode exactly")
    func quantizedMoEBlockMatchesSingletons() throws {
        MLXRandom.seed(44)
        let block = Qwen35SparseMoeBlock(try Self.configuration())
        Self.quantizeForProduction(block)
        let input = MLXRandom.normal([1, 4, 512]).asType(.bfloat16)

        let exact = block.exactSpeculativeForward(input)
        let singletons = concatenated(
            (0 ..< input.dim(1)).map { token in
                block(input[0..., token ..< (token + 1), 0...])
            },
            axis: 1)

        eval(exact, singletons)
        #expect(arrayEqual(exact, singletons).item(Bool.self))
    }

    @Test("Four-bit hybrid MoE verification matches sequential target decode exactly")
    func quantizedHybridBlockMatchesSingletons() throws {
        MLXRandom.seed(41)
        let configuration = try Self.configuration()
        let model = Qwen35TextModel(configuration)
        Self.quantizeForProduction(model)

        let blockCache = try model.newCache(parameters: nil)
        let singletonCache = try model.newCache(parameters: nil)
        let prompt = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4))
        let blockPrompt = model(prompt, cache: blockCache, state: nil)
        let singletonPrompt = model(prompt, cache: singletonCache, state: nil)
        let blockPromptCache = blockCache.flatMap(\.state)
        let singletonPromptCache = singletonCache.flatMap(\.state)
        eval(blockPrompt.logits, singletonPrompt.logits, blockPromptCache, singletonPromptCache)
        let promptCacheMatches = zip(blockPromptCache, singletonPromptCache).map {
            arrayEqual($0, $1).item(Bool.self)
        }
        #expect(arrayEqual(blockPrompt.logits, singletonPrompt.logits).item(Bool.self))
        #expect(promptCacheMatches.allSatisfy { $0 })

        let verifyTokens = MLXArray([Int32(5), 6, 7, 8]).reshaped(1, 4)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpTargetLayerCaptureIDsKey] = [0, 1]
        state[mtpCacheCheckpointIndexKey] = 1
        let block = model.verifySpeculativeBlock(
            LMInput.Text(tokens: verifyTokens), cache: blockCache, state: state)
        let singletonOutputs = (0 ..< verifyTokens.dim(1)).map { token in
            model(
                LMInput.Text(
                    tokens: verifyTokens[0..., token ..< (token + 1)]),
                cache: singletonCache,
                state: state)
        }
        let singletonLogits = concatenated(
            singletonOutputs.map(\.logits),
            axis: 1)
        let blockCaptured = try #require(block.state?[mtpLastHiddenStatesKey])
        let singletonCaptured = concatenated(
            singletonOutputs.map { $0.state![mtpLastHiddenStatesKey]! },
            axis: 1)

        eval(block.logits, singletonLogits, blockCaptured, singletonCaptured)
        let firstLayerMatches = (0 ..< verifyTokens.dim(1)).map { token in
            arrayEqual(
                blockCaptured[0..., token, ..<512],
                singletonCaptured[0..., token, ..<512]
            ).item(Bool.self)
        }
        let secondLayerMatches = (0 ..< verifyTokens.dim(1)).map { token in
            arrayEqual(
                blockCaptured[0..., token, 512...],
                singletonCaptured[0..., token, 512...]
            ).item(Bool.self)
        }
        #expect(firstLayerMatches == [true, true, true, true])
        #expect(secondLayerMatches == [true, true, true, true])
        #expect(arrayEqual(block.logits, singletonLogits).item(Bool.self))
        #expect(blockCache.map(\.offset) == singletonCache.map(\.offset))
        #expect(blockCaptured.shape == [1, 4, 1024])
    }

    @Test("Production quantized text model fuses greedy decoding with its LM head")
    func quantizedTextModelFusedGreedyMatchesLogits() throws {
        MLXRandom.seed(45)
        let model = Qwen35TextModel(try Self.configuration())
        Self.quantizeForProduction(model)
        let referenceCache = try model.newCache(parameters: nil)
        let fusedCache = try model.newCache(parameters: nil)
        let input = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4))

        let reference = model(input, cache: referenceCache, state: nil)
        let fused = try #require(
            model.fusedGreedyDecode(
                input, cache: fusedCache, state: nil, tokenMask: nil))
        let referenceTokens = argMax(reference.logits, axis: -1)
        let referenceCacheState = referenceCache.flatMap(\.state)
        let fusedCacheState = fusedCache.flatMap(\.state)

        eval(referenceTokens, fused.tokens, referenceCacheState, fusedCacheState)
        #expect(arrayEqual(referenceTokens, fused.tokens).item(Bool.self))
        #expect(referenceCache.map(\.offset) == fusedCache.map(\.offset))
        #expect(
            zip(referenceCacheState, fusedCacheState).allSatisfy {
                arrayEqual($0, $1).item(Bool.self)
            })
    }

    @Test("Production Qwen fused greedy decoding applies a packed grammar mask")
    func quantizedTextModelFusedGreedyAppliesPackedMask() throws {
        MLXRandom.seed(46)
        let model = Qwen35TextModel(try Self.configuration())
        Self.quantizeForProduction(model)
        let cache = try model.newCache(parameters: nil)
        let input = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4))
        let mask = MLXArray(
            Array(repeating: Int32(1 << 7), count: 4)
        ).reshaped(4, 1)

        let fused = try #require(
            model.fusedGreedyDecode(
                input, cache: cache, state: nil, tokenMask: mask))
        eval(fused.tokens)

        #expect(fused.tokens.asArray(Int32.self) == [7, 7, 7, 7])
        #expect(cache.last?.offset == input.cacheSequenceLength)
    }

    private static func configuration() throws -> Qwen35TextConfiguration {
        try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
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
            "rope_theta": 100000.0,
            "partial_rotary_factor": 1.0
          }
        }
        """
}
