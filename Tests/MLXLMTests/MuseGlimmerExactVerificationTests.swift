// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite("Muse Glimmer exact speculative verification")
struct MuseGlimmerExactVerificationTests {
    /// A Muse-sized block forward rounds differently from one-token decode, so speculation
    /// equals target-only output only when the verification block takes the exact path, both
    /// on the target's caches and through the staged caches of a speculative round.
    @Test("A four-bit verification block matches one-token decode bit for bit past the window")
    func quantizedBlockMatchesSingletons() throws {
        let target = try fourBitMuseTarget()
        let model: any LanguageModel = target
        let exact = try #require(model as? any ExactSpeculativeVerificationModel)
        let prefix = MLXArray((0 ..< 24).map { Int32(($0 * 37 + 5) % 500) }).reshaped(1, -1)
        let block = (0 ..< 16).map { Int32(($0 * 53 + 11) % 500) }
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpTargetLayerIdsKey] = [0, 2]

        func prefilledCache() throws -> [any KVCache] {
            let cache = try target.newCache(parameters: nil)
            _ = target(LMInput.Text(tokens: prefix), cache: cache, state: nil)
            eval(cache)
            return cache
        }
        let singletonCache = try prefilledCache()
        let plainCache = try prefilledCache()
        let directCache = try prefilledCache()
        let storage = KVCacheStorage(try prefilledCache(), plan: .disabled)
        let round = try #require(storage.beginRound(maximumPositions: block.count))

        let blockInput = LMInput.Text(tokens: MLXArray(block).reshaped(1, -1))
        let plain = target(blockInput, cache: plainCache, state: state)
        let direct = exact.verifySpeculativeBlock(blockInput, cache: directCache, state: state)
        let staged = exact.verifySpeculativeBlock(blockInput, cache: round.caches, state: state)
        let singletons = block.map { token in
            target(
                LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1)), cache: singletonCache,
                state: state)
        }
        let singletonLogits = concatenated(singletons.map(\.logits), axis: 1)
        let singletonHidden = concatenated(
            try singletons.map { try #require($0.state?[mtpLastHiddenStatesKey]) }, axis: 1)

        #expect(!arrayEqual(plain.logits, singletonLogits).item(Bool.self))
        for verified in [direct, staged] {
            let verifiedHidden = try #require(verified.state?[mtpLastHiddenStatesKey])
            #expect(arrayEqual(verified.logits, singletonLogits).item(Bool.self))
            #expect(arrayEqual(verifiedHidden, singletonHidden).item(Bool.self))
        }
        _ = storage.commit(round, retaining: block.count)
        #expect(directCache.map(\.offset) == singletonCache.map(\.offset))
        #expect(storage.cache.map(\.offset) == singletonCache.map(\.offset))
    }
}

/// A Muse target wide enough for the exact quantized projection kernels, in bfloat16 with
/// four-bit text weights. Its two sliding layers hold 16 positions; the last layer is global.
private func fourBitMuseTarget() throws -> MuseGlimmer {
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
                "hidden_size": 512,
                "num_hidden_layers": 3,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "intermediate_size": 512,
                "sliding_window": 16,
                "vocab_size": 512,
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
    let target = withRandomState(MLXRandom.RandomState(seed: 29)) {
        MuseGlimmer(configuration)
    }
    target.apply { array in
        array.dtype.isFloatingPoint ? array.asType(.bfloat16) : array
    }
    quantize(model: target, groupSize: 64, bits: 4) { path, _ in
        path.hasPrefix("language_model")
    }
    eval(target)
    return target
}
