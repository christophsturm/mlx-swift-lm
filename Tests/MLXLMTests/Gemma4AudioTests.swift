// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// MARK: - Gemma 4 VLM audio input
//
// The base Gemma 4 VLM runs audio through the shared `Gemma4AudioModel` and the
// `EmbeddingGemma2AudioProcessor` log-mel extractor; their numerics against Transformers
// are pinned in `EmbeddingGemma2Tests`. These tests cover the Gemma 4 wiring:
// configuration, checkpoint keys and layout, the processor's extractor settings and
// prompt expansion, and the per-clip soft tokens the model scatters onto the audio
// placeholders.

struct Gemma4AudioTests {

    private static let audioTokenId = 102

    /// A tiny `gemma4` config whose audio tower reads the real 128 mel bins.
    private static func tinyConfig() throws -> Gemma4Configuration {
        let json = """
            {
              "model_type": "gemma4",
              "image_token_id": 100,
              "audio_token_id": \(audioTokenId),
              "boa_token_id": 103,
              "eoa_token_id": 104,
              "text_config": {
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                "global_head_dim": 32, "vocab_size": 200, "vocab_size_per_layer_input": 200,
                "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 8,
                "sliding_window": 16, "sliding_window_pattern": 2, "max_position_embeddings": 512
              },
              "vision_config": {
                "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
                "num_attention_heads": 2, "head_dim": 8, "patch_size": 4,
                "default_output_length": 4, "pooling_kernel_size": 1, "position_embedding_size": 8
              },
              "audio_config": {
                "model_type": "gemma4_audio", "hidden_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "subsampling_conv_channels": [128, 4],
                "conv_kernel_size": 3, "attention_chunk_size": 4, "attention_context_left": 5,
                "attention_context_right": 0, "output_proj_dims": 8, "use_clipped_linears": true
              }
            }
            """
        return try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
    }

    private static func processor() throws -> Gemma4Processor {
        let json = """
            {
              "processor_class": "Gemma4Processor",
              "do_normalize": true,
              "image_seq_length": 280,
              "image_token_id": 258880,
              "audio_token_id": \(AudioTestTokenizer.audioTokenId),
              "boa_token_id": 256000,
              "eoa_token_id": 258883
            }
            """
        let config = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self, from: Data(json.utf8))
        return Gemma4Processor(config, tokenizer: AudioTestTokenizer())
    }

    @Test("Gemma4Configuration decodes audio_config and the audio token ids")
    func configurationDecodesAudio() throws {
        let config = try Self.tinyConfig()
        #expect(config.audioConfiguration?.hiddenSize == 16)
        #expect(config.audioConfiguration?.outputProjectionDimensions == 8)
        #expect(config.audioTokenId == Self.audioTokenId)
        #expect(config.boaTokenId == 103)
        #expect(config.eoaTokenId == 104)
    }

    @Test("The audio tower and embedder sit at the checkpoint's module paths")
    func audioModulesUseCheckpointKeys() throws {
        let keys = Set(Gemma4(try Self.tinyConfig()).parameters().flattened().map(\.0))
        for key in [
            "audio_tower.subsample_conv_projection.layer0.conv.weight",
            "audio_tower.subsample_conv_projection.layer1.conv.weight",
            "audio_tower.subsample_conv_projection.input_proj_linear.weight",
            "audio_tower.layers.0.self_attn.q_proj.linear.weight",
            "audio_tower.layers.0.self_attn.relative_k_proj.weight",
            "audio_tower.layers.0.self_attn.per_dim_scale",
            "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight",
            "audio_tower.output_proj.weight",
            "audio_tower.output_proj.bias",
            "embed_audio.embedding_projection.weight",
        ] {
            #expect(keys.contains(key), "missing \(key)")
        }
    }

    @Test("Sanitize keeps audio weights and moves PyTorch convolutions to channels-last")
    func sanitizeKeepsAudioInMLXLayout() throws {
        let model = Gemma4(try Self.tinyConfig())
        let sanitized = model.sanitize(weights: [
            "audio_tower.subsample_conv_projection.layer0.conv.weight":
                MLXArray.zeros([128, 1, 3, 3]),
            "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight": MLXArray.zeros([16, 1, 3]),
            "embed_audio.embedding_projection.weight": MLXArray.zeros([32, 8]),
        ])
        #expect(
            sanitized["audio_tower.subsample_conv_projection.layer0.conv.weight"]?.shape
                == [128, 3, 3, 1])
        #expect(
            sanitized["audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"]?.shape
                == [16, 3, 1])
        #expect(sanitized["embed_audio.embedding_projection.weight"]?.shape == [32, 8])
    }

    @Test(
        "audioTokenCount = ceil(ceil(frames / 2) / 2)",
        arguments: [0, 1, 2, 4, 13, 25, 50, 99, 100, 400, 2999])
    func audioTokenCountFormula(frames: Int) {
        let half = (Double(frames) / 2).rounded(.up)
        #expect(Gemma4Processor.audioTokenCount(frames: frames) == Int((half / 2).rounded(.up)))
    }

    @Test("The token count equals the soft tokens the tower yields for real clips")
    func tokenCountMatchesTheTower() throws {
        let extractor = try Self.processor().audioFeatureExtractor()
        let tower = Gemma4AudioModel(
            config: try #require(try Self.tinyConfig().audioConfiguration))
        eval(tower)
        for samples in [8_000, 16_000, 24_321] {
            let frames = extractor.features(samples: MLXRandom.normal([samples]))
            let softTokens = tower(frames)
            eval(softTokens)
            #expect(
                softTokens.dim(1) == Gemma4Processor.audioTokenCount(frames: frames.dim(0)),
                "\(samples) samples: \(frames.dim(0)) frames gave \(softTokens.dim(1)) tokens")
        }
    }

    /// Numerical parity with the Python reference (`mlx-vlm`
    /// `audio_feature_extractor.py`). The reference pads the 8,000 samples to 50 frames and
    /// zeroes the one that holds padding; the shared extractor keeps only the 49 real
    /// frames, so the reference mean is renormalised over 50 frames.
    @Test("The processor's extractor matches the Python reference")
    func extractorMatchesPythonReference() throws {
        let n = 8000
        // Reduce each tone's phase into [0, 2π) via its fractional cycle count before
        // sin(): the raw phase of a 3 kHz tone at large sample indices (~9400 rad)
        // exceeds float32 precision (and MLX has no GPU float64), so a naive float32
        // computation would diverge from the numpy (float64) reference.
        let i = MLXArray(Array(0 ..< n)).asType(.float32)
        func tone(_ freq: Float) -> MLXArray {
            let cycles = (freq / 16_000.0) * i
            return sin(2 * Float.pi * (cycles - floor(cycles)))
        }
        let wave = 0.6 * tone(220) + 0.3 * tone(1500) + 0.1 * tone(3000)

        let features = try Self.processor().audioFeatureExtractor().features(samples: wave)
        eval(features)

        #expect(features.shape == [49, 128])

        // The reference uses numpy's float64 FFT while this runs MLX's float32 FFT, so
        // log-mel values agree to ~1e-2.
        let referenceMean = features.sum().item(Float.self) / Float(50 * 128)
        #expect(abs(referenceMean - (-3.516562)) < 2e-2)
        // The empty first filter is log(mel_floor), independent of the FFT.
        #expect(abs(features[0, 0].item(Float.self) - (-6.907755)) < 1e-4)

        let probes: [(Int, Int, Float)] = [
            (0, 64, 0.609170), (5, 10, 1.511158), (10, 40, -5.128715),
            (20, 80, -4.040174), (48, 50, -3.333589),
        ]
        for (frame, mel, expected) in probes {
            let value = features[frame, mel].item(Float.self)
            #expect(
                abs(value - expected) < 2e-2,
                "mel[\(frame),\(mel)]: got \(value), reference \(expected)")
        }
    }

    @Test("The processor expands each audio placeholder to its clip's soft tokens")
    func processorExpandsAudioPlaceholders() async throws {
        let input = UserInput(chat: [
            .user(
                "transcribe",
                audios: [
                    .array(MLXArray.zeros([16_000])),
                    .array(MLXArray.zeros([4_000]), sampleRate: 8_000),
                ])
        ])

        let prepared = try await Self.processor().prepare(input: input)

        // 16,000 samples give 99 frames and 25 soft tokens; 4,000 samples at 8 kHz
        // resample to 8,000, 49 frames and 13 soft tokens.
        let audio = AudioTestTokenizer.audioTokenId
        let expected =
            [AudioTestTokenizer.bosTokenId]
            + [256_000] + Array(repeating: audio, count: 25) + [258_883]
            + [256_000] + Array(repeating: audio, count: 13) + [258_883]
            + [AudioTestTokenizer.textTokenId]
        #expect(prepared.text.tokens.asArray(Int.self) == expected)
        let processed = try #require(prepared.audio)
        #expect(processed.features.shape == [2, 99, 128])
        let mask = try #require(processed.mask)
        #expect(
            mask.asArray(Bool.self)
                == Array(repeating: true, count: 99)
                + Array(repeating: true, count: 49) + Array(repeating: false, count: 50))
    }

    @Test("Batched clips give the soft tokens each clip gives alone")
    func paddingDoesNotReachTheSoftTokens() throws {
        let model = Gemma4(try Self.tinyConfig())
        eval(model)
        let long = MLXRandom.normal([20, 128])
        let short = MLXRandom.normal([9, 128])
        // Padding that the tower would mix into the short clip if it read it.
        let padding = MLXArray.ones([11, 128]) * 7
        let batch = stacked([long, concatenated([short, padding], axis: 0)])
        let mask = MLXArray(
            Array(repeating: true, count: 20) + Array(repeating: true, count: 9)
                + Array(repeating: false, count: 11), [2, 20])

        let batched = try model.getAudioFeatures(features: batch, mask: mask)
        let separate = concatenated(
            [
                try model.getAudioFeatures(features: long, mask: nil),
                try model.getAudioFeatures(features: short, mask: nil),
            ], axis: 0)
        eval(batched, separate)

        try #require(batched.shape == [5 + 3, 32])
        #expect(allClose(batched, separate, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    }

    @Test("A prompt with audio prefills through the audio tower")
    func promptWithAudioPrefills() throws {
        let model = Gemma4(try Self.tinyConfig())
        eval(model)
        let tokens = [5, 103] + Array(repeating: Self.audioTokenId, count: 5) + [104, 7]
        let input = LMInput(
            text: .init(tokens: MLXArray(tokens.map(Int32.init)).expandedDimensions(axis: 0)),
            audio: .init(features: MLXRandom.normal([1, 20, 128])))

        let result = try model.prepare(
            input, cache: try model.newCache(parameters: nil), state: nil,
            prefill: .init(stepSize: 1024))

        guard case .logits(let output) = result else {
            Issue.record("expected logits from the multimodal prefill")
            return
        }
        eval(output.logits)
        #expect(output.logits.shape == [1, 1, 200])
    }

    @Test("A prompt whose audio placeholders disagree with the clip is rejected")
    func audioTokenMismatchThrows() throws {
        let model = Gemma4(try Self.tinyConfig())
        eval(model)
        let tokens = [5] + Array(repeating: Self.audioTokenId, count: 4) + [7]
        let input = LMInput(
            text: .init(tokens: MLXArray(tokens.map(Int32.init)).expandedDimensions(axis: 0)),
            audio: .init(features: MLXRandom.normal([1, 20, 128])))

        let error = #expect(throws: (any Error).self) {
            _ = try model.prepare(
                input, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 1024))
        }
        #expect(
            error?.localizedDescription
                == "Gemma4 audio token count mismatch: encoder produced 5 soft tokens, but the prompt contains 4 audio tokens."
        )
    }
}

/// Renders each message as `<bos>`, one placeholder per audio part, and one token per
/// text part, which is all `Gemma4Processor` needs to expand audio placeholders.
private struct AudioTestTokenizer: Tokenizer {
    static let bosTokenId = 2
    static let audioTokenId = 258_881
    static let textTokenId = 7

    let vocabularySize: Int = 262_144
    let bosToken: String? = nil
    let eosToken: String? = nil
    let eosTokenId: Int? = 1
    let unknownToken: String? = nil
    let unknownTokenId: Int? = 0

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [Self.bosTokenId]
            + messages.flatMap { message in
                (message["content"] as? [[String: String]] ?? []).compactMap { part in
                    switch part["type"] {
                    case "audio": Self.audioTokenId
                    case "text": Self.textTokenId
                    default: nil
                    }
                }
            }
    }
}
