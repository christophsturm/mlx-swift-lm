// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers Qwen3.5/Qwen3.6 VLM MTP drafter model types.
///
/// Callers should invoke this once before loading a Qwen VLM drafter through
/// `MTPDrafterModelFactory`. The `"qwen3_5"` names overlap with the text
/// registration; config-shape predicates keep both registrations usable in
/// the same process.
public enum Qwen35VLMMTPRegistration {
    private struct StandaloneConfiguration: Decodable {
        let textConfiguration: Qwen35Configuration.TextConfiguration
        let blockSize: Int?

        enum CodingKeys: String, CodingKey {
            case textConfiguration = "text_config"
            case blockSize = "block_size"
        }
    }

    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5",
            matches: qwen35VLMMTPConfiguration,
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35VLMNextNDraftModel(config)
            }
        )
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5_mtp",
            matches: qwen35VLMMTPConfiguration,
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35VLMNextNDraftModel(config, preconvertedNorms: true)
            }
        )
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5_moe",
            matches: qwen35VLMMTPConfiguration,
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35Configuration.self, from: data)
                return Qwen35VLMNextNDraftModel(config)
            }
        )

        for modelType in ["qwen3_5", "qwen3_5_moe"] {
            await MTPDrafterTypeRegistry.visionLanguage.registerModelType(
                modelType,
                creator: { data in
                    let config = try JSONDecoder.json5().decode(
                        Qwen35Configuration.self, from: data)
                    return Qwen35VLMNextNDraftModel(config)
                }
            )
        }
        await MTPDrafterTypeRegistry.visionLanguage.registerModelType(
            "qwen3_5_mtp",
            creator: { data in
                let config = try JSONDecoder.json5().decode(
                    StandaloneConfiguration.self, from: data)
                return Qwen35VLMNextNDraftModel(
                    config.textConfiguration,
                    preconvertedNorms: true,
                    maximumBlockSize: config.blockSize)
            }
        )
    }
}

private func qwen35VLMMTPConfiguration(_ data: Data) -> Bool {
    guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let vision = root["vision_config"] as? [String: Any]
    else { return false }
    return !vision.isEmpty
}
