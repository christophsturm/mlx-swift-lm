// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers standalone Qwen-style DFlash, DFlash2, and DSpark speculative checkpoints.
///
/// Callers invoke this once before loading one of these drafters through
/// `MTPDrafterModelFactory`. The raw Qwen `model_type` registration is
/// guarded by checkpoint shape so ordinary Qwen language models do not match.
public enum Qwen3DFlashRegistration {
    /// Adds all published and normalized checkpoint type names to the shared registry.
    public static func register() async {
        for modelType in ["qwen3", "qwen3_dflash", "dflash2", "dspark"] {
            await MTPDrafterTypeRegistry.shared.registerModelType(
                modelType,
                matches: qwen3DFlashConfiguration,
                creator: { data in
                    let configuration = try JSONDecoder.json5().decode(
                        Qwen3DFlashConfiguration.self, from: data)
                    return Qwen3DFlashDraftModel(configuration)
                }
            )
        }
    }
}

/// Recognizes the nested metadata that distinguishes a drafter from a Qwen target.
private func qwen3DFlashConfiguration(_ data: Data) -> Bool {
    guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let dflash = root["dflash_config"] as? [String: Any]
    else { return false }
    return dflash["target_layer_ids"] is [Any]
}
