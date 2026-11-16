// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// The `architectures` entry that identifies a Liquid AI DSpark checkpoint.
let lfm2DSparkArchitecture = "Lfm2DSparkDraftModel"

/// Registers Liquid AI's `Lfm2DSparkDraftModel` checkpoints with the shared
/// block-drafter factory.
public enum LFM2DSparkRegistration {
    public static func register() async {
        await register(into: MTPDrafterTypeRegistry.shared)
    }

    package static func register(into registry: ModelTypeRegistry<any MTPDrafterModel>) async {
        await registry.registerModelType(
            "qwen3",
            matches: { data in
                guard
                    let object = try? JSONSerialization.jsonObject(with: data)
                        as? [String: Any],
                    let architectures = object["architectures"] as? [String]
                else { return false }
                return architectures.contains(lfm2DSparkArchitecture)
            },
            creator: { data in
                let configuration = try JSONDecoder.json5().decode(
                    LFM2DSparkConfiguration.self, from: data)
                try configuration.validateModelConfiguration()
                return LFM2DSparkDraftModel(configuration)
            })
    }
}
