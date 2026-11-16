// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers the `muse_glimmer_assistant` DFlash drafter for Muse Glimmer targets.
///
/// Muse Glimmer loads through `VLMModelFactory`, so the drafter goes into the
/// vision-language registry that `MTPDrafterModelFactory.visionLanguage` reads.
/// Callers invoke this once before loading the assistant.
public enum MuseGlimmerAssistantRegistration {
    public static func register() async {
        await register(into: MTPDrafterTypeRegistry.visionLanguage)
    }

    package static func register(into registry: ModelTypeRegistry<any MTPDrafterModel>) async {
        await registry.registerModelType(
            "muse_glimmer_assistant",
            creator: { data in
                MuseGlimmerAssistantDraftModel(
                    try JSONDecoder.json5().decode(
                        MuseGlimmerAssistantConfiguration.self, from: data))
            })
    }
}
