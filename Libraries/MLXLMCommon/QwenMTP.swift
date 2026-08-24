// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Normalizes embedded and standalone Qwen MTP checkpoints to the runtime module tree.
package func qwenMTPSanitizeWeights(
    weights: [String: MLXArray],
    mtpNumHiddenLayers: Int,
    numExperts: Int,
    shiftNormWeights: Bool
) -> [String: MLXArray] {
    let standalonePrefixes = [
        "embed_tokens.",
        "fc.",
        "layers.",
        "norm.",
        "pre_fc_norm_embedding.",
        "pre_fc_norm_hidden.",
    ]
    let hasEmbeddedTopology = weights.keys.contains { $0.hasPrefix("mtp.") }
    let hasStandaloneTopology = weights.keys.contains { key in
        standalonePrefixes.contains { key.hasPrefix($0) }
    }
    precondition(
        !(hasEmbeddedTopology && hasStandaloneTopology),
        "Qwen MTP checkpoint mixes embedded and standalone predictor weights")

    var sanitized = weights.reduce(into: [String: MLXArray]()) { result, entry in
        if hasEmbeddedTopology, entry.key.hasPrefix("mtp.") {
            result[entry.key] = entry.value
        } else if !hasEmbeddedTopology,
            standalonePrefixes.contains(where: { entry.key.hasPrefix($0) })
        {
            result["mtp.\(entry.key)"] = entry.value
        }
    }

    for layer in 0 ..< max(mtpNumHiddenLayers, 1) {
        let prefix = "mtp.layers.\(layer).mlp"
        if numExperts > 0,
            sanitized["\(prefix).experts.0.up_proj.weight"] != nil
        {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let toJoin = (0 ..< numExperts).map { expert in
                    sanitized.removeValue(
                        forKey: "\(prefix).experts.\(expert).\(projection).weight")!
                }
                sanitized["\(prefix).switch_mlp.\(projection).weight"] = MLX.stacked(toJoin)
            }
        }

        let gateUpKey = "\(prefix).experts.gate_up_proj"
        if let gateUp = sanitized.removeValue(forKey: gateUpKey) {
            let mid = gateUp.dim(-2) / 2
            sanitized["\(prefix).switch_mlp.gate_proj.weight"] =
                gateUp[.ellipsis, ..<mid, 0...]
            sanitized["\(prefix).switch_mlp.up_proj.weight"] =
                gateUp[.ellipsis, mid..., 0...]
        }

        let downProjKey = "\(prefix).experts.down_proj"
        if let downProj = sanitized.removeValue(forKey: downProjKey) {
            sanitized["\(prefix).switch_mlp.down_proj.weight"] = downProj
        }
    }

    let normKeys = [
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ]

    for key in Array(sanitized.keys) {
        guard let value = sanitized[key] else { continue }
        if shiftNormWeights,
            normKeys.contains(where: { key.hasSuffix($0) }), value.ndim == 1
        {
            sanitized[key] = value + MLXArray(1, dtype: value.dtype)
        }
    }

    return sanitized
}

package func draftMTPTokenBlock(
    targetEmbedTokens: Embedding,
    lmHead: Linear?,
    inputEmbedding: Embedding,
    lastToken: MLXArray,
    lastHidden: MLXArray,
    queryOffset: Int,
    blockSize: Int,
    sampler: any LogitSampler,
    cache: [KVCache],
    forward: (
        _ inputsEmbeds: MLXArray, _ hiddenStates: MLXArray, _ cache: [KVCache],
        _ positionOffset: Int
    ) -> MLXArray
) -> MLXArray {
    precondition(blockSize >= 2, "blockSize must be >= 2")

    var tok = lastToken.ndim == 1 ? lastToken.reshaped([lastToken.dim(0), 1]) : lastToken
    var hidden = lastHidden
    precondition(!cache.isEmpty, "Qwen MTP drafter cache must not be empty")
    var tokens: [MLXArray] = []
    tokens.reserveCapacity(blockSize - 1)

    for stepIndex in 0 ..< (blockSize - 1) {
        let mtpHidden = forward(
            inputEmbedding(tok),
            hidden,
            cache,
            queryOffset + stepIndex
        )
        hidden = mtpHidden

        let logits: MLXArray
        if let lmHead {
            logits = lmHead(mtpHidden)
        } else {
            logits = targetEmbedTokens.asLinear(mtpHidden)
        }

        let next = sampler.sample(logits: logits[0..., -1, 0...])
        tok = next.ndim == 1 ? next.reshaped([next.dim(0), 1]) : next
        tokens.append(tok)
    }

    return concatenated(tokens, axis: 1)
}

/// Completes a stateful Qwen proposal after consuming any seed prepared by the prior round.
package func draftStatefulMTPTokenBlock(
    targetEmbedTokens: Embedding,
    lmHead: Linear?,
    inputEmbedding: Embedding,
    lastToken: MLXArray,
    lastHidden: MLXArray,
    queryOffset: Int,
    blockSize: Int,
    sampler: any LogitSampler,
    state: inout MTPDrafterState,
    forward: (
        _ inputsEmbeds: MLXArray, _ hiddenStates: MLXArray, _ cache: [KVCache],
        _ positionOffset: Int
    ) -> MLXArray
) -> MLXArray {
    precondition(blockSize >= 2, "blockSize must be >= 2")
    precondition(
        (state.seedToken == nil) == (state.seedHidden == nil),
        "Qwen MTP seed token and hidden state must be installed together")

    let seedToken = state.seedToken
    let seedHidden = state.seedHidden
    state.seedToken = nil
    state.seedHidden = nil

    let seededCount = seedToken == nil ? 0 : 1
    state.proposalAppended = blockSize - 1 - seededCount
    guard state.proposalAppended > 0 else {
        return seedToken!
    }

    let tail = draftMTPTokenBlock(
        targetEmbedTokens: targetEmbedTokens,
        lmHead: lmHead,
        inputEmbedding: inputEmbedding,
        lastToken: seedToken ?? lastToken,
        lastHidden: seedHidden ?? lastHidden,
        queryOffset: queryOffset,
        blockSize: state.proposalAppended + 1,
        sampler: sampler,
        cache: state.cache,
        forward: forward)
    state.nextPosition += state.proposalAppended

    if let seedToken {
        return concatenated([normalizedMTPColumn(seedToken), tail], axis: 1)
    }
    return tail
}

/// Restores a Qwen MTP cache to the shifted target position retained by its session.
package func rewindQwenMTPDrafterState(
    toNextPosition nextPosition: Int,
    state: inout MTPDrafterState
) {
    precondition(nextPosition >= 0 && nextPosition <= state.nextPosition)
    let discarded = state.nextPosition - nextPosition
    if discarded > 0 {
        let trimmed = trimPromptCache(state.cache, numTokens: discarded)
        precondition(
            trimmed == discarded,
            "Qwen MTP cache could not discard its unreturned speculative suffix")
    }
    state.nextPosition = nextPosition
    state.seedToken = nil
    state.seedHidden = nil
    state.proposalAppended = 0
}

package func normalizedMTPTokenBatch(_ tokens: MLXArray) -> MLXArray {
    switch tokens.ndim {
    case 1:
        return tokens[.newAxis, 0...]
    default:
        return tokens
    }
}

package func normalizedMTPColumn(_ tokens: MLXArray) -> MLXArray {
    let tokens = normalizedMTPTokenBatch(tokens)
    return tokens.dim(-1) == 1 ? tokens : tokens[0..., (-1)...]
}

package func sampleMTPSeed(
    hidden: MLXArray,
    targetEmbedTokens: Embedding,
    lmHead: Linear?,
    sampler: any LogitSampler
) -> MLXArray {
    let logits = lmHead.map { $0(hidden) } ?? targetEmbedTokens.asLinear(hidden)
    let sampled = sampler.sample(logits: logits[0..., -1, 0...])
    return normalizedMTPColumn(sampled)
}
