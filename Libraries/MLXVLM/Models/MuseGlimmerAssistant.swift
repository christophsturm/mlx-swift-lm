// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of mlx_vlm/speculative/drafters/muse_glimmer_assistant, the DFlash drafter
// published as meta-models/Muse-Glimmer-30B-assistant.

/// Configuration of the published Muse Glimmer DFlash assistant checkpoint.
public struct MuseGlimmerAssistantConfiguration: Decodable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let rmsNormEpsilon: Float
    public let maximumPositionEmbeddings: Int
    public let ropeParameters: [String: StringOrNumber]
    public let layerTypes: [String]

    /// Attention reaches keys at most this many positions away from a query.
    public let slidingWindow: Int

    /// Verification width the assistant was trained for: one anchor plus its drafts.
    public let blockSize: Int
    public let maskTokenID: Int

    /// Target decoder layers whose residuals condition the assistant, one per assistant layer.
    public let targetLayerIDs: [Int]

    /// Decoder layers of the target. The published config omits it; the reference
    /// supplies Muse Glimmer 30B's 52.
    public let targetLayerCount: Int

    /// Vocabulary shared with the target. The published config omits it; the reference
    /// supplies Muse Glimmer 30B's 202048.
    public let vocabularySize: Int

    public var ropeTheta: Float { ropeParameters["rope_theta"]?.asFloat() ?? 500_000 }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case rmsNormEpsilon = "rms_norm_eps"
        case maximumPositionEmbeddings = "max_position_embeddings"
        case ropeParameters = "rope_parameters"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case blockSize = "block_size"
        case maskTokenID = "mask_token_id"
        case targetLayerIDs = "target_layer_ids"
        case targetLayerCount = "num_target_layers"
        case vocabularySize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        keyValueHeads = try container.decode(Int.self, forKey: .keyValueHeads)
        headDimension = try container.decode(Int.self, forKey: .headDimension)
        rmsNormEpsilon = try container.decode(Float.self, forKey: .rmsNormEpsilon)
        maximumPositionEmbeddings = try container.decode(
            Int.self, forKey: .maximumPositionEmbeddings)
        ropeParameters = try container.decode(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        blockSize = try container.decode(Int.self, forKey: .blockSize)
        maskTokenID = try container.decode(Int.self, forKey: .maskTokenID)
        targetLayerIDs = try container.decode([Int].self, forKey: .targetLayerIDs)
        targetLayerCount =
            try container.decodeIfPresent(Int.self, forKey: .targetLayerCount) ?? 52
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 202_048
        try validate()
    }

    /// Rejects what the reference's `MuseGlimmerAssistantConfig.validate` rejects.
    public func validate() throws {
        guard modelType == "muse_glimmer_assistant" else {
            throw MuseGlimmerAssistantConfigurationError.unsupportedModelType(modelType)
        }
        let positiveFields = [
            ("hidden_size", hiddenSize),
            ("intermediate_size", intermediateSize),
            ("num_hidden_layers", hiddenLayers),
            ("num_attention_heads", attentionHeads),
            ("num_key_value_heads", keyValueHeads),
            ("head_dim", headDimension),
            ("max_position_embeddings", maximumPositionEmbeddings),
            ("sliding_window", slidingWindow),
            ("block_size", blockSize),
            ("num_target_layers", targetLayerCount),
            ("vocab_size", vocabularySize),
        ]
        if let invalid = positiveFields.first(where: { $0.1 <= 0 }) {
            throw MuseGlimmerAssistantConfigurationError.nonPositiveField(invalid.0, invalid.1)
        }
        guard attentionHeads.isMultiple(of: keyValueHeads) else {
            throw MuseGlimmerAssistantConfigurationError.invalidAttentionHeads(
                attentionHeads, keyValueHeads: keyValueHeads)
        }
        guard (0 ..< vocabularySize).contains(maskTokenID) else {
            throw MuseGlimmerAssistantConfigurationError.invalidMaskTokenID(
                maskTokenID, vocabularySize: vocabularySize)
        }
        guard layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ $0 == "sliding_attention" })
        else {
            throw MuseGlimmerAssistantConfigurationError.unsupportedLayerTypes(layerTypes)
        }
        let validTargetLayers =
            targetLayerIDs.count == hiddenLayers
            && zip(targetLayerIDs, targetLayerIDs.dropFirst()).allSatisfy { $0 < $1 }
            && targetLayerIDs.allSatisfy { (0 ..< targetLayerCount).contains($0) }
        guard validTargetLayers else {
            throw MuseGlimmerAssistantConfigurationError.invalidTargetLayerIDs(
                targetLayerIDs, targetLayerCount: targetLayerCount)
        }
    }
}

private final class MuseGlimmerAssistantAttention: Module {
    private let attentionHeads: Int
    private let keyValueHeads: Int
    private let headDimension: Int
    private let scale: Float
    private let slidingWindow: Int

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    init(_ configuration: MuseGlimmerAssistantConfiguration) {
        attentionHeads = configuration.attentionHeads
        keyValueHeads = configuration.keyValueHeads
        headDimension = configuration.headDimension
        scale = pow(Float(configuration.headDimension), -0.5)
        slidingWindow = configuration.slidingWindow
        let queryWidth = configuration.attentionHeads * configuration.headDimension
        let keyWidth = configuration.keyValueHeads * configuration.headDimension
        _queryProjection.wrappedValue = Linear(
            configuration.hiddenSize, queryWidth, bias: false)
        _keyProjection.wrappedValue = Linear(configuration.hiddenSize, keyWidth, bias: false)
        _valueProjection.wrappedValue = Linear(configuration.hiddenSize, keyWidth, bias: false)
        _outputProjection.wrappedValue = Linear(
            queryWidth, configuration.hiddenSize, bias: false)
        _queryNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension, eps: configuration.rmsNormEpsilon)
        _keyNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension, eps: configuration.rmsNormEpsilon)
        super.init()
    }

    /// Persists the new target context in `cache` and attends the proposal block over it.
    func callAsFunction(
        _ proposals: MLXArray,
        context originalContext: MLXArray,
        rope: RoPELayer,
        cache: RotatingKVCache
    ) -> MLXArray {
        var context = originalContext
        if context.dim(1) > slidingWindow {
            let skipped = context.dim(1) - slidingWindow
            context = context[0..., skipped..., 0...]
            cache.offset += skipped
        }
        let batchSize = proposals.dim(0)
        let proposalLength = proposals.dim(1)
        let contextLength = context.dim(1)
        let contextOffset = cache.offset

        let both = concatenated([context, proposals], axis: 1)
        var queries = queryNorm(
            queryProjection(proposals).reshaped(
                batchSize, proposalLength, attentionHeads, headDimension)
        ).transposed(0, 2, 1, 3)
        var keys = keyNorm(
            keyProjection(both).reshaped(
                batchSize, contextLength + proposalLength, keyValueHeads, headDimension)
        ).transposed(0, 2, 1, 3)
        let values = valueProjection(both).reshaped(
            batchSize, contextLength + proposalLength, keyValueHeads, headDimension
        ).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(
            rope, to: queries, offset: .scalar(contextOffset + contextLength))
        keys = applyRotaryPosition(rope, to: keys, offset: .scalar(contextOffset))

        // A one-row update returns the ring in storage order, so the attention reads the
        // retained context back in temporal order, where row positions are known.
        _ = cache.update(
            keys: keys[.ellipsis, ..<contextLength, 0...],
            values: values[.ellipsis, ..<contextLength, 0...])
        guard let (cachedKeys, cachedValues) = cache.logicalView(tail: .max) else {
            preconditionFailure("The assistant cache holds no context after an update")
        }
        let queryStart = cache.offset
        let queryPositions = MLXArray(queryStart ..< (queryStart + proposalLength))
        let keyPositions = MLXArray(
            (queryStart - cachedKeys.dim(2)) ..< (queryStart + proposalLength))
        let visible =
            abs(
                queryPositions.expandedDimensions(axis: 1)
                    - keyPositions.expandedDimensions(axis: 0))
            .<= slidingWindow
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: concatenated([cachedKeys, keys[.ellipsis, contextLength..., 0...]], axis: 2),
            values: concatenated(
                [cachedValues, values[.ellipsis, contextLength..., 0...]], axis: 2),
            scale: scale,
            mask: .array(visible))
        return outputProjection(
            attended.transposed(0, 2, 1, 3).reshaped(batchSize, proposalLength, -1))
    }
}

private final class MuseGlimmerAssistantMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(_ configuration: MuseGlimmerAssistantConfiguration) {
        _gateProjection.wrappedValue = Linear(
            configuration.hiddenSize, configuration.intermediateSize, bias: false)
        _upProjection.wrappedValue = Linear(
            configuration.hiddenSize, configuration.intermediateSize, bias: false)
        _downProjection.wrappedValue = Linear(
            configuration.intermediateSize, configuration.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private final class MuseGlimmerAssistantDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MuseGlimmerAssistantAttention
    @ModuleInfo var mlp: MuseGlimmerAssistantMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(_ configuration: MuseGlimmerAssistantConfiguration) {
        _attention.wrappedValue = MuseGlimmerAssistantAttention(configuration)
        _mlp.wrappedValue = MuseGlimmerAssistantMLP(configuration)
        _inputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        super.init()
    }

    func callAsFunction(
        _ proposals: MLXArray, context: MLXArray, rope: RoPELayer, cache: RotatingKVCache
    ) -> MLXArray {
        let attended =
            proposals + attention(inputNorm(proposals), context: context, rope: rope, cache: cache)
        return attended + mlp(postAttentionNorm(attended))
    }
}

/// Projects the concatenated target residuals to the assistant's width.
private final class MuseGlimmerAssistantContextProjection: Module {
    @ModuleInfo var fc: Linear
    @ModuleInfo(key: "output_norm_enc") var outputNorm: RMSNorm

    init(_ configuration: MuseGlimmerAssistantConfiguration) {
        _fc.wrappedValue = Linear(
            configuration.targetLayerIDs.count * configuration.hiddenSize,
            configuration.hiddenSize, bias: false)
        _outputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        super.init()
    }

    func callAsFunction(_ targetHidden: MLXArray) -> MLXArray {
        outputNorm(fc(targetHidden))
    }
}

/// The Muse Glimmer DFlash assistant: denoises a block of mask tokens after the verified
/// anchor, conditioned on target residuals it keeps in a sliding-window cache.
///
/// It borrows the target's raw token embeddings and raw output head through
/// ``DFlashTargetModel`` and retains nothing from the target.
public final class MuseGlimmerAssistantDraftModel: Module, StatefulMTPDrafterModel {
    public let configuration: MuseGlimmerAssistantConfiguration
    public var maximumBlockSize: Int? { configuration.blockSize }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    public var targetLayerIds: [Int]? { configuration.targetLayerIDs }
    public let consumesTargetHiddenHistory = true
    /// The reference starts adaptive verification at four positions.
    public let adaptiveBlockSizeInitial: Int? = 4

    @ModuleInfo fileprivate var encoder: MuseGlimmerAssistantContextProjection
    @ModuleInfo fileprivate var layers: [MuseGlimmerAssistantDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    private let rope: RoPELayer

    public init(_ configuration: MuseGlimmerAssistantConfiguration) {
        self.configuration = configuration
        _encoder.wrappedValue = MuseGlimmerAssistantContextProjection(configuration)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map { _ in
            MuseGlimmerAssistantDecoderLayer(configuration)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        rope = initializeRope(
            dims: configuration.headDimension,
            base: configuration.ropeTheta,
            traditional: false,
            scalingConfig: configuration.ropeParameters,
            maxPositionEmbeddings: configuration.maximumPositionEmbeddings)
        super.init()
    }

    /// Requires a Muse Glimmer target with the checkpoint's width, depth, and vocabulary.
    public func validateCompatibility(with target: any LanguageModel) throws {
        guard let target = target as? MuseGlimmer else {
            throw DFlashTargetCompatibilityError.unsupportedTarget(
                String(describing: type(of: target)))
        }
        guard target.dflashHiddenSize == configuration.hiddenSize else {
            throw DFlashTargetCompatibilityError.hiddenSize(
                drafter: configuration.hiddenSize, target: target.dflashHiddenSize)
        }
        guard target.dflashTargetLayerCount == configuration.targetLayerCount else {
            throw DFlashTargetCompatibilityError.layerCount(
                drafter: configuration.targetLayerCount, target: target.dflashTargetLayerCount)
        }
        guard target.dflashVocabularySize == configuration.vocabularySize else {
            throw DFlashTargetCompatibilityError.vocabularySize(
                drafter: configuration.vocabularySize, target: target.dflashVocabularySize)
        }
    }

    /// One cache per layer holding the projected target context of the attention window.
    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(
            cache: layers.map { _ in RotatingKVCache(maxSize: configuration.slidingWindow) })
    }

    /// Stateless entry point backed by a temporary stream state.
    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden,
            sharedKV: sharedKV, positionDeltas: positionDeltas,
            queryOffset: queryOffset, blockSize: blockSize,
            state: &state, sampler: sampler)
    }

    /// Appends `lastHidden` to the context and samples `blockSize - 1` drafts after the anchor.
    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        guard let target = target as? any DFlashTargetModel else {
            preconditionFailure("MuseGlimmerAssistantDraftModel requires a DFlashTargetModel")
        }
        precondition(state.cache.count == layers.count)
        let batchSize = lastToken.size
        let anchor = lastToken.flattened().reshaped(batchSize, 1).asType(.int32)
        let masks = MLXArray.full(
            [batchSize, blockSize - 1],
            values: MLXArray(Int32(configuration.maskTokenID)),
            dtype: .int32)

        var hiddenStates = target.dflashTokenEmbeddings(concatenated([anchor, masks], axis: 1))
        let context = encoder(lastHidden)
        for (layer, cache) in zip(layers, state.cache) {
            guard let cache = cache as? RotatingKVCache else {
                preconditionFailure("Muse Glimmer assistant layers use rotating caches")
            }
            hiddenStates = layer(hiddenStates, context: context, rope: rope, cache: cache)
        }
        state.nextPosition += lastHidden.dim(1)
        state.proposalAppended = 0

        let logits = target.dflashLogits(norm(hiddenStates)[0..., 1..., 0...])
        return sampler.sample(logits: logits).asType(.int32)
    }

    /// Keeps the context cache unchanged: it only ever holds committed target context.
    public func commitDrafterState(
        target _: any LanguageModel,
        targetHidden _: MLXArray,
        draftTokens _: MLXArray,
        acceptedCount _: Int,
        finalToken _: MLXArray,
        positionDeltas _: MLXArray?,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {}
}

/// Structural errors in Muse Glimmer assistant checkpoint metadata.
public enum MuseGlimmerAssistantConfigurationError: Error, Equatable, Sendable {
    case unsupportedModelType(String)
    case nonPositiveField(String, Int)
    case invalidAttentionHeads(Int, keyValueHeads: Int)
    case invalidMaskTokenID(Int, vocabularySize: Int)
    case unsupportedLayerTypes([String])
    case invalidTargetLayerIDs([Int], targetLayerCount: Int)
}
