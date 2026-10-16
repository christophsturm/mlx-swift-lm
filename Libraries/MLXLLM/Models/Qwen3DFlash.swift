// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Configuration shared by Qwen-style DFlash and DSpark speculative drafters.
public struct Qwen3DFlashConfiguration: Decodable, Sendable {
    /// Projector used by the checkpoint after normalizing Hugging Face's nested config.
    public enum Projector: String, Decodable, Sendable {
        case dflash
        case dflash2
        case dspark
    }

    /// Runtime policy for choosing the verification width between speculative rounds.
    public enum BlockSizePolicy: String, Decodable, Sendable {
        case fixed
        case adaptive
    }

    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let vocabularySize: Int
    public let maximumPositionEmbeddings: Int
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let ropeIsNeoXStyle: Bool
    public let rmsNormEpsilon: Float
    public let hiddenActivation: String
    public let attentionBias: Bool
    public let maskTokenID: Int
    public let targetLayerIDs: [Int]
    public let targetLayerCount: Int
    public let proposalLength: Int
    public let verificationBlockSize: Int
    public let runtimeBlockSize: Int
    public let initialBlockSize: Int
    public let blockSizePolicy: BlockSizePolicy
    public let projector: Projector
    public let markovRank: Int
    public let markovHeadType: String
    public let layerTypes: [String]
    public let slidingWindow: Int?
    public let draftWindowSize: Int?
    public let finalLogitSoftcapping: Float?
    public let enableConfidenceHead: Bool
    public let confidenceHeadWithMarkov: Bool
    public let inputEmbeddingScale: Float
    public let outputMultiplier: Float
    public let convolutionKernelSize: Int
    public let convolutionGroupSize: Int
    public let selectorRank: Int
    public let selectorTopK: Int
    public let isCausal: Bool

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case vocabularySize = "vocab_size"
        case maximumPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeIsNeoXStyle = "rope_is_neox_style"
        case rmsNormEpsilon = "rms_norm_eps"
        case hiddenActivation = "hidden_act"
        case attentionBias = "attention_bias"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case draftWindowSize = "draft_window_size"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case blockSize = "block_size"
        case runtimeBlockSize = "runtime_block_size"
        case initialBlockSize = "dflash_initial_block_size"
        case blockSizePolicy = "block_size_policy"
        case targetLayerCount = "num_target_layers"
        case markovRank = "markov_rank"
        case markovHeadType = "markov_head_type"
        case enableConfidenceHead = "enable_confidence_head"
        case confidenceHeadWithMarkov = "confidence_head_with_markov"
        case inputEmbeddingScale = "input_embedding_scale"
        case outputMultiplier = "output_multiplier"
        case convolutionKernelSize = "conv_kernel_size"
        case convolutionGroupSize = "conv_group_size"
        case selectorRank = "selector_rank"
        case selectorTopK = "selector_top_k"
        case isCausal = "is_causal"
        case dflash = "dflash_config"
    }

    private enum DFlashCodingKeys: String, CodingKey {
        case projector = "projector_type"
        case blockSize = "block_size"
        case runtimeBlockSize = "runtime_block_size"
        case initialBlockSize = "dflash_initial_block_size"
        case blockSizePolicy = "block_size_policy"
        case maskTokenID = "mask_token_id"
        case targetLayerIDs = "target_layer_ids"
        case targetLayerCount = "num_target_layers"
        case markovRank = "markov_rank"
        case markovHeadType = "markov_head_type"
        case draftWindowSize = "draft_window_size"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case enableConfidenceHead = "enable_confidence_head"
        case confidenceHeadWithMarkov = "confidence_head_with_markov"
        case inputEmbeddingScale = "input_embedding_scale"
        case outputMultiplier = "output_multiplier"
        case convolutionKernelSize = "conv_kernel_size"
        case convolutionGroupSize = "conv_group_size"
        case selectorRank = "selector_rank"
        case selectorTopK = "selector_top_k"
    }

    /// Decodes the published checkpoint schema and normalizes proposal count to verification width.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let dflash = try container.nestedContainer(keyedBy: DFlashCodingKeys.self, forKey: .dflash)
        let decodedRopeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        let rootRopeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        keyValueHeads = try container.decode(Int.self, forKey: .keyValueHeads)
        headDimension = try container.decode(Int.self, forKey: .headDimension)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        maximumPositionEmbeddings = try container.decode(
            Int.self, forKey: .maximumPositionEmbeddings)
        ropeTheta =
            decodedRopeScaling?["rope_theta"]?.asFloat()
            ?? rootRopeTheta
            ?? 10_000_000
        ropeScaling = decodedRopeScaling
        ropeIsNeoXStyle =
            try container.decodeIfPresent(Bool.self, forKey: .ropeIsNeoXStyle)
            ?? true
        rmsNormEpsilon = try container.decodeIfPresent(Float.self, forKey: .rmsNormEpsilon) ?? 1e-6
        hiddenActivation =
            try container.decodeIfPresent(String.self, forKey: .hiddenActivation)
            ?? "silu"
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        let explicitProjector = try dflash.decodeIfPresent(Projector.self, forKey: .projector)
        let hasDFlash2Shape =
            dflash.contains(.convolutionKernelSize)
            && dflash.contains(.convolutionGroupSize)
            && dflash.contains(.selectorRank)
            && dflash.contains(.selectorTopK)
        projector = explicitProjector ?? (hasDFlash2Shape ? .dflash2 : .dflash)
        let publishedBlockSize =
            try dflash.decodeIfPresent(Int.self, forKey: .blockSize)
            ?? container.decode(Int.self, forKey: .blockSize)
        if projector == .dspark {
            proposalLength = publishedBlockSize
            verificationBlockSize = publishedBlockSize + 1
        } else {
            proposalLength = publishedBlockSize - 1
            verificationBlockSize = publishedBlockSize
        }
        let defaultRuntimeBlockSize =
            switch projector {
            case .dflash: verificationBlockSize
            case .dflash2: min(5, verificationBlockSize)
            case .dspark: min(8, verificationBlockSize)
            }
        runtimeBlockSize =
            try dflash.decodeIfPresent(Int.self, forKey: .runtimeBlockSize)
            ?? container.decodeIfPresent(Int.self, forKey: .runtimeBlockSize)
            ?? defaultRuntimeBlockSize
        blockSizePolicy =
            try dflash.decodeIfPresent(BlockSizePolicy.self, forKey: .blockSizePolicy)
            ?? container.decodeIfPresent(BlockSizePolicy.self, forKey: .blockSizePolicy)
            ?? (projector == .dflash ? .fixed : .adaptive)
        let defaultInitialBlockSize =
            switch projector {
            case .dflash: runtimeBlockSize
            case .dflash2: min(3, runtimeBlockSize)
            case .dspark: min(4, runtimeBlockSize)
            }
        initialBlockSize =
            try dflash.decodeIfPresent(Int.self, forKey: .initialBlockSize)
            ?? container.decodeIfPresent(Int.self, forKey: .initialBlockSize)
            ?? (blockSizePolicy == .adaptive ? defaultInitialBlockSize : runtimeBlockSize)
        maskTokenID = try dflash.decode(Int.self, forKey: .maskTokenID)
        targetLayerIDs = try dflash.decode([Int].self, forKey: .targetLayerIDs)
        targetLayerCount =
            try dflash.decodeIfPresent(Int.self, forKey: .targetLayerCount)
            ?? container.decode(Int.self, forKey: .targetLayerCount)
        markovRank =
            try dflash.decodeIfPresent(Int.self, forKey: .markovRank)
            ?? container.decodeIfPresent(Int.self, forKey: .markovRank)
            ?? 256
        markovHeadType =
            try dflash.decodeIfPresent(String.self, forKey: .markovHeadType)
            ?? container.decodeIfPresent(String.self, forKey: .markovHeadType)
            ?? "vanilla"
        layerTypes =
            try container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: hiddenLayers)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        draftWindowSize =
            try dflash.decodeIfPresent(Int.self, forKey: .draftWindowSize)
            ?? container.decodeIfPresent(Int.self, forKey: .draftWindowSize)
        finalLogitSoftcapping =
            try dflash.decodeIfPresent(
                Float.self, forKey: .finalLogitSoftcapping)
            ?? container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        enableConfidenceHead =
            try dflash.decodeIfPresent(Bool.self, forKey: .enableConfidenceHead)
            ?? container.decodeIfPresent(Bool.self, forKey: .enableConfidenceHead)
            ?? true
        confidenceHeadWithMarkov =
            try dflash.decodeIfPresent(
                Bool.self, forKey: .confidenceHeadWithMarkov)
            ?? container.decodeIfPresent(Bool.self, forKey: .confidenceHeadWithMarkov)
            ?? true
        inputEmbeddingScale =
            try dflash.decodeIfPresent(
                Float.self, forKey: .inputEmbeddingScale)
            ?? container.decodeIfPresent(Float.self, forKey: .inputEmbeddingScale)
            ?? 1
        outputMultiplier =
            try dflash.decodeIfPresent(Float.self, forKey: .outputMultiplier)
            ?? container.decodeIfPresent(Float.self, forKey: .outputMultiplier)
            ?? 1
        convolutionKernelSize =
            try dflash.decodeIfPresent(
                Int.self, forKey: .convolutionKernelSize)
            ?? container.decodeIfPresent(Int.self, forKey: .convolutionKernelSize)
            ?? 0
        convolutionGroupSize =
            try dflash.decodeIfPresent(
                Int.self, forKey: .convolutionGroupSize)
            ?? container.decodeIfPresent(Int.self, forKey: .convolutionGroupSize)
            ?? 0
        selectorRank =
            try dflash.decodeIfPresent(Int.self, forKey: .selectorRank)
            ?? container.decodeIfPresent(Int.self, forKey: .selectorRank)
            ?? 0
        selectorTopK =
            try dflash.decodeIfPresent(Int.self, forKey: .selectorTopK)
            ?? container.decodeIfPresent(Int.self, forKey: .selectorTopK)
            ?? 0
        isCausal = try container.decodeIfPresent(Bool.self, forKey: .isCausal) ?? false

        try validate()
    }

    /// Rejects checkpoint metadata that cannot describe a compatible target and draft graph.
    public func validate() throws {
        let positiveFields = [
            "hidden_size": hiddenSize,
            "intermediate_size": intermediateSize,
            "num_hidden_layers": hiddenLayers,
            "num_attention_heads": attentionHeads,
            "num_key_value_heads": keyValueHeads,
            "head_dim": headDimension,
            "vocab_size": vocabularySize,
            "max_position_embeddings": maximumPositionEmbeddings,
            "block_size": verificationBlockSize,
            "proposal_length": proposalLength,
            "num_target_layers": targetLayerCount,
            "markov_rank": markovRank,
        ]
        if let invalid = positiveFields.first(where: { $0.value <= 0 }) {
            throw Qwen3DFlashConfigurationError.nonPositiveField(invalid.key, invalid.value)
        }
        guard verificationBlockSize == proposalLength + 1 else {
            throw Qwen3DFlashConfigurationError.invalidVerificationBlockSize(
                verificationBlockSize, proposalLength: proposalLength)
        }
        guard (2 ... verificationBlockSize).contains(runtimeBlockSize) else {
            throw Qwen3DFlashConfigurationError.invalidRuntimeBlockSize(
                runtimeBlockSize, maximum: verificationBlockSize)
        }
        guard (2 ... runtimeBlockSize).contains(initialBlockSize) else {
            throw Qwen3DFlashConfigurationError.invalidInitialBlockSize(
                initialBlockSize, maximum: runtimeBlockSize)
        }
        guard attentionHeads.isMultiple(of: keyValueHeads) else {
            throw Qwen3DFlashConfigurationError.invalidAttentionDimensions
        }
        if projector != .dflash2, hiddenSize != attentionHeads * headDimension {
            throw Qwen3DFlashConfigurationError.invalidAttentionDimensions
        }
        let supportedLayerTypes =
            projector == .dspark
            ? layerTypes.allSatisfy { $0 == "full_attention" }
            : layerTypes.allSatisfy { $0 == "full_attention" || $0 == "sliding_attention" }
        guard layerTypes.count == hiddenLayers, supportedLayerTypes else {
            throw Qwen3DFlashConfigurationError.unsupportedLayerTypes(layerTypes)
        }
        if layerTypes.contains("sliding_attention"), (slidingWindow ?? 0) <= 1 {
            throw Qwen3DFlashConfigurationError.missingSlidingWindow
        }
        if let draftWindowSize, draftWindowSize <= 0 {
            throw Qwen3DFlashConfigurationError.invalidDraftWindowSize(draftWindowSize)
        }
        guard hiddenActivation == "silu" else {
            throw Qwen3DFlashConfigurationError.unsupportedHiddenActivation(hiddenActivation)
        }
        guard !attentionBias else {
            throw Qwen3DFlashConfigurationError.unsupportedAttentionBias
        }
        if projector == .dspark, markovHeadType != "vanilla" {
            throw Qwen3DFlashConfigurationError.unsupportedMarkovHeadType(markovHeadType)
        }
        if projector == .dflash2 {
            let positiveDFlash2Fields = [
                "conv_kernel_size": convolutionKernelSize,
                "conv_group_size": convolutionGroupSize,
                "selector_rank": selectorRank,
                "selector_top_k": selectorTopK,
            ]
            if let invalid = positiveDFlash2Fields.first(where: { $0.value <= 0 }) {
                throw Qwen3DFlashConfigurationError.nonPositiveField(invalid.key, invalid.value)
            }
            guard hiddenSize.isMultiple(of: convolutionGroupSize) else {
                throw Qwen3DFlashConfigurationError.invalidConvolutionGroupSize(
                    convolutionGroupSize, hiddenSize: hiddenSize)
            }
            guard selectorTopK <= vocabularySize else {
                throw Qwen3DFlashConfigurationError.invalidSelectorTopK(
                    selectorTopK, vocabularySize: vocabularySize)
            }
            guard !isCausal else {
                throw Qwen3DFlashConfigurationError.unsupportedCausalDraft
            }
        }
        guard (0 ..< vocabularySize).contains(maskTokenID) else {
            throw Qwen3DFlashConfigurationError.invalidMaskTokenID(
                maskTokenID, vocabularySize: vocabularySize)
        }
        let validTargetLayers =
            !targetLayerIDs.isEmpty
            && zip(targetLayerIDs, targetLayerIDs.dropFirst()).allSatisfy { lhs, rhs in
                lhs < rhs
            }
            && targetLayerIDs.allSatisfy { (0 ..< targetLayerCount).contains($0) }
        guard validTargetLayers else {
            throw Qwen3DFlashConfigurationError.invalidTargetLayerIDs(
                targetLayerIDs, targetLayerCount: targetLayerCount)
        }
    }
}

/// Structural errors in Qwen-style DFlash and DSpark checkpoint metadata.
public enum Qwen3DFlashConfigurationError: Error, Equatable, Sendable {
    case nonPositiveField(String, Int)
    case invalidVerificationBlockSize(Int, proposalLength: Int)
    case invalidRuntimeBlockSize(Int, maximum: Int)
    case invalidInitialBlockSize(Int, maximum: Int)
    case invalidAttentionDimensions
    case unsupportedLayerTypes([String])
    case missingSlidingWindow
    case invalidDraftWindowSize(Int)
    case unsupportedHiddenActivation(String)
    case unsupportedAttentionBias
    case unsupportedMarkovHeadType(String)
    case invalidConvolutionGroupSize(Int, hiddenSize: Int)
    case invalidSelectorTopK(Int, vocabularySize: Int)
    case unsupportedCausalDraft
    case invalidMaskTokenID(Int, vocabularySize: Int)
    case invalidTargetLayerIDs([Int], targetLayerCount: Int)
}

private final class Qwen3DFlashAttention: Module {
    private let attentionHeads: Int
    private let keyValueHeads: Int
    private let headDimension: Int
    private let scale: Float
    private let slidingWindow: Int?

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    /// Creates one draft layer with the checkpoint's full or sliding attention policy.
    init(_ configuration: Qwen3DFlashConfiguration, layerIndex: Int) {
        attentionHeads = configuration.attentionHeads
        keyValueHeads = configuration.keyValueHeads
        headDimension = configuration.headDimension
        scale = pow(Float(configuration.headDimension), -0.5)
        slidingWindow =
            configuration.layerTypes[layerIndex] == "sliding_attention"
            ? configuration.slidingWindow
            : nil
        _queryProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.attentionHeads * configuration.headDimension,
            bias: false)
        _keyProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.keyValueHeads * configuration.headDimension,
            bias: false)
        _valueProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.keyValueHeads * configuration.headDimension,
            bias: false)
        _outputProjection.wrappedValue = Linear(
            configuration.attentionHeads * configuration.headDimension,
            configuration.hiddenSize,
            bias: false)
        _queryNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension,
            eps: configuration.rmsNormEpsilon)
        _keyNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension,
            eps: configuration.rmsNormEpsilon)
        super.init()
    }

    /// Attends every proposed position over persisted target context and the whole proposal block.
    func callAsFunction(
        _ proposals: MLXArray,
        context originalContext: MLXArray,
        rope: RoPELayer,
        cache: KVCache
    ) -> MLXArray {
        var context = originalContext
        if let slidingWindow {
            let retainedContext = slidingWindow - 1
            if context.dim(1) > retainedContext {
                let skippedContext = context.dim(1) - retainedContext
                context = context[0..., skippedContext..., 0...]
                guard let cache = cache as? BaseKVCache else {
                    preconditionFailure("DFlash sliding attention requires a mutable base cache")
                }
                cache.offset += skippedContext
            }
        }
        let batchSize = proposals.dim(0)
        let proposalLength = proposals.dim(1)
        let contextLength = context.dim(1)
        let contextOffset = cache.offset

        var queries = queryProjection(proposals)
        var contextKeys = keyProjection(context)
        var contextValues = valueProjection(context)
        var proposalKeys = keyProjection(proposals)
        var proposalValues = valueProjection(proposals)

        queries = queryNorm(
            queries.reshaped(batchSize, proposalLength, attentionHeads, headDimension)
        ).transposed(0, 2, 1, 3)
        contextKeys = keyNorm(
            contextKeys.reshaped(batchSize, contextLength, keyValueHeads, headDimension)
        ).transposed(0, 2, 1, 3)
        contextValues = contextValues.reshaped(
            batchSize, contextLength, keyValueHeads, headDimension
        ).transposed(0, 2, 1, 3)
        proposalKeys = keyNorm(
            proposalKeys.reshaped(batchSize, proposalLength, keyValueHeads, headDimension)
        ).transposed(0, 2, 1, 3)
        proposalValues = proposalValues.reshaped(
            batchSize, proposalLength, keyValueHeads, headDimension
        ).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(
            rope, to: queries, offset: .scalar(contextOffset + contextLength))
        contextKeys = applyRotaryPosition(
            rope, to: contextKeys, offset: .scalar(contextOffset))
        proposalKeys = applyRotaryPosition(
            rope, to: proposalKeys, offset: .scalar(contextOffset + contextLength))

        let persisted = cache.update(keys: contextKeys, values: contextValues)
        let keys = concatenated([persisted.0, proposalKeys], axis: 2)
        let values = concatenated([persisted.1, proposalValues], axis: 2)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none)
        return outputProjection(
            attended.transposed(0, 2, 1, 3).reshaped(batchSize, proposalLength, -1))
    }
}

private final class Qwen3DFlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear

    /// Creates the Qwen3 gated feed-forward block used by draft checkpoints.
    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProjection.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _upProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    /// Applies the checkpoint's SwiGLU projection.
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private class Qwen3DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: Qwen3DFlashAttention
    @ModuleInfo fileprivate var mlp: Qwen3DFlashMLP
    @ModuleInfo(key: "input_layernorm") fileprivate var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") fileprivate var postAttentionNorm: RMSNorm

    /// Creates one Qwen3 draft decoder layer.
    init(_ configuration: Qwen3DFlashConfiguration, layerIndex: Int) {
        _attention.wrappedValue = Qwen3DFlashAttention(
            configuration, layerIndex: layerIndex)
        _mlp.wrappedValue = Qwen3DFlashMLP(
            hiddenSize: configuration.hiddenSize,
            intermediateSize: configuration.intermediateSize)
        _inputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon)
        super.init()
    }

    /// Denoises one proposal block against newly verified target context.
    func callAsFunction(
        _ proposals: MLXArray,
        context: MLXArray,
        rope: RoPELayer,
        cache: KVCache
    ) -> MLXArray {
        let attended =
            proposals
            + attention(
                inputNorm(proposals), context: context, rope: rope, cache: cache)
        return attended + mlp(postAttentionNorm(attended))
    }
}

/// Applies one grouped dynamic causal convolution from DFlash2.
func qwen3DFlash2GroupedDynamicConvolution(
    hidden: MLXArray,
    dynamic: MLXArray,
    base: MLXArray,
    groupSize: Int
) -> MLXArray {
    let batchSize = hidden.dim(0)
    let length = hidden.dim(1)
    let hiddenSize = hidden.dim(2)
    let groups = hiddenSize / groupSize
    let blocks = hidden.reshaped(batchSize, length, groups, groupSize)
    let dynamic = dynamic.reshaped(batchSize, length, base.dim(0), groups, 1)
    var output = MLXArray.zeros(like: blocks)
    for offset in 0 ..< base.dim(0) {
        let values: MLXArray
        if offset == 0 {
            values = blocks
        } else {
            let prefixLength = min(offset, length)
            let zeros = MLXArray.zeros(like: blocks[0..., 0 ..< prefixLength, 0..., 0...])
            if offset >= length {
                values = zeros
            } else {
                values = concatenated(
                    [zeros, blocks[0..., 0 ..< (length - offset), 0..., 0...]],
                    axis: 1)
            }
        }
        let kernel = base[offset].reshaped(1, 1, groups, groupSize).asType(hidden.dtype)
        output = output + (kernel + dynamic[0..., 0..., offset, 0..., 0...]) * values
    }
    return output.reshaped(hidden.shape)
}

private final class Qwen3DFlash2GroupedDynamicCausalConv: Module {
    private let kernelSize: Int
    private let groupSize: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    /// Creates one learned dynamic convolution around an attention or MLP branch.
    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        let groups = hiddenSize / groupSize
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(
            hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    /// Produces the pre-branch features and the dynamic post-branch kernel.
    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let groups = hidden.dim(-1) / groupSize
        let dynamic = kernelProjection(hidden).reshaped(
            hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        let prepared = qwen3DFlash2GroupedDynamicConvolution(
            hidden: hidden,
            dynamic: dynamic[0..., 0..., 0, 0..., 0...],
            base: baseKernel[0],
            groupSize: groupSize)
        return (prepared, dynamic[0..., 0..., 1, 0..., 0...])
    }

    /// Applies the post-branch half of the dynamic convolution.
    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        qwen3DFlash2GroupedDynamicConvolution(
            hidden: hidden,
            dynamic: dynamic,
            base: baseKernel[1],
            groupSize: groupSize)
    }
}

private final class Qwen3DFlash2DecoderLayer: Qwen3DFlashDecoderLayer {
    @ModuleInfo(key: "attention_conv") var attentionConvolution:
        Qwen3DFlash2GroupedDynamicCausalConv
    @ModuleInfo(key: "mlp_conv") var mlpConvolution: Qwen3DFlash2GroupedDynamicCausalConv

    /// Creates one DFlash2 layer with dynamic convolutions around both residual branches.
    override init(_ configuration: Qwen3DFlashConfiguration, layerIndex: Int) {
        _attentionConvolution.wrappedValue = Qwen3DFlash2GroupedDynamicCausalConv(
            hiddenSize: configuration.hiddenSize,
            kernelSize: configuration.convolutionKernelSize,
            groupSize: configuration.convolutionGroupSize)
        _mlpConvolution.wrappedValue = Qwen3DFlash2GroupedDynamicCausalConv(
            hiddenSize: configuration.hiddenSize,
            kernelSize: configuration.convolutionKernelSize,
            groupSize: configuration.convolutionGroupSize)
        super.init(configuration, layerIndex: layerIndex)
    }

    /// Denoises a block through DFlash2's dynamically convolved residual paths.
    override func callAsFunction(
        _ proposals: MLXArray,
        context: MLXArray,
        rope: RoPELayer,
        cache: KVCache
    ) -> MLXArray {
        let attentionResidual = proposals
        let (attentionInput, attentionKernel) = attentionConvolution.prepare(
            inputNorm(proposals))
        let attended =
            attentionResidual
            + attentionConvolution.finish(
                attention(
                    attentionInput,
                    context: context,
                    rope: rope,
                    cache: cache),
                dynamic: attentionKernel)
        let mlpResidual = attended
        let (mlpInput, mlpKernel) = mlpConvolution.prepare(postAttentionNorm(attended))
        return mlpResidual + mlpConvolution.finish(mlp(mlpInput), dynamic: mlpKernel)
    }
}

/// Scores a coherent path through the LM head's top candidates at every DFlash2 position.
final class Qwen3DFlash2CandidateSelector: Module {
    private let topK: Int

    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    /// Creates the low-rank transition selector carried by a DFlash2 checkpoint.
    init(_ configuration: Qwen3DFlashConfiguration) {
        topK = configuration.selectorTopK
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.selectorRank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.selectorRank)
        _hiddenProjection.wrappedValue = Linear(
            configuration.hiddenSize, configuration.selectorRank, bias: false)
        super.init()
    }

    /// Selects each successor while conditioning its score on the preceding selected token.
    func select(
        hidden: MLXArray,
        logits: MLXArray,
        anchorTokenIDs: MLXArray,
        sampler: any LogitSampler
    ) -> MLXArray {
        let candidates = argPartition(logits, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projectedHidden = hiddenProjection(hidden)
        var predecessor = anchorTokenIDs.flattened()
        var path = [MLXArray]()
        for position in 0 ..< hidden.dim(1) {
            let edges = sum(
                predecessorCodebook(predecessor)[0..., .newAxis]
                    * projectedHidden[0..., position, .newAxis, 0...]
                    * successorCodebook(candidates[0..., position, 0...]),
                axis: -1)
            let scores = unary[0..., position, 0...] + edges
            let selected = sampler.sample(logits: scores).flattened()
            predecessor = takeAlong(
                candidates[0..., position, 0...],
                selected[0..., .newAxis],
                axis: -1
            ).squeezed(axis: -1)
            path.append(predecessor)
        }
        return stacked(path, axis: 1)
    }
}

private final class Qwen3DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var tokenEmbedding: Embedding
    @ModuleInfo(key: "markov_w2") var outputProjection: Linear

    /// Creates DSpark's low-rank token-transition correction.
    init(vocabularySize: Int, rank: Int) {
        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: vocabularySize, dimensions: rank)
        _outputProjection.wrappedValue = Linear(rank, vocabularySize, bias: false)
        super.init()
    }

    /// Adds the previous token's learned transition logits to one proposal position.
    func callAsFunction(_ tokenIDs: MLXArray) -> MLXArray {
        outputProjection(tokenEmbedding(tokenIDs))
    }
}

private final class Qwen3DSparkConfidenceHead: Module {
    @ModuleInfo var proj: Linear

    /// Creates the optional confidence projection carried by published DSpark weights.
    init(hiddenSize: Int, markovRank: Int, includesMarkov: Bool) {
        _proj.wrappedValue = Linear(
            hiddenSize + (includesMarkov ? markovRank : 0), 1, bias: true)
        super.init()
    }
}

/// Qwen-style DFlash or DSpark drafter loaded from a standalone checkpoint.
public final class Qwen3DFlashDraftModel: Module, StatefulMTPDrafterModel {
    public let configuration: Qwen3DFlashConfiguration
    public var maximumBlockSize: Int? { configuration.runtimeBlockSize }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    public var targetLayerCaptureIDs: [Int] { configuration.targetLayerIDs }
    public let consumesTargetHiddenHistory = true
    public var adaptiveBlockSizeInitial: Int? {
        configuration.blockSizePolicy == .adaptive ? configuration.initialBlockSize : nil
    }
    public var adaptiveBlockSizeMinimum: Int {
        configuration.projector == .dflash2 ? 3 : 4
    }

    @ModuleInfo var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo fileprivate var layers: [Qwen3DFlashDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "markov_head") fileprivate var markovHead: Qwen3DSparkMarkovHead?
    @ModuleInfo(key: "confidence_head") fileprivate var confidenceHead: Qwen3DSparkConfidenceHead?
    @ModuleInfo(key: "candidate_selector") fileprivate var candidateSelector:
        Qwen3DFlash2CandidateSelector?

    private let rope: RoPELayer

    /// Creates the standalone drafter without binding or retaining a target model.
    public init(_ configuration: Qwen3DFlashConfiguration) {
        self.configuration = configuration
        _fc.wrappedValue = Linear(
            configuration.targetLayerIDs.count * configuration.hiddenSize,
            configuration.hiddenSize,
            bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map { layerIndex in
            if configuration.projector == .dflash2 {
                Qwen3DFlash2DecoderLayer(configuration, layerIndex: layerIndex)
            } else {
                Qwen3DFlashDecoderLayer(configuration, layerIndex: layerIndex)
            }
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon)
        if configuration.projector == .dspark {
            _markovHead.wrappedValue = Qwen3DSparkMarkovHead(
                vocabularySize: configuration.vocabularySize,
                rank: configuration.markovRank)
            if configuration.enableConfidenceHead {
                _confidenceHead.wrappedValue = Qwen3DSparkConfidenceHead(
                    hiddenSize: configuration.hiddenSize,
                    markovRank: configuration.markovRank,
                    includesMarkov: configuration.confidenceHeadWithMarkov)
            }
        } else if configuration.projector == .dflash2 {
            _candidateSelector.wrappedValue = Qwen3DFlash2CandidateSelector(configuration)
        }
        rope = initializeRope(
            dims: configuration.headDimension,
            base: configuration.ropeTheta,
            traditional: !configuration.ropeIsNeoXStyle,
            scalingConfig: configuration.ropeScaling,
            maxPositionEmbeddings: configuration.maximumPositionEmbeddings)
        super.init()
    }

    public func isCompatible(with target: any LanguageModel) -> Bool {
        (try? requireCompatibility(with: target)) != nil
    }

    /// Verifies the target dimensions encoded in the standalone checkpoint.
    public func requireCompatibility(with target: any LanguageModel) throws {
        guard let target = target as? any DFlashTargetModel else {
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

    /// Creates private draft caches using the checkpoint's global or per-layer window policy.
    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        let cache = configuration.layerTypes.map { layerType -> any KVCache in
            if let draftWindowSize = configuration.draftWindowSize {
                return RotatingKVCache(maxSize: draftWindowSize)
            }
            if layerType == "sliding_attention" {
                return RotatingKVCache(maxSize: configuration.slidingWindow! - 1)
            }
            return KVCacheSimple()
        }
        return MTPDrafterState(cache: cache)
    }

    /// Stateless compatibility entry point backed by a temporary stream state.
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

    /// Denoises a whole DFlash block or samples DSpark's Markov-corrected positions.
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
            preconditionFailure("Qwen3DFlashDraftModel requires a DFlashTargetModel")
        }
        precondition(state.cache.count == layers.count)
        let batchSize = lastToken.size
        let anchor = lastToken.flattened().reshaped(batchSize, 1).asType(.int32)
        let proposalCount = blockSize - 1
        let inputLength = configuration.projector == .dspark ? proposalCount : blockSize
        let maskCount = inputLength - 1
        let masks = MLXArray.full(
            [batchSize, maskCount],
            values: MLXArray(configuration.maskTokenID),
            dtype: .int32)
        let inputs = concatenated([anchor, masks], axis: 1)

        var hiddenStates =
            target.dflashTokenEmbeddings(inputs)
            * configuration.inputEmbeddingScale
        let context = hiddenNorm(fc(lastHidden))
        for (layer, cache) in zip(layers, state.cache) {
            hiddenStates = layer(
                hiddenStates, context: context, rope: rope, cache: cache)
        }
        hiddenStates = norm(hiddenStates)
        state.nextPosition += lastHidden.dim(1)
        state.proposalAppended = 0

        if configuration.projector == .dspark {
            guard let markovHead else {
                preconditionFailure("DSpark checkpoint is missing its Markov head")
            }
            let baseLogits = projectLogits(hiddenStates, target: target)
            var previous = anchor.flattened()
            var proposals = [MLXArray]()
            for position in 0 ..< proposalCount {
                let logits = baseLogits[0..., position, 0...] + markovHead(previous)
                previous = sampler.sample(logits: logits).flattened()
                proposals.append(previous[0..., .newAxis])
            }
            return concatenated(proposals, axis: 1).asType(.int32)
        }

        let proposalHidden = hiddenStates[0..., 1..., 0...]
        let logits = projectLogits(proposalHidden, target: target)
        if configuration.projector == .dflash2 {
            guard let candidateSelector else {
                preconditionFailure("DFlash2 checkpoint is missing its candidate selector")
            }
            return candidateSelector.select(
                hidden: proposalHidden,
                logits: logits,
                anchorTokenIDs: anchor,
                sampler: sampler
            ).asType(.int32)
        }
        return sampler.sample(logits: logits).asType(.int32)
    }

    /// Keeps the private draft cache unchanged because it stores only committed target context.
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

    /// Normalizes standalone checkpoint paths to this module's root.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(
            uniqueKeysWithValues: weights.map { key, value in
                var key = key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key
                if configuration.projector == .dflash2,
                    key == "candidate_selector.predecessor_codebook"
                        || key == "candidate_selector.successor_codebook"
                {
                    key += ".weight"
                }
                return (key, value)
            })
    }

    /// Applies the target LM head and the checkpoint's optional logit soft cap.
    private func projectLogits(
        _ hiddenStates: MLXArray,
        target: any DFlashTargetModel
    ) -> MLXArray {
        var logits = target.dflashLogits(hiddenStates) * configuration.outputMultiplier
        if let softcap = configuration.finalLogitSoftcapping {
            logits = tanh(logits / softcap) * softcap
        }
        return logits
    }
}
