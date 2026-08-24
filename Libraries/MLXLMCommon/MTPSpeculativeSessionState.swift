// Copyright © 2026 Apple Inc.

import Foundation

/// Persistent target and drafter state shared by successive speculative generation passes.
///
/// A chat runtime retains one instance per append-only conversation. The state owns the
/// realized target cache array, target continuation metadata, private drafter cache, and
/// adaptive-width history so a later prompt suffix continues rather than prefilling the
/// transcript again. Callers serialize mutation through their model-container boundary.
public final class MTPSpeculativeSessionState: @unchecked Sendable {
    package let mainCacheStorage: KVCacheStorage
    package var mainState: LMOutput.State?
    package var drafterState: MTPDrafterState?
    package var adaptiveBlockSizeController: AdaptiveSpeculativeBlockSizeController?

    /// Exact target input tokens represented by the cache when this state was created with a ledger.
    public private(set) var tokenHistory: [Int]?

    /// The realized target caches at the authoritative logical position.
    public var cache: [KVCache] { mainCacheStorage.cache }

    /// Number of target input positions represented by ``cache``.
    public var processedTokenCount: Int { mainCacheStorage.processedTokenCount }

    /// Model-specific continuation state aligned with ``processedTokenCount``.
    public var modelState: LMOutput.State? { mainState }

    /// Creates or adopts the cache identity for one speculative conversation.
    ///
    /// - Parameters:
    ///   - mainModel: Target model whose cache topology is used when `mainCache` is absent.
    ///   - drafter: Immutable companion model whose per-stream state is created here.
    ///   - mainCache: Existing target cache identity, or `nil` for a fresh model cache.
    ///   - modelState: Target continuation metadata saved at the same logical position.
    ///   - processedTokenCount: Explicit logical position for restored caches. When absent,
    ///     attention-cache offsets are used as the initial position.
    ///   - tokenHistory: Exact tokens at the restored position. Fresh states create an empty ledger;
    ///     adopted non-empty caches may omit it when their caller does not need append-only history.
    ///   - parameters: Immutable cache policy used by every pass in this conversation.
    public init(
        mainModel: any LanguageModel,
        drafter: any MTPDrafterModel,
        mainCache: [KVCache]? = nil,
        modelState: LMOutput.State? = nil,
        processedTokenCount: Int? = nil,
        tokenHistory: [Int]? = nil,
        parameters: GenerateParameters
    ) throws {
        precondition(processedTokenCount.map { $0 >= 0 } ?? true)
        try drafter.validateTarget(mainModel)
        let plan = try parameters.kvCachePlan()
        let cache = try plan.validated(
            mainCache ?? (try mainModel.newCache(parameters: parameters)))
        let storage = KVCacheStorage(
            cache,
            plan: plan,
            processedTokenCount: processedTokenCount)
        precondition(
            storage.nativeAttentionOffsetsAreAligned,
            "Restored target cache offsets do not match their logical position")
        self.mainCacheStorage = storage
        self.mainState = modelState
        self.drafterState = (drafter as? any StatefulMTPDrafterModel)?
            .makeState(parameters: parameters)
        if let tokenHistory {
            precondition(tokenHistory.count == storage.processedTokenCount)
            self.tokenHistory = tokenHistory
        } else {
            self.tokenHistory = storage.processedTokenCount == 0 ? [] : nil
        }
    }

    /// Records target inputs immediately after their cache timeline commits.
    package func recordCommittedTokens(_ tokens: [Int]) {
        guard var tokenHistory else { return }
        tokenHistory.append(contentsOf: tokens)
        precondition(tokenHistory.count == processedTokenCount)
        self.tokenHistory = tokenHistory
    }

    /// Removes target inputs immediately after the cache timeline rewinds.
    package func discardCommittedTokens(_ count: Int) {
        guard var tokenHistory else { return }
        precondition(count >= 0 && count <= tokenHistory.count)
        tokenHistory.removeLast(count)
        precondition(tokenHistory.count == processedTokenCount)
        self.tokenHistory = tokenHistory
    }
}

/// A restored speculative prompt cache whose target and companion state came from one file.
public struct MTPSpeculativePromptCacheSnapshot: Sendable {
    /// Session state ready to be adopted by a later append-only iterator.
    public let sessionState: MTPSpeculativeSessionState

    /// Caller metadata stored beside the native target and drafter tensors.
    public let metadata: [String: String]

    /// Creates one inseparable restored speculative state and metadata value.
    public init(sessionState: MTPSpeculativeSessionState, metadata: [String: String]) {
        self.sessionState = sessionState
        self.metadata = metadata
    }
}

/// Saves target cache, target continuation state, private drafter cache, and adaptive history together.
///
/// - Parameters:
///   - url: Destination `.safetensors` file.
///   - sessionState: Stable state captured after generation finalization.
///   - metadata: Caller metadata carried through the native prompt-cache file.
public func saveSpeculativePromptCache(
    url: URL,
    sessionState: MTPSpeculativeSessionState,
    metadata: [String: String] = [:]
) throws {
    guard !metadata.keys.contains(speculativeSessionMetadataKey) else {
        throw KVCacheError(message: "Caller metadata uses the speculative session key")
    }
    var mainState = sessionState.mainState
    let emitDrafterState = mainState?[mtpEmitFlagKey]
    let targetLayerCaptureIDs = mainState?[mtpTargetLayerCaptureIDsKey]
    let cacheCheckpointIndex = mainState?[mtpCacheCheckpointIndexKey]
    let sharedKV = mainState?[mtpSharedKVStatesKey] ?? [:]
    let sharedKVOffsets = mainState?[mtpSharedKVOffsetsKey]
    let sharedKVSourceIndices = mainState?[mtpSharedKVSourceIndicesKey]
    mainState?[mtpEmitFlagKey] = nil
    mainState?[mtpTargetLayerCaptureIDsKey] = nil
    mainState?[mtpCacheCheckpointIndexKey] = nil
    mainState?[mtpSharedKVStatesKey] = nil
    mainState?[mtpSharedKVOffsetsKey] = nil
    mainState?[mtpSharedKVSourceIndicesKey] = nil
    var arrays = try mainState?.serializedArrays() ?? [:]

    let sharedKVKeys = sharedKV.keys.sorted()
    for (index, key) in sharedKVKeys.enumerated() {
        guard let value = sharedKV[key] else { continue }
        arrays["\(speculativeStateArrayPrefix)shared.\(index).key"] = value.0
        arrays["\(speculativeStateArrayPrefix)shared.\(index).value"] = value.1
    }

    let drafterMetadata: SpeculativeSessionMetadata.Drafter?
    let drafterCache: [KVCache]
    if let drafterState = sessionState.drafterState {
        drafterCache = drafterState.cache
        if let seedToken = drafterState.seedToken {
            arrays["\(speculativeStateArrayPrefix)drafter.seedToken"] = seedToken
        }
        if let seedHidden = drafterState.seedHidden {
            arrays["\(speculativeStateArrayPrefix)drafter.seedHidden"] = seedHidden
        }
        drafterMetadata = .init(
            cacheCount: drafterState.cache.count,
            nextPosition: drafterState.nextPosition,
            proposalAppended: drafterState.proposalAppended,
            hasSeedToken: drafterState.seedToken != nil,
            hasSeedHidden: drafterState.seedHidden != nil)
    } else {
        drafterCache = []
        drafterMetadata = nil
    }

    let adaptive = sessionState.adaptiveBlockSizeController.map {
        SpeculativeSessionMetadata.Adaptive(
            maximumBlockSize: $0.maximumBlockSize,
            initialBlockSize: $0.initialBlockSize,
            minimumBlockSize: $0.minimumBlockSize,
            rounds: $0.checkpointRounds.map {
                .init(accepted: $0.accepted, drafted: $0.drafted)
            })
    }
    let nativeMetadata = SpeculativeSessionMetadata(
        schemaVersion: 1,
        mainCacheCount: sessionState.cache.count,
        processedTokenCount: sessionState.processedTokenCount,
        tokenHistory: sessionState.tokenHistory,
        emitDrafterState: emitDrafterState,
        targetLayerCaptureIDs: targetLayerCaptureIDs,
        cacheCheckpointIndex: cacheCheckpointIndex,
        sharedKVKeys: sharedKVKeys,
        sharedKVOffsets: sharedKVOffsets,
        sharedKVSourceIndices: sharedKVSourceIndices,
        drafter: drafterMetadata,
        adaptive: adaptive)
    let metadataData = try JSONEncoder().encode(nativeMetadata)
    guard let metadataJSON = String(data: metadataData, encoding: .utf8) else {
        throw KVCacheError(message: "Speculative session metadata is not UTF-8")
    }
    var combinedMetadata = metadata
    combinedMetadata[speculativeSessionMetadataKey] = metadataJSON
    let state = arrays.isEmpty ? nil : LMOutput.State(serializedArrays: arrays)
    try savePromptCache(
        url: url,
        cache: sessionState.cache + drafterCache,
        metadata: combinedMetadata,
        state: state)
}

/// Restores an inseparable target/drafter checkpoint for the exact loaded model pair.
///
/// - Parameters:
///   - url: Source `.safetensors` file.
///   - mainModel: Loaded target model whose cache topology must match the file.
///   - drafter: Loaded companion model whose private cache must match the file.
///   - parameters: Immutable cache policy used when the checkpoint was created.
public func loadSpeculativePromptCacheSnapshot(
    url: URL,
    mainModel: any LanguageModel,
    drafter: any MTPDrafterModel,
    parameters: GenerateParameters
) throws -> MTPSpeculativePromptCacheSnapshot {
    let snapshot = try loadPromptCacheSnapshot(url: url)
    guard let metadataJSON = snapshot.metadata[speculativeSessionMetadataKey],
        let metadataData = metadataJSON.data(using: .utf8)
    else {
        throw KVCacheError(message: "Prompt cache is missing speculative session metadata")
    }
    let metadata = try JSONDecoder().decode(SpeculativeSessionMetadata.self, from: metadataData)
    guard metadata.schemaVersion == 1,
        metadata.mainCacheCount >= 0,
        metadata.mainCacheCount <= snapshot.cache.count,
        metadata.processedTokenCount >= 0
    else {
        throw KVCacheError(message: "Unsupported speculative session metadata")
    }
    let storedState = snapshot.state
    var arrays = try storedState?.serializedArrays() ?? [:]
    let sharedArrays = try metadata.sharedKVKeys.enumerated().map { index, key in
        guard
            let keys = arrays.removeValue(
                forKey: "\(speculativeStateArrayPrefix)shared.\(index).key"),
            let values = arrays.removeValue(
                forKey: "\(speculativeStateArrayPrefix)shared.\(index).value")
        else {
            throw KVCacheError(message: "Speculative shared KV state is incomplete")
        }
        return (key, keys, values)
    }
    let seedToken = arrays.removeValue(forKey: "\(speculativeStateArrayPrefix)drafter.seedToken")
    let seedHidden = arrays.removeValue(forKey: "\(speculativeStateArrayPrefix)drafter.seedHidden")
    var mainState = arrays.isEmpty ? nil : LMOutput.State(serializedArrays: arrays)
    if mainState == nil,
        metadata.emitDrafterState != nil || metadata.targetLayerCaptureIDs != nil
            || metadata.cacheCheckpointIndex != nil || !sharedArrays.isEmpty
            || metadata.sharedKVOffsets != nil || metadata.sharedKVSourceIndices != nil
    {
        mainState = LMOutput.State()
    }
    mainState?[mtpEmitFlagKey] = metadata.emitDrafterState
    mainState?[mtpTargetLayerCaptureIDsKey] = metadata.targetLayerCaptureIDs
    mainState?[mtpCacheCheckpointIndexKey] = metadata.cacheCheckpointIndex
    mainState?[mtpSharedKVOffsetsKey] = metadata.sharedKVOffsets
    mainState?[mtpSharedKVSourceIndicesKey] = metadata.sharedKVSourceIndices
    if !sharedArrays.isEmpty {
        mainState?[mtpSharedKVStatesKey] = Dictionary(
            uniqueKeysWithValues: sharedArrays.map { ($0.0, ($0.1, $0.2)) })
    }

    let mainCache = Array(snapshot.cache.prefix(metadata.mainCacheCount))
    let state = try MTPSpeculativeSessionState(
        mainModel: mainModel,
        drafter: drafter,
        mainCache: mainCache,
        modelState: mainState,
        processedTokenCount: metadata.processedTokenCount,
        tokenHistory: metadata.tokenHistory,
        parameters: parameters)
    let restoredDrafterCache = Array(snapshot.cache.dropFirst(metadata.mainCacheCount))
    if let drafterMetadata = metadata.drafter {
        guard let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            restoredDrafterCache.count == drafterMetadata.cacheCount,
            drafterMetadata.hasSeedToken == (seedToken != nil),
            drafterMetadata.hasSeedHidden == (seedHidden != nil)
        else {
            throw KVCacheError(message: "Speculative drafter state does not match the loaded companion")
        }
        let expectedState = statefulDrafter.makeState(parameters: parameters)
        guard expectedState.cache.count == restoredDrafterCache.count else {
            throw KVCacheError(message: "Speculative drafter cache topology changed")
        }
        state.drafterState = MTPDrafterState(
            cache: restoredDrafterCache,
            nextPosition: drafterMetadata.nextPosition,
            seedToken: seedToken,
            seedHidden: seedHidden,
            proposalAppended: drafterMetadata.proposalAppended)
    } else {
        guard restoredDrafterCache.isEmpty, seedToken == nil, seedHidden == nil else {
            throw KVCacheError(message: "Unexpected speculative drafter state")
        }
        state.drafterState = nil
    }
    if let adaptive = metadata.adaptive {
        var controller = AdaptiveSpeculativeBlockSizeController(
            maximumBlockSize: adaptive.maximumBlockSize,
            initialBlockSize: adaptive.initialBlockSize,
            minimumBlockSize: adaptive.minimumBlockSize)
        for round in adaptive.rounds {
            controller.record(accepted: round.accepted, drafted: round.drafted)
        }
        state.adaptiveBlockSizeController = controller
    }
    var callerMetadata = snapshot.metadata
    callerMetadata.removeValue(forKey: speculativeSessionMetadataKey)
    return MTPSpeculativePromptCacheSnapshot(sessionState: state, metadata: callerMetadata)
}

private let speculativeSessionMetadataKey = "mlx.mtp.speculative-session"
private let speculativeStateArrayPrefix = "mlx.mtp.checkpoint."

private struct SpeculativeSessionMetadata: Codable {
    let schemaVersion: Int
    let mainCacheCount: Int
    let processedTokenCount: Int
    let tokenHistory: [Int]?
    let emitDrafterState: Bool?
    let targetLayerCaptureIDs: [Int]?
    let cacheCheckpointIndex: Int?
    let sharedKVKeys: [String]
    let sharedKVOffsets: [String: Int]?
    let sharedKVSourceIndices: [String: Int]?
    let drafter: Drafter?
    let adaptive: Adaptive?

    struct Drafter: Codable {
        let cacheCount: Int
        let nextPosition: Int
        let proposalAppended: Int
        let hasSeedToken: Bool
        let hasSeedHidden: Bool
    }

    struct Adaptive: Codable {
        let maximumBlockSize: Int
        let initialBlockSize: Int
        let minimumBlockSize: Int
        let rounds: [Round]
    }

    struct Round: Codable {
        let accepted: Int
        let drafted: Int
    }
}
