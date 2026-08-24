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
