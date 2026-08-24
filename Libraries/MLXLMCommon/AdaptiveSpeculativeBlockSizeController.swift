// Copyright © 2026 Apple Inc.

import Foundation

/// Chooses a bounded verification width from recent speculative acceptance.
public struct AdaptiveSpeculativeBlockSizeController: Sendable {
    public let maximumBlockSize: Int
    public let initialBlockSize: Int
    public let minimumBlockSize: Int

    private var rounds: [(accepted: Int, drafted: Int)] = []

    /// Completed acceptance rounds retained when a speculative session is checkpointed.
    package var checkpointRounds: [(accepted: Int, drafted: Int)] { rounds }

    /// Creates a controller whose widths include the verifier's anchor token.
    public init(maximumBlockSize: Int, initialBlockSize: Int, minimumBlockSize: Int = 4) {
        precondition(maximumBlockSize >= 2)
        precondition((2 ... maximumBlockSize).contains(initialBlockSize))
        precondition(minimumBlockSize >= 2)
        self.maximumBlockSize = maximumBlockSize
        self.initialBlockSize = initialBlockSize
        self.minimumBlockSize = min(minimumBlockSize, maximumBlockSize)
    }

    /// Records one completed round for the next width decision.
    public mutating func record(accepted: Int, drafted: Int) {
        precondition(drafted > 0)
        precondition((0 ... drafted).contains(accepted))
        rounds.append((accepted, drafted))
        if rounds.count > 8 {
            rounds.removeFirst(rounds.count - 8)
        }
    }

    /// Returns the next verification width without exceeding the remaining output budget.
    public func nextBlockSize(remainingTokenBudget: Int) -> Int {
        let availableBlockSize = min(maximumBlockSize, remainingTokenBudget)
        guard availableBlockSize > 1 else {
            return availableBlockSize
        }
        guard let lastRound = rounds.last else {
            return min(availableBlockSize, initialBlockSize)
        }

        let currentBlockSize = min(availableBlockSize, max(2, lastRound.drafted + 1))
        let minimumBlockSize = min(availableBlockSize, self.minimumBlockSize)
        let drafted = rounds.reduce(0) { $0 + $1.drafted }
        let accepted = rounds.reduce(0) { $0 + $1.accepted }
        let acceptanceRate = Double(accepted) / Double(drafted)
        let meanAccepted = Double(accepted) / Double(rounds.count)

        if acceptanceRate < 0.30 || meanAccepted < 2 {
            if currentBlockSize >= 8 {
                return max(minimumBlockSize, min(availableBlockSize, currentBlockSize / 2))
            }
            return max(minimumBlockSize, min(availableBlockSize, currentBlockSize - 2))
        }
        if acceptanceRate < 0.50 {
            return max(minimumBlockSize, min(availableBlockSize, currentBlockSize - 2))
        }

        let fullyAccepted = rounds.count { $0.accepted >= $0.drafted }
        let fullyAcceptedRate = Double(fullyAccepted) / Double(rounds.count)
        if acceptanceRate >= 0.85, fullyAcceptedRate >= 0.75 {
            return min(availableBlockSize, currentBlockSize + 2)
        }
        return min(availableBlockSize, currentBlockSize)
    }
}
