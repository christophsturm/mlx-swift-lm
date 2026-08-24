import MLXLMCommon
import Testing

@Suite("Adaptive speculative verification width")
struct AdaptiveSpeculativeBlockSizeControllerTests {
    @Test("Low acceptance backs an eight-token verification block down to four")
    func backsOffAfterLowAcceptance() {
        var controller = AdaptiveSpeculativeBlockSizeController(
            maximumBlockSize: 8, initialBlockSize: 8)
        for _ in 0 ..< 8 {
            controller.record(accepted: 0, drafted: 7)
        }

        #expect(controller.nextBlockSize(remainingTokenBudget: 20) == 4)
    }

    @Test("Consistently full prefixes grow an initial four-token block to six")
    func growsAfterStrongAcceptance() {
        var controller = AdaptiveSpeculativeBlockSizeController(
            maximumBlockSize: 8, initialBlockSize: 4)
        for _ in 0 ..< 8 {
            controller.record(accepted: 3, drafted: 3)
        }

        #expect(controller.nextBlockSize(remainingTokenBudget: 20) == 6)
    }

    @Test("A drafter-specific minimum keeps DFlash2 at three after low acceptance")
    func honorsDrafterMinimum() {
        var controller = AdaptiveSpeculativeBlockSizeController(
            maximumBlockSize: 5, initialBlockSize: 3, minimumBlockSize: 3)
        controller.record(accepted: 0, drafted: 2)

        #expect(controller.nextBlockSize(remainingTokenBudget: 20) == 3)
    }

    @Test("Full acceptance grows DFlash2 from three to five")
    func growsFromDrafterMinimum() {
        var controller = AdaptiveSpeculativeBlockSizeController(
            maximumBlockSize: 5, initialBlockSize: 3, minimumBlockSize: 3)
        for _ in 0 ..< 4 {
            controller.record(accepted: 2, drafted: 2)
        }

        #expect(controller.nextBlockSize(remainingTokenBudget: 20) == 5)
    }
}
