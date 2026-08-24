import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXGuidedGeneration

@Suite("Guided fused greedy decode")
struct FusedGreedyDecodeTests {
    @Test("Pure grammar masks reach the model as packed bitsets")
    func pureGrammarMaskUsesFusedGreedyDecode() throws {
        let vocabulary = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
        let grammarTokenizer = try GrammarTokenizer(
            vocab: vocabulary, vocabType: .byteFallback, eosTokenId: 255)
        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer,
            jsonSchema: #"{ "type": "integer" }"#)
        let model = MaskRecordingLanguageModel(vocabularySize: vocabulary.count)
        let tokenizer = ByteFallbackTokenizer()
        let context = ModelContext(
            configuration: ModelConfiguration(id: "guided-fused-test", eosTokenIds: [255]),
            model: model,
            processor: StandInUserInputProcessor(),
            tokenizer: tokenizer)
        var emitted = ""

        let count = try GuidedGenerationLoop.run(
            input: LMInput(tokens: MLXArray([Int32(1)]).reshaped(1, 1)),
            context: context,
            constraint: constraint,
            maxTokens: 4,
            vocabSize: vocabulary.count
        ) { text in
            emitted += text
            return true
        }

        #expect(count == 1)
        #expect(emitted == "0")
        #expect(model.fusedDecodeCallCount == 1)
        #expect(model.regularDecodeCallCount == 0)
        #expect(model.receivedTokenMask?.count == 8)
    }
}

private final class MaskRecordingLanguageModel: Module, LanguageModel,
    FusedGreedyDecodeModel, KVCacheDimensionProvider
{
    let vocabularySize: Int
    var kvHeads: [Int] { [] }
    private(set) var fusedDecodeCallCount = 0
    private(set) var regularDecodeCallCount = 0
    private(set) var receivedTokenMask: [Int32]?

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        .logits(LMOutput(logits: logits(selecting: 48)))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        regularDecodeCallCount += 1
        return logits(selecting: 48)
    }

    func fusedGreedyDecode(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?,
        tokenMask: MLXArray?
    ) -> FusedGreedyDecodeOutput? {
        fusedDecodeCallCount += 1
        let words = tokenMask?.asArray(Int32.self)
        receivedTokenMask = words
        guard let words else { return nil }
        let token = isAllowed(255, words: words) ? 255 : firstAllowedToken(words: words)
        return FusedGreedyDecodeOutput(
            tokens: MLXArray([Int32(token)]).reshaped(1, 1), state: state)
    }

    private func logits(selecting token: Int) -> MLXArray {
        var values = [Float](repeating: -100, count: vocabularySize)
        values[token] = 100
        return MLXArray(values).reshaped(1, 1, vocabularySize)
    }

    private func isAllowed(_ token: Int, words: [Int32]) -> Bool {
        let word = UInt32(bitPattern: words[token / 32])
        return word & (UInt32(1) << UInt32(token % 32)) != 0
    }

    private func firstAllowedToken(words: [Int32]) -> Int {
        for token in 0 ..< vocabularySize where isAllowed(token, words: words) {
            return token
        }
        return 0
    }
}

private struct ByteFallbackTokenizer: Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = "<eos>"
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map(Int.init)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.filter { $0 != 255 }.map(UInt8.init), as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == eosToken ? 255 : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == 255 ? eosToken : String(UnicodeScalar(id)!)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
