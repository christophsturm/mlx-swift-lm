// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// Muse answers wrongly when earlier reasoning is missing from its history, and its
/// chat template replays an assistant message's `reasoning_content` in a `to=self` frame.
@Suite("Muse Glimmer reasoning history")
struct MuseGlimmerReasoningHistoryTests {
    @Test(
        "A reply's reasoning reaches the chat template as reasoning_content",
        arguments: [false, true])
    func replyReasoningReachesTheTemplate(withToolCall: Bool) async throws {
        let tokenizer = ChatTemplateRecorder()
        let processor = MuseGlimmerProcessor(
            MuseGlimmerProcessorConfiguration(), tokenizer: tokenizer)
        let image = UserInput.Image.ciImage(
            CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 56, height: 56)))
        let call = ToolCall(
            function: .init(name: "weather.get", arguments: ["city": .string("Paris")]))
        let reasoning = "The user wants the weather in Paris."
        let reply = Chat.Message.assistant(
            "Here it is.", images: [image], toolCalls: withToolCall ? [call] : nil,
            reasoning: reasoning)

        _ = try await processor.prepare(
            input: UserInput(chat: [.user("Weather in Paris?"), reply, .user("Thanks.")]))

        let messages = tokenizer.messages
        #expect(messages.map { $0["role"] as? String } == ["user", "assistant", "user"])
        let assistant = messages[1]
        #expect(assistant["reasoning_content"] as? String == reasoning)
        #expect(
            assistant["content"] as? [[String: String]] == [
                ["type": "image"], ["type": "text", "text": "Here it is."],
            ])
        let toolCalls = assistant["tool_calls"] as? [[String: any Sendable]]
        let functions = toolCalls?.map { $0["function"] as? [String: any Sendable] }
        #expect(functions?.map { $0?["name"] as? String } == (withToolCall ? ["weather.get"] : nil))
        #expect(messages[0]["reasoning_content"] == nil)
        #expect(messages[2]["reasoning_content"] == nil)
    }
}

/// Records the messages handed to the chat template and renders one image placeholder
/// per image content part.
private final class ChatTemplateRecorder: Tokenizer, @unchecked Sendable {
    private let lock = NSLock()
    private var recorded = [[String: any Sendable]]()

    var messages: [[String: any Sendable]] { lock.withLock { recorded } }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        lock.withLock { recorded = messages }
        let images = messages.flatMap { $0["content"] as? [[String: String]] ?? [] }
            .filter { $0["type"] == "image" }
        // 200092 is the published `<|patch|>` id the processor falls back to.
        return [1] + images.map { _ in 200092 } + [2]
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
}
