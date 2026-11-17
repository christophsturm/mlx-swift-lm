// Copyright © 2026 Apple Inc.

import Foundation
import Jinja
import MLXLMCommon
import Testing

@testable import MLXVLM

/// Muse Glimmer writes an answer and then calls as consecutive frames of one turn, the answer
/// ending in `<|eom|>`. Its chat template renders that only for an answer whose `end_turn` is
/// false; an assistant message with calls renders the calls alone.
@Suite("An assistant message's turn end reaches the chat template")
struct ChatMessageTurnEndTests {
    @Test("Muse Glimmer ends an answer that its calls follow with end-of-message")
    func museEndsAnAnswerBeforeItsCallsWithEndOfMessage() throws {
        var answer = Chat.Message.assistant("Fixing it.")
        answer.endsTurn = false
        let call = ToolCall(
            function: .init(name: "record_edit", arguments: ["target": .string("b.txt")]))
        let messages = Qwen2VLMessageGenerator().generate(messages: [
            .user("Fix b.txt."), answer, .assistant("", toolCalls: [call]),
        ])

        let rendered = try render("muse-glimmer-30B", messages: messages)

        let turn: String = """
            <|start|>user<|message|>Fix b.txt.<|eot|>\
            <|start|>assistant to=user<|message|>Fixing it.<|eom|>\
            <|start|>assistant to=record_edit<|message|><atem:function_calls>
            <atem:invoke name="record_edit">
            <atem:parameter name="target">b.txt</atem:parameter>
            </atem:invoke>
            </atem:function_calls><|eot|>
            """
        #expect(rendered.hasSuffix(turn), "\(rendered)")
    }

    @Test("an assistant message that does not end its turn reaches the template as end_turn false")
    func turnEndReachesTheTemplate() {
        var answer = Chat.Message.assistant("Fixing it.")
        answer.endsTurn = false

        let generated = DefaultMessageGenerator().generate(message: answer)

        #expect(generated["end_turn"] as? Bool == false)
    }

    @Test("a message that leaves its turn end to the template reaches it as before")
    func unsetTurnEndAddsNothing() {
        let generated = DefaultMessageGenerator().generate(message: .assistant("Fixing it."))

        #expect(Set(generated.keys) == ["role", "content"])
        #expect(generated["content"] as? String == "Fixing it.")
    }

    private func render(_ template: String, messages: [Message]) throws -> String {
        let url = try #require(
            Bundle.module.url(
                forResource: template, withExtension: "jinja", subdirectory: "ChatTemplates"))
        return try Template(String(contentsOf: url, encoding: .utf8)).render([
            "messages": try Value(any: messages),
            "bos_token": "<|begin_of_text|>",
            "add_generation_prompt": false,
        ])
    }
}
