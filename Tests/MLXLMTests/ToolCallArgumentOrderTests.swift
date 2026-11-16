// Copyright © 2026 Apple Inc.

import Foundation
import Jinja
import MLXLMCommon
import Testing

@testable import MLXVLM

/// A call the model generated must render back into the prompt exactly as generated, or the
/// rendered history no longer matches the tokens a prompt cache holds. Rendering uses the
/// published templates (Muse-Glimmer-30B at cec94fc, Qwen3.5-0.8B at 2fc0636) with swift-jinja
/// 2.3.5, whose `Value(any:)` sorts dictionary keys.
@Suite("Tool-call arguments render in the order the model generated them")
struct ToolCallArgumentOrderTests {
    static let tools: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "note",
                "description": "Writes a note.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "b": ["type": "string"] as [String: any Sendable],
                        "a": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["b", "a"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    @Test("Muse Glimmer renders a parsed call as the model generated it")
    func museRendersTheGeneratedCall() throws {
        let generated = """
            <atem:function_calls>
            <atem:invoke name="note">
            <atem:parameter name="b">Lantern</atem:parameter>
            <atem:parameter name="a">Bring oil</atem:parameter>
            </atem:invoke>
            </atem:function_calls>
            """
        let call = try #require(ATEMToolCallParser().parse(content: generated, tools: Self.tools))
        let messages = Qwen2VLMessageGenerator().generate(messages: conversation(calling: call))

        let rendered = try render("muse-glimmer-30B", messages: messages)

        #expect(call.function.generatedArguments == #"{"b":"Lantern","a":"Bring oil"}"#)
        #expect(rendered.contains("<|start|>assistant to=note<|message|>" + generated + "<|eot|>"))
    }

    @Test("Qwen 3.5 renders a parsed call as the model generated it")
    func qwenRendersTheGeneratedCall() throws {
        let generated = """
            <tool_call>
            <function=note>
            <parameter=b>
            Lantern
            </parameter>
            <parameter=a>
            Bring oil
            </parameter>
            </function>
            </tool_call>
            """
        let parser = Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let call = try #require(parser.parse(content: generated, tools: Self.tools))
        let messages = DefaultMessageGenerator().generate(messages: conversation(calling: call))

        let rendered = try render("qwen3.5-0.8B", messages: messages)

        #expect(call.function.generatedArguments == #"{"b":"Lantern","a":"Bring oil"}"#)
        #expect(rendered.contains(generated))
    }

    @Test("A call restored from its generated arguments renders them in that order")
    func restoredCallRendersInGeneratedOrder() throws {
        let generatedArguments = #"{"b": "Lantern", "a": "Bring oil"}"#
        let function = try ToolCall.Function(name: "note", generatedArguments: generatedArguments)
        let messages = DefaultMessageGenerator().generate(
            messages: conversation(calling: ToolCall(function: function)))

        let rendered = try render("qwen3.5-0.8B", messages: messages)

        #expect(function.arguments == ["b": .string("Lantern"), "a": .string("Bring oil")])
        #expect(function.generatedArguments == generatedArguments)
        #expect(
            rendered.contains(
                "<parameter=b>\nLantern\n</parameter>\n<parameter=a>\nBring oil\n</parameter>"))
    }

    private func conversation(calling call: ToolCall) -> [Chat.Message] {
        [.user("Write a note."), .assistant("", toolCalls: [call])]
    }

    private func render(_ template: String, messages: [Message]) throws -> String {
        let url = try #require(
            Bundle.module.url(
                forResource: template, withExtension: "jinja", subdirectory: "ChatTemplates"))
        return try Template(String(contentsOf: url, encoding: .utf8)).render([
            "messages": .array(try messages.map(templateValue)),
            "tools": try Value(any: Self.tools),
            "bos_token": "<|begin_of_text|>",
            "add_generation_prompt": false,
        ])
    }
}

/// Converts a message the way a Jinja bridge must for calls to render as generated: like
/// `Value(any:)`, which sorts dictionary keys, except that each call's arguments follow the
/// names the message lists for it under ``ToolCall/argumentOrderKey``.
private func templateValue(_ message: Message) throws -> Value {
    guard case .object(var object) = try Value(any: message) else {
        throw TemplateValueError.notAnObject
    }
    if let calls = message["tool_calls"] as? [[String: any Sendable]],
        let orders = message[ToolCall.argumentOrderKey] as? [[String]]
    {
        object["tool_calls"] = .array(try zip(calls, orders).map(orderedCallValue))
    }
    return .object(object)
}

private func orderedCallValue(_ call: [String: any Sendable], order: [String]) throws -> Value {
    guard case .object(var object) = try Value(any: call),
        let function = call["function"] as? [String: any Sendable],
        let arguments = function["arguments"] as? [String: any Sendable],
        case .object(var functionObject) = try Value(any: function)
    else { throw TemplateValueError.notAnObject }
    var ordered = OrderedDictionary<String, Value>()
    for name in order {
        ordered[name] = try Value(any: arguments[name])
    }
    functionObject["arguments"] = .object(ordered)
    object["function"] = .object(functionObject)
    return .object(object)
}

private enum TemplateValueError: Error {
    case notAnObject
}
