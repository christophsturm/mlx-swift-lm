// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

/// A restored session must hold the tokens the live one held, so a call's generated arguments
/// keep every value the model wrote as JSON exactly as it wrote it: key order, separators and
/// number spellings. Each expectation is the written input itself.
@Suite("Tool-call arguments keep each value as the model wrote it")
struct ToolCallWrittenArgumentsTests {
    static let tools: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "edit",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string"] as [String: any Sendable],
                        "edits": ["type": "array"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["path", "edits"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
        [
            "type": "function",
            "function": [
                "name": "tune",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "label": ["type": "string"] as [String: any Sendable],
                        "ratio": ["type": "number"] as [String: any Sendable],
                        "limit": ["type": "integer"] as [String: any Sendable],
                        "options": ["type": "object"] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
    ]

    /// The edits as the Python reference renders them: schema key order, `", "` and `": "`.
    static let referenceEdits =
        #"[{"oldText": "alpha", "newText": "beta"}, {"oldText": "gamma", "newText": "delta"}]"#

    @Test("Qwen 3.5 keeps an array parameter in the form the model wrote it")
    func qwenKeepsAWrittenArray() throws {
        let generated =
            "<tool_call>\n<function=edit>\n<parameter=path>\nnotes.txt\n</parameter>\n"
            + "<parameter=edits>\n\(Self.referenceEdits)\n</parameter>\n</function>\n</tool_call>"
        let call = try #require(
            ToolCallFormat.qwen35.createParser().parse(content: generated, tools: Self.tools))

        #expect(
            call.function.generatedArguments
                == #"{"path":"notes.txt","edits":"# + Self.referenceEdits + "}")
    }

    @Test("Muse Glimmer keeps an array parameter in the form the model wrote it")
    func museKeepsAWrittenArray() throws {
        let generated =
            "<atem:function_calls>\n<atem:invoke name=\"edit\">\n"
            + "<atem:parameter name=\"path\">notes.txt</atem:parameter>\n"
            + "<atem:parameter name=\"edits\">\(Self.referenceEdits)</atem:parameter>\n"
            + "</atem:invoke>\n</atem:function_calls>"
        let call = try #require(
            ToolCallFormat.atem.createParser().parse(content: generated, tools: Self.tools))

        #expect(
            call.function.generatedArguments
                == #"{"path":"notes.txt","edits":"# + Self.referenceEdits + "}")
    }

    @Test("Muse Glimmer keeps a nested object's unsorted keys as the model wrote them")
    func museKeepsUnsortedNestedKeys() throws {
        let options = #"{"zeta": 1, "alpha": {"yankee": true, "bravo": null}}"#
        let generated =
            "<atem:function_calls>\n<atem:invoke name=\"tune\">\n"
            + "<atem:parameter name=\"options\">\(options)</atem:parameter>\n"
            + "</atem:invoke>\n</atem:function_calls>"
        let call = try #require(
            ToolCallFormat.atem.createParser().parse(content: generated, tools: Self.tools))

        #expect(call.function.generatedArguments == #"{"options":"# + options + "}")
    }

    @Test("Qwen 3.5 keeps number spellings and JSON-encodes a raw string parameter")
    func qwenKeepsNumberSpellings() throws {
        let generated =
            "<tool_call>\n<function=tune>\n"
            + "<parameter=label>\nsay \"hi\", then go\n</parameter>\n"
            + "<parameter=ratio>\n1.0\n</parameter>\n"
            + "<parameter=limit>\n1e3\n</parameter>\n</function>\n</tool_call>"
        let call = try #require(
            ToolCallFormat.qwen35.createParser().parse(content: generated, tools: Self.tools))

        #expect(
            call.function.generatedArguments
                == #"{"label":"say \"hi\", then go","ratio":1.0,"limit":1e3}"#)
    }

    @Test("Gemma 4 keeps number spellings as the model wrote them")
    func gemmaKeepsNumberSpellings() throws {
        let generated = "<|tool_call>call:tune{ratio:1.0,limit:1e3}<tool_call|>"
        let call = try #require(
            ToolCallFormat.gemma4.createParser().parse(content: generated, tools: Self.tools))

        #expect(call.function.generatedArguments == #"{"ratio":1.0,"limit":1e3}"#)
    }

    @Test("A call lists its arguments in written order, each with its value's written JSON text")
    func writtenArgumentsListNamesAndValueTexts() throws {
        let generated =
            "<tool_call>\n<function=edit>\n<parameter=path>\nnotes.txt\n</parameter>\n"
            + "<parameter=edits>\n\(Self.referenceEdits)\n</parameter>\n</function>\n</tool_call>"
        let parsed = try #require(
            ToolCallFormat.qwen35.createParser().parse(content: generated, tools: Self.tools))
        let restored = try ToolCall.Function(
            name: "edit",
            generatedArguments: #"{"path": "notes.txt", "edits": "# + Self.referenceEdits + "}")
        let unordered = ToolCall.Function(name: "edit", arguments: ["path": .string("notes.txt")])

        let parsedArguments = try #require(parsed.function.writtenArguments)
        let restoredArguments = try #require(restored.writtenArguments)
        #expect(parsedArguments.map(\.name) == ["path", "edits"])
        #expect(parsedArguments.map(\.value) == [#""notes.txt""#, Self.referenceEdits])
        #expect(restoredArguments.map(\.name) == ["path", "edits"])
        #expect(restoredArguments.map(\.value) == [#""notes.txt""#, Self.referenceEdits])
        #expect(unordered.writtenArguments == nil)
    }

    @Test("A JSON call keeps its whole arguments object as the model wrote it")
    func jsonKeepsTheWrittenObject() throws {
        let arguments =
            #"{"label": "say \"hi\", then go", "ratio": 1.0, "limit": 1e3, "#
            + #""options": {"zeta": 1, "alpha": {"yankee": true, "bravo": null}}}"#
        let generated = #"<tool_call>{"name": "tune", "arguments": "# + arguments + "}</tool_call>"
        let call = try #require(
            ToolCallFormat.json.createParser().parse(content: generated, tools: Self.tools))

        #expect(call.function.generatedArguments == arguments)
    }

    @Test("Mistral keeps an edit call's arguments object as the model wrote it")
    func mistralKeepsTheWrittenObject() throws {
        let arguments = #"{"path": "notes.txt", "edits": "# + Self.referenceEdits + "}"
        let generated = "[TOOL_CALLS]edit[ARGS]" + arguments
        let call = try #require(
            ToolCallFormat.mistral.createParser().parse(content: generated, tools: Self.tools))

        #expect(call.function.generatedArguments == arguments)
    }

    @Test("Two JSON calls in one reply each keep their arguments object as the model wrote it")
    func twoJSONCallsKeepTheirWrittenObjects() throws {
        let processor = ToolCallProcessor(format: .json, tools: Self.tools)
        let edit = #"{"path": "notes.txt", "edits": "# + Self.referenceEdits + "}"
        let tune = #"{"options": {"zeta": 2, "alpha": [3, 1]}, "ratio": 2.50}"#
        let generated =
            #"<tool_call>{"name": "edit", "arguments": "# + edit + "}</tool_call>\n"
            + #"<tool_call>{"name": "tune", "arguments": "# + tune + "}</tool_call>"

        _ = processor.processChunk(generated)
        processor.processEOS()

        #expect(processor.toolCalls.map(\.function.generatedArguments) == [edit, tune])
    }

    @Test("Two Qwen 3.5 calls in one reply each keep their values as the model wrote them")
    func twoQwenCallsKeepTheirWrittenValues() throws {
        let processor = ToolCallProcessor(format: .qwen35, tools: Self.tools)
        let generated =
            "<tool_call>\n<function=edit>\n<parameter=path>\nnotes.txt\n</parameter>\n"
            + "<parameter=edits>\n\(Self.referenceEdits)\n</parameter>\n</function>\n</tool_call>\n"
            + "<tool_call>\n<function=tune>\n<parameter=options>\n"
            + #"{"zeta": 2, "alpha": [3, 1]}"#
            + "\n</parameter>\n</function>\n</tool_call>"

        _ = processor.processChunk(generated)
        processor.processEOS()

        #expect(
            processor.toolCalls.map(\.function.generatedArguments) == [
                #"{"path":"notes.txt","edits":"# + Self.referenceEdits + "}",
                #"{"options":{"zeta": 2, "alpha": [3, 1]}}"#,
            ])
    }
}
