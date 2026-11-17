// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for JSON format: <tag>{"name": "...", "arguments": {...}}</tag>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/default.py
public struct JSONToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?
    public let supportsBareJSON: Bool
    private let jsonObjectScanner = JSONLeadingObjectScanner(startCharacter: "{")

    public init(startTag: String, endTag: String, supportsBareJSON: Bool = false) {
        self.supportsBareJSON = supportsBareJSON
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let start = startTag, let end = endTag else { return nil }

        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        // A bare JSON payload may contain literal protocol markers in its
        // strings. Only wrapper tags outside that payload are delimiters.
        if !text.hasPrefix("{"), let startRange = text.range(of: start) {
            text = String(text[startRange.upperBound...]).trimmingCharacters(
                in: .whitespacesAndNewlines)
        }
        if text.hasSuffix(end) {
            text.removeLast(end.count)
        }

        return parsePayload(text)
    }

    /// Parse an already-extracted JSON payload without searching for protocol
    /// delimiters again. Framing-aware callers use this after finding the
    /// structural outer close, so a literal end-tag string inside an argument
    /// remains JSON data rather than truncating the payload.
    func parsePayload(_ payload: String) -> ToolCall? {
        let json = payload.trimmingCharacters(in: .whitespacesAndNewlines)
        return parseToolCall(from: json) ?? parseRedundantOuterBraces(from: json)
    }

    /// Some Qwen chat templates emit an EOS-delimited JSON call with a
    /// redundant leading brace and two redundant closing braces.
    /// Recover only that exact shape and only when the enclosed prefix is one
    /// complete, valid tool-call object followed solely by those braces.
    private func parseRedundantOuterBraces(from text: String) -> ToolCall? {
        guard text.hasPrefix("{{") else { return nil }
        let withoutLeadingBrace = String(text.dropFirst())
        guard let split = jsonObjectScanner.splitLeadingObject(from: withoutLeadingBrace) else {
            return nil
        }
        let trailing = split.trailing.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trailing == "}}" else {
            return nil
        }
        return parseToolCall(from: split.object)
    }

    /// Parses one call object, its arguments in the order the model wrote them.
    ///
    /// The object is decoded through `JSONSerialization`, which keeps no key order, so the
    /// order comes from the arguments' own text: the `arguments` member as written, or the
    /// JSON text a stringified `arguments` holds.
    private func parseToolCall(from text: String) -> ToolCall? {
        guard let data = text.data(using: .utf8),
            var jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        var members = jsonObjectMembers(text[...])

        var id = jsonObject["id"] as? String
        if let functionObject = jsonObject["function"] as? [String: Any] {
            id = id ?? functionObject["id"] as? String
            jsonObject = functionObject
            members = members?.first { $0.key == "function" }.flatMap {
                jsonObjectMembers($0.value)
            }
        }
        var argumentsText = members?.first { $0.key == "arguments" }.map { String($0.value) }

        if let stringifiedArguments = jsonObject["arguments"] as? String {
            guard
                let argumentsData = stringifiedArguments.data(using: .utf8),
                let argumentsObject = try? JSONSerialization.jsonObject(with: argumentsData)
                    as? [String: Any]
            else { return nil }
            jsonObject["arguments"] = argumentsObject
            argumentsText = stringifiedArguments
        }

        guard
            let normalizedData = try? JSONSerialization.data(withJSONObject: jsonObject),
            let function = try? JSONDecoder().decode(ToolCall.Function.self, from: normalizedData)
        else { return nil }

        guard let argumentsText else { return ToolCall(function: function, id: id) }
        return ToolCall(
            function: ToolCall.Function(
                name: function.name, arguments: function.arguments, writtenAs: argumentsText),
            id: id)
    }
}
