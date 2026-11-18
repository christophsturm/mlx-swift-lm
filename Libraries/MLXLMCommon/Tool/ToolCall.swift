// Copyright © 2025 Apple Inc.

import Foundation

public struct ToolCall: Hashable, Codable, Sendable {
    /// Represents the function details for a tool call
    public struct Function: Hashable, Codable, Sendable {
        /// The name of the function
        public let name: String

        /// The arguments passed to the function
        public let arguments: [String: JSONValue]

        /// The arguments as one JSON object listing them in the order the model generated
        /// them, when the call was parsed from model output or restored from that text.
        ///
        /// ``arguments`` keeps no order, but a call rendered back into a prompt must list its
        /// arguments as generated, or the prompt no longer matches the generated tokens.
        /// A rendered message hands the order on under ``ToolCall/argumentOrderKey``. Calls
        /// compare and hash by ``name`` and ``arguments`` alone.
        public let generatedArguments: String?

        public init(name: String, arguments: [String: JSONValue]) {
            self.name = name
            self.arguments = arguments
            self.generatedArguments = nil
        }

        public init(name: String, arguments: [String: any Sendable]) {
            self.name = name
            self.arguments = arguments.mapValues { JSONValue.from($0) }
            self.generatedArguments = nil
        }

        /// Restores a call from its arguments' JSON text as the model generated it, so that
        /// rendering it lists the arguments in that order.
        ///
        /// - Throws: `DecodingError` when `generatedArguments` is not one JSON object.
        public init(name: String, generatedArguments: String) throws {
            self.name = name
            self.arguments = try JSONDecoder().decode(
                [String: JSONValue].self, from: Data(generatedArguments.utf8))
            self.generatedArguments = generatedArguments
        }

        /// A call parsed from model output, its arguments in the order the model generated them.
        ///
        /// `written` is the text the parser read each value from. A value keeps that text in
        /// ``generatedArguments`` when it is the JSON the value decodes from, and is encoded
        /// otherwise, as a string parameter written raw is. A repeated name keeps its first
        /// position and, as in ``arguments``, its last value and text.
        package init(
            name: String,
            orderedArguments: [(name: String, value: any Sendable, written: Substring)]
        ) {
            var arguments = [String: JSONValue]()
            var written = [String: Substring]()
            var names = [String]()
            for argument in orderedArguments {
                let value = JSONValue.from(argument.value)
                if arguments.updateValue(value, forKey: argument.name) == nil {
                    names.append(argument.name)
                }
                written[argument.name] = argument.written
            }
            self.name = name
            self.arguments = arguments
            self.generatedArguments = writtenObjectText(
                names.map { ($0, arguments[$0]!, written[$0]) })
        }

        /// A call whose arguments the model wrote as the JSON object `text`, which `arguments`
        /// was decoded from into Swift values; ``generatedArguments`` follows the same rules as
        /// for arguments decoded into `JSONValue`s.
        package init(name: String, arguments: [String: any Sendable], writtenAs text: String) {
            self.init(
                name: name, arguments: arguments.mapValues { JSONValue.from($0) }, writtenAs: text)
        }

        /// A call whose arguments the model wrote as the JSON object `text`, which `arguments`
        /// was decoded from.
        ///
        /// ``generatedArguments`` is `text` as written when it names each argument once and
        /// decodes to exactly `arguments`. Otherwise it lists the arguments in the order `text`
        /// does, each value as written where that text decodes to it, a repeated name in its
        /// first position. When `text` is not one JSON object naming exactly the arguments, the
        /// order stays unknown.
        package init(name: String, arguments: [String: JSONValue], writtenAs text: String) {
            self.name = name
            self.arguments = arguments
            let members = jsonObjectMembers(text[...])
            var names = [String]()
            var written = [String: Substring]()
            for member in members ?? [] {
                if written.updateValue(member.value, forKey: member.key) == nil {
                    names.append(member.key)
                }
            }
            guard let members, Set(names) == Set(arguments.keys) else {
                self.generatedArguments = nil
                return
            }
            if names.count == members.count,
                let object = jsonText(text[...], decodingTo: .object(arguments))
            {
                self.generatedArguments = String(object)
            } else {
                self.generatedArguments = writtenObjectText(
                    names.map { ($0, arguments[$0]!, written[$0]) })
            }
        }

        /// The arguments as a chat message hands them to a template engine.
        var argumentsObject: [String: any Sendable] {
            arguments.mapValues { $0.messageValue }
        }

        /// The arguments in the order the model wrote them, each with its value's JSON text from
        /// ``generatedArguments``: the model's own text where it wrote the value as the JSON it
        /// decodes from, the encoded value otherwise, as for a string parameter written raw.
        ///
        /// A renderer that must reproduce the generated tokens reads arguments here rather than
        /// re-encoding ``arguments``. `nil` when the written order is unknown.
        public var writtenArguments: [(name: String, value: String)]? {
            generatedArguments.flatMap { jsonObjectMembers($0[...]) }?
                .map { (name: $0.key, value: String($0.value)) }
        }

        /// The call with `arguments` in place of its own, as schema normalization converts them.
        ///
        /// The written order stays; a value `arguments` leaves as written keeps its written text,
        /// and a converted value is encoded. Without a written order, the order stays unknown.
        package func replacingArguments(_ arguments: [String: JSONValue]) -> Function {
            guard let written = writtenArguments,
                Set(written.map(\.name)) == Set(arguments.keys)
            else { return Function(name: name, arguments: arguments) }
            return Function(
                name: name, arguments: arguments,
                generatedArguments: writtenObjectText(
                    written.map { ($0.name, arguments[$0.name]!, $0.value[...]) }))
        }

        private init(name: String, arguments: [String: JSONValue], generatedArguments: String?) {
            self.name = name
            self.arguments = arguments
            self.generatedArguments = generatedArguments
        }

        /// The argument names in generated order when known, otherwise sorted, which is the
        /// order a template engine gives an unordered dictionary.
        var argumentNames: [String] {
            writtenArguments?.map(\.name) ?? arguments.keys.sorted()
        }

        public static func == (lhs: Self, rhs: Self) -> Bool {
            lhs.name == rhs.name && lhs.arguments == rhs.arguments
        }

        public func hash(into hasher: inout Hasher) {
            hasher.combine(name)
            hasher.combine(arguments)
        }
    }

    /// The function to be called
    public let function: Function

    /// Optional id used to correlate a tool call with its tool result.
    public let id: String?

    /// The key under which a rendered message lists, for each of its `tool_calls`, the
    /// argument names in the order the model generated them.
    ///
    /// Template engines receive each call's `arguments` as an unordered dictionary
    /// (swift-jinja's `Value(any:)` sorts the keys), so a bridge that renders calls as they
    /// were generated builds each arguments mapping in this order. Messages carry the key only
    /// when a call knows its ``Function/generatedArguments``; the other calls list their names
    /// sorted. It sits beside `tool_calls` rather than in them, so templates that serialize
    /// whole call objects do not render it.
    public static let argumentOrderKey = "tool_call_argument_order"

    /// The key under which a rendered message lists, for each of its `tool_calls`, the JSON text
    /// of each argument value as the model wrote it, in the order ``argumentOrderKey`` names them.
    ///
    /// It is for a renderer that wants a call exactly as the model wrote it, which re-encoding
    /// `arguments` cannot reproduce: the texts are ``Function/writtenArguments``. A call without
    /// written arguments has `nil` in its place. Messages carry the key whenever they carry
    /// ``argumentOrderKey``, beside `tool_calls`, so templates that iterate or serialize a call
    /// do not render it.
    public static let argumentTextsKey = "tool_call_argument_texts"

    public init(function: Function, id: String? = nil) {
        self.function = function
        self.id = id
    }
}

/// One JSON object listing `members` in order, or `nil` when a value has no JSON text.
///
/// Each value is the text it was written as when that text is the JSON it decodes from, and is
/// encoded otherwise.
private func writtenObjectText(
    _ members: [(name: String, value: JSONValue, written: Substring?)]
) -> String? {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    var parts = [String]()
    for member in members {
        guard let name = try? encoder.encode(member.name) else { return nil }
        let value: String
        if let written = member.written.flatMap({ jsonText($0, decodingTo: member.value) }) {
            value = String(written)
        } else {
            guard let encoded = try? encoder.encode(member.value) else { return nil }
            value = String(decoding: encoded, as: UTF8.self)
        }
        parts.append(String(decoding: name, as: UTF8.self) + ":" + value)
    }
    return "{" + parts.joined(separator: ",") + "}"
}

/// `text` without surrounding whitespace when that is exactly one JSON value which decodes to
/// `value`, read as the parsers read JSON.
private func jsonText(_ text: Substring, decodingTo value: JSONValue) -> Substring? {
    let whitespace: Set<Character> = [" ", "\t", "\n", "\r"]
    guard let first = text.firstIndex(where: { !whitespace.contains($0) }),
        let last = text.lastIndex(where: { !whitespace.contains($0) })
    else { return nil }
    let trimmed = text[first ... last]
    // A trailing space ends a number, which the scanner cannot tell is complete otherwise.
    var scanner = JSONPrefixScanner()
    guard case .complete(let length) = scanner.scan(trimmed + " "),
        length == trimmed.utf8.count,
        let decoded = try? JSONSerialization.jsonObject(
            with: Data(trimmed.utf8), options: .fragmentsAllowed),
        JSONValue.from(asSendable(decoded)) == value
    else { return nil }
    return trimmed
}

/// The members of one JSON object in the order its text lists them, each value as the text it is
/// written as, or `nil` when `text` is not one JSON object.
func jsonObjectMembers(_ text: Substring) -> [(key: String, value: Substring)]? {
    let utf8 = text.utf8
    var index = utf8.startIndex
    func skipWhitespace() {
        while index < utf8.endIndex, [9, 10, 13, 32].contains(utf8[index]) {
            utf8.formIndex(after: &index)
        }
    }
    func take(_ byte: Character) -> Bool {
        skipWhitespace()
        guard index < utf8.endIndex, utf8[index] == byte.asciiValue else { return false }
        utf8.formIndex(after: &index)
        return true
    }
    /// Advances over one JSON value and returns its text.
    func value() -> Substring? {
        skipWhitespace()
        var scanner = JSONPrefixScanner()
        guard case .complete(let length) = scanner.scan(text[index...]) else { return nil }
        let start = index
        index = utf8.index(index, offsetBy: length)
        return text[start ..< index]
    }

    guard take("{") else { return nil }
    var members = [(key: String, value: Substring)]()
    if !take("}") {
        repeat {
            skipWhitespace()
            guard index < utf8.endIndex, utf8[index] == Character("\"").asciiValue,
                let keyText = value(),
                let key = try? JSONDecoder().decode(String.self, from: Data(keyText.utf8)),
                take(":"), let memberValue = value()
            else { return nil }
            members.append((key, memberValue))
        } while take(",")
        guard take("}") else { return nil }
    }
    skipWhitespace()
    return index == utf8.endIndex ? members : nil
}

extension ToolCall {
    public func execute<Input, Output>(with tool: Tool<Input, Output>) async throws -> Output {
        // Check that the tool name matches the function name
        guard tool.name == function.name else {
            throw ToolError.nameMismatch(toolName: tool.name, functionName: function.name)
        }

        // Convert the JSONValue arguments dictionary to a JSON-encoded Data object
        let jsonObject = function.arguments.mapValues { $0.anyValue }
        let jsonData = try JSONSerialization.data(withJSONObject: jsonObject)

        // Decode the Input type from the JSON data
        let input = try JSONDecoder().decode(Input.self, from: jsonData)

        // Execute the tool's handler with the decoded input
        return try await tool.handler(input)
    }
}

// Define Tool-related errors
public enum ToolError: Error, LocalizedError {
    case nameMismatch(toolName: String, functionName: String)

    public var errorDescription: String? {
        switch self {
        case .nameMismatch(let toolName, let functionName):
            return "Tool name mismatch: expected '\(toolName)' but got '\(functionName)'"
        }
    }
}
