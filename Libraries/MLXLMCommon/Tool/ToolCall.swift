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
        /// A repeated name keeps its first position and, as in ``arguments``, its last value.
        package init(name: String, orderedArguments: [(name: String, value: any Sendable)]) {
            var arguments = [String: JSONValue]()
            var names = [String]()
            for argument in orderedArguments {
                let value = JSONValue.from(argument.value)
                if arguments.updateValue(value, forKey: argument.name) == nil {
                    names.append(argument.name)
                }
            }
            self.name = name
            self.arguments = arguments
            self.generatedArguments = jsonObjectText(names.map { ($0, arguments[$0]!) })
        }

        /// A call whose arguments the model wrote as the JSON object `text`, which `arguments`
        /// was decoded from, listing them in the order `text` does. When the keys of `text`
        /// cannot be read in order, the order stays unknown.
        package init(name: String, arguments: [String: any Sendable], writtenAs text: String) {
            guard let names = jsonObjectKeys(text) else {
                self.init(name: name, arguments: arguments)
                return
            }
            self.init(
                name: name,
                orderedArguments: names.compactMap { argument in
                    arguments[argument].map { (argument, $0) }
                })
        }

        var argumentsObject: [String: any Sendable] {
            arguments.mapValues { $0.sendableValue }
        }

        /// The argument names in generated order when known, otherwise sorted, which is the
        /// order a template engine gives an unordered dictionary.
        var argumentNames: [String] {
            generatedArguments.flatMap(jsonObjectKeys) ?? arguments.keys.sorted()
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

    public init(function: Function, id: String? = nil) {
        self.function = function
        self.id = id
    }
}

/// One JSON object listing `members` in order, or `nil` when a value has no JSON text.
private func jsonObjectText(_ members: [(name: String, value: JSONValue)]) -> String? {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    var parts = [String]()
    for member in members {
        guard let name = try? encoder.encode(member.name),
            let value = try? encoder.encode(member.value)
        else { return nil }
        parts.append(
            String(decoding: name, as: UTF8.self) + ":" + String(decoding: value, as: UTF8.self))
    }
    return "{" + parts.joined(separator: ",") + "}"
}

/// The keys of one JSON object in the order its text lists them, or `nil` when `text` is not
/// one JSON object.
private func jsonObjectKeys(_ text: String) -> [String]? {
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
    var keys = [String]()
    if !take("}") {
        repeat {
            skipWhitespace()
            guard index < utf8.endIndex, utf8[index] == Character("\"").asciiValue,
                let keyText = value(),
                let key = try? JSONDecoder().decode(String.self, from: Data(keyText.utf8)),
                take(":"), value() != nil
            else { return nil }
            keys.append(key)
        } while take(",")
        guard take("}") else { return nil }
    }
    skipWhitespace()
    return index == utf8.endIndex ? keys : nil
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
