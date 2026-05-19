// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXOptimizers
import XCTest

@testable import MLXLMCommon

private enum RecordingChatToken {
    static let bos = 1
    static let system = 2
    static let weatherTool = 3
    static let assistantStart = 4
    static let timeTool = 5
    static let contextFlag = 6
    static let firstUser = 10
    static let secondUser = 11
    static let generated = 20
    static let eos = 99

    static func name(for id: Int) -> String {
        switch id {
        case bos: "<bos>"
        case system: "<system>"
        case weatherTool: "tool:weather"
        case assistantStart: "<assistant>"
        case timeTool: "tool:time"
        case contextFlag: "context:enable_thinking=false"
        case firstUser: "user:SF"
        case secondUser: "user:NYC"
        case generated: "<generated>"
        case eos: "<eos>"
        default: "token\(id)"
        }
    }
}

private final class RecordingPromptStore: @unchecked Sendable {
    private let lock = NSLock()
    private var _modelInputEvents = [String]()

    func recordModelInputEvent(kind: String, tokens: [Int]) {
        let markers = tokens.compactMap(Self.markerName(for:))
        lock.lock()
        _modelInputEvents.append(contentsOf: markers.map { "\(kind):\($0)" })
        lock.unlock()
    }

    var modelInputEvents: [String] {
        lock.lock()
        defer { lock.unlock() }
        return _modelInputEvents
    }

    private static func markerName(for id: Int) -> String? {
        switch id {
        case RecordingChatToken.bos:
            "bos"
        case RecordingChatToken.system:
            "system"
        case RecordingChatToken.weatherTool:
            "weatherTool"
        case RecordingChatToken.timeTool:
            "timeTool"
        case RecordingChatToken.contextFlag:
            "contextFlag"
        case RecordingChatToken.firstUser:
            "firstUser"
        case RecordingChatToken.secondUser:
            "secondUser"
        default:
            nil
        }
    }
}

private struct RecordingChatTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        []
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(RecordingChatToken.name(for:)).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case bosToken:
            RecordingChatToken.bos
        case eosToken:
            RecordingChatToken.eos
        default:
            nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        RecordingChatToken.name(for: id)
    }

    var bosToken: String? { "<bos>" }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var tokens = [Int]()
        if messages.contains(where: { ($0["role"] as? String) == "system" }) {
            tokens.append(RecordingChatToken.bos)
            tokens.append(RecordingChatToken.system)
        }
        for tool in tools ?? [] {
            if !tokens.contains(RecordingChatToken.bos) {
                tokens.append(RecordingChatToken.bos)
            }
            guard let function = tool["function"] as? [String: any Sendable],
                let name = function["name"] as? String
            else {
                continue
            }
            if name.contains("time") {
                tokens.append(RecordingChatToken.timeTool)
            } else {
                tokens.append(RecordingChatToken.weatherTool)
            }
        }
        if (additionalContext?["enable_thinking"] as? Bool) == false {
            tokens.append(RecordingChatToken.contextFlag)
        }
        for message in messages {
            guard (message["role"] as? String) == "user" else {
                continue
            }
            let content = message["content"] as? String
            tokens.append(
                content?.contains("NYC") == true
                    ? RecordingChatToken.secondUser
                    : RecordingChatToken.firstUser)
        }
        tokens.append(RecordingChatToken.assistantStart)
        return tokens
    }
}

private final class RecordingLanguageModel: Module, LanguageModel {
    let store: RecordingPromptStore

    init(store: RecordingPromptStore) {
        self.store = store
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        store.recordModelInputEvent(
            kind: "prepare",
            tokens: input.text.tokens.asArray(Int.self))
        return .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        store.recordModelInputEvent(
            kind: "callText",
            tokens: input.tokens.asArray(Int.self))
        let logits = callAsFunction(input.tokens, cache: cache)
        return .init(logits: logits)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let shape = inputs.shape
        let batch = shape.count > 0 ? shape[0] : 1
        let sequence = shape.count > 1 ? shape[1] : inputs.size
        let vocabularySize = 128
        var logits = Array(repeating: Float(-1_000), count: batch * sequence * vocabularySize)
        for index in 0 ..< (batch * sequence) {
            logits[index * vocabularySize + RecordingChatToken.generated] = 1_000
        }
        return MLXArray(logits).reshaped(batch, sequence, vocabularySize)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }
}

/// See also ChatSessionIntegrationTests
public class ChatSessionTests: XCTestCase {

    private func model() -> ModelContext {
        let config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64, attentionHeads: 4,
            headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let model = Gemma3TextModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        // Force evaluation of all model weights before concurrent usage
        // This ensures all weight promises are realized and avoids race conditions
        eval(model)

        let processor = TestInputProcessor()

        return .init(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer)
    }

    private let generationParameters = GenerateParameters(maxTokens: 50)

    private let targetLength = 1

    private func weatherTool() -> ToolSpec {
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string",
                            "description": "City name",
                        ] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["location"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as ToolSpec
    }

    private func timeTool() -> ToolSpec {
        [
            "type": "function",
            "function": [
                "name": "get_time",
                "description": "Get the current time",
                "parameters": [
                    "type": "object",
                    "properties": [:] as [String: any Sendable],
                    "required": [] as [String],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as ToolSpec
    }

    func testChatSessionSync() async throws {
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        let result1 = try await session.respond(to: "hello")
        XCTAssertGreaterThan(result1.count, targetLength, result1)
        let result2 = try await session.respond(to: "hello again")
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionAsync() async throws {
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        var result1 = ""
        for try await part in session.streamResponse(to: "hello") {
            result1 += part
        }
        XCTAssertGreaterThan(result1.count, targetLength, result1)

        var result2 = ""
        for try await part in session.streamResponse(to: "hello again") {
            result2 += part
        }
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionAsyncInterrupt() async throws {
        // interrupt the streamResponse and continue with another request
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        for _ in 0 ..< 10 {
            var result1 = ""
            for try await part in session.streamResponse(to: "hello") {
                result1 += part
                break
            }

            // at this point the performStreaming/generate code may still be running.
            // the next call can corrupt the state if not thread safe

            var result2 = ""
            for try await part in session.streamResponse(to: "hello again") {
                result2 += part
                if result2.count > 100 {
                    break
                }
            }
        }

        // since we are interrupting we need to wait for everything to finish
        // (avoids shutdown issues if this is the last/only test). because the
        // streaming task is not a synchronous shutdown
        await session.synchronize()
    }

    func testChatSessionDoesNotReemitStaticPromptMarkersAcrossTurns() async throws {
        let store = RecordingPromptStore()
        let tokenizer = RecordingChatTokenizer()
        let configuration = ModelConfiguration(id: "recording-chat")
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: DefaultMessageGenerator()
        )
        let model = ModelContext(
            configuration: configuration,
            model: RecordingLanguageModel(store: store),
            processor: processor,
            tokenizer: tokenizer
        )
        let session = ChatSession(
            model,
            instructions: "You are a trace assistant.",
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0),
            tools: [weatherTool(), timeTool()]
        )

        _ = try await session.respond(to: "What is the weather in SF?")
        _ = try await session.respond(to: "How about NYC?")

        XCTAssertEqual(
            store.modelInputEvents,
            [
                "prepare:bos",
                "prepare:system",
                "prepare:weatherTool",
                "prepare:timeTool",
                "prepare:firstUser",
                "callText:bos",
                "callText:system",
                "callText:weatherTool",
                "callText:timeTool",
                "callText:firstUser",
                "prepare:secondUser",
                "callText:secondUser",
            ],
            "ChatSession should append only the newly rendered turn at the model boundary."
        )
    }

    func testChatSessionKeepsAdditionalContextAcrossCachedTurns() async throws {
        let store = RecordingPromptStore()
        let tokenizer = RecordingChatTokenizer()
        let configuration = ModelConfiguration(id: "recording-chat")
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: DefaultMessageGenerator()
        )
        let model = ModelContext(
            configuration: configuration,
            model: RecordingLanguageModel(store: store),
            processor: processor,
            tokenizer: tokenizer
        )
        let session = ChatSession(
            model,
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0),
            additionalContext: ["enable_thinking": false]
        )

        _ = try await session.respond(to: "What is the weather in SF?")
        _ = try await session.respond(to: "How about NYC?")

        XCTAssertEqual(
            store.modelInputEvents,
            [
                "prepare:contextFlag",
                "prepare:firstUser",
                "callText:contextFlag",
                "callText:firstUser",
                "prepare:contextFlag",
                "prepare:secondUser",
                "callText:contextFlag",
                "callText:secondUser",
            ],
            "ChatSession should pass template additionalContext for every rendered suffix."
        )
    }

    func testRestoredCachePreparesFirstPromptAdditions() async throws {
        let store = RecordingPromptStore()
        let tokenizer = RecordingChatTokenizer()
        let configuration = ModelConfiguration(id: "recording-chat")
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: DefaultMessageGenerator()
        )
        let model = ModelContext(
            configuration: configuration,
            model: RecordingLanguageModel(store: store),
            processor: processor,
            tokenizer: tokenizer
        )
        let container = ModelContainer(context: model)
        let restoredTool = weatherTool()
        let initial = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0)
        )

        _ = try await initial.respond(to: "What is the weather in SF?")

        try await initial.withCache { cache in
            guard let cache else {
                XCTFail("expected cache after initial response")
                return
            }

            let restored = ChatSession(
                container,
                instructions: "You are a restored trace assistant.",
                cache: cache.map { $0.copy() },
                generateParameters: GenerateParameters(maxTokens: 1, temperature: 0),
                additionalContext: ["enable_thinking": false],
                tools: [restoredTool]
            )

            _ = try await restored.respond(to: "How about NYC?")
            _ = try await restored.respond(to: "How about NYC?")
        }

        XCTAssertEqual(
            store.modelInputEvents,
            [
                "prepare:firstUser",
                "callText:firstUser",
                "prepare:bos",
                "prepare:system",
                "prepare:weatherTool",
                "prepare:contextFlag",
                "prepare:secondUser",
                "callText:bos",
                "callText:system",
                "callText:weatherTool",
                "callText:contextFlag",
                "callText:secondUser",
                "prepare:contextFlag",
                "prepare:secondUser",
                "callText:contextFlag",
                "callText:secondUser",
            ],
            "A restored KV cache should render this session's first prompt additions once."
        )
    }

    func testChatSessionWithTools() async throws {
        let model = model()
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": [
                                "type": "string",
                                "description": "City name",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model, generateParameters: generationParameters, tools: tools
        )

        let result = try await session.respond(to: "What is the weather in SF?")
        XCTAssertGreaterThan(result.count, targetLength, result)

        // second turn to verify tools persist through cache
        let result2 = try await session.respond(to: "How about NYC?")
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionWithToolsStreaming() async throws {
        let model = model()
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model, generateParameters: generationParameters, tools: tools
        )

        var result = ""
        for try await part in session.streamResponse(to: "hello") {
            result += part
        }
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    // MARK: - KV Cache

    func testCurrentCacheNilBeforeGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCurrentCacheAfterGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
    }

    func testInitWithKVCache() async throws {
        // build a cache from an initial session
        let container = ModelContainer(context: model())
        let initial = ChatSession(container, generateParameters: generationParameters)
        _ = try await initial.respond(to: "hello")

        try await initial.withCache { [targetLength, generationParameters] cache in
            XCTAssertNotNil(cache)

            if let cache {
                // restore the cache into a new session and verify generation continues
                let restored = ChatSession(
                    container,
                    cache: cache.map { $0.copy() },
                    generateParameters: generationParameters)
                let result = try await restored.respond(to: "hello again")
                XCTAssertGreaterThan(result.count, targetLength, result)
            }
        }
    }

    func testSaveCacheThrowsBeforeGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        do {
            try await session.saveCache(to: url)
            XCTFail("expected ChatSessionError.noCacheAvailable")
        } catch ChatSessionError.noCacheAvailable {
            // expected
        }
    }

    func testSaveAndRestoreCache() async throws {
        let ctx = model()
        let initial = ChatSession(ctx, generateParameters: generationParameters)
        _ = try await initial.respond(to: "hello")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        try await initial.saveCache(to: url)

        let (loadedCache, _) = try loadPromptCache(url: url)
        let restored = ChatSession(
            ctx, cache: loadedCache, generateParameters: generationParameters)
        let result = try await restored.respond(to: "hello again")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testCurrentCacheNilForHistorySessionBeforeGeneration() async throws {
        // .history state should behave like .empty: no cache until first generation
        let history: [Chat.Message] = [.user("hello"), .assistant("hi")]
        let session = ChatSession(
            model(), history: history, generateParameters: generationParameters)
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCurrentCacheNonNilForHistorySessionAfterGeneration() async throws {
        // after generation from .history state, cache transitions to .kvcache
        let history: [Chat.Message] = [.user("hello"), .assistant("hi")]
        let session = ChatSession(
            model(),
            history: history,
            generateParameters: generationParameters)
        _ = try await session.respond(to: "hello again")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
    }

    func testCurrentCacheNilAfterClear() async throws {
        // clear() resets to .empty; currentCache() should return nil again
        let session = ChatSession(model(), generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
        await session.clear()
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    /// something that looks like a view model
    @MainActor class ChatModel {
        let session: ChatSession

        public var messages = [Chat.Message]()

        private var task: Task<Void, Error>?
        public var isBusy: Bool {
            task != nil
        }

        init(model: ModelContext) {
            self.session = ChatSession(model)
        }

        public func cancel() {
            task?.cancel()
        }

        public func respond(_ message: String) {
            guard task == nil else { return }

            self.messages.append(.init(role: .user, content: message))
            self.messages.append(.init(role: .assistant, content: "..."))
            let lastIndex = self.messages.count - 1

            self.task = Task {
                var first = true
                for try await item in session.streamResponse(to: message) {
                    if first {
                        self.messages[lastIndex].content = item
                        first = false
                    } else {
                        self.messages[lastIndex].content += item
                    }
                }
                self.task = nil
            }
        }
    }

    @MainActor
    func testViewModel() async throws {
        let model = ChatModel(model: model())

        // start producing a response but interrupt it
        // triggers https://github.com/ml-explore/mlx-swift/pull/323
        model.respond("message1")
        try await Task.sleep(for: .milliseconds(50))
        model.cancel()

        // wait for it to finish
        while model.isBusy {
            try await Task.sleep(for: .milliseconds(10))
        }

        // try another message, wait for full completion (but cap the length)
        model.session.generateParameters = self.generationParameters
        model.respond("message2")
        while model.isBusy {
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}
