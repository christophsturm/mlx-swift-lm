import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
import Tokenizers

private let promptTraceModels = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

struct ChatSessionPromptTraceTests {

    @Test
    func qwen36ChatSessionToolPromptBoundaries() async throws {
        let container = try await promptTraceModels.llmContainer(
            for: LLMRegistry.qwen3_6_27b_4bit)
        let tokenizer = await container.tokenizer
        let trace = await TracingLanguageModel.install(on: container)
        let markers = PromptMarker.encode(
            [
                TraceText.weatherTool,
                TraceText.timeTool,
                TraceText.firstUser,
                TraceText.secondUser,
            ],
            tokenizer: tokenizer)

        let session = ChatSession(
            container,
            instructions: "You are a trace assistant. \(TraceText.system)",
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0),
            tools: [
                tool(named: TraceText.weatherTool, description: "Return weather for a city."),
                tool(named: TraceText.timeTool, description: "Return current local time for a city."),
            ])

        try await session.send(.user("\(TraceText.firstUser) What is the weather in SF?"))
        try await session.send(.user("\(TraceText.secondUser) How about NYC?"))

        let actualMarkerTrace = trace.markerTrace(for: markers)
        let expectedMarkerTrace = [
            "prepare:\(TraceText.weatherTool)",
            "prepare:\(TraceText.timeTool)",
            "prepare:\(TraceText.firstUser)",
            "callText:\(TraceText.weatherTool)",
            "callText:\(TraceText.timeTool)",
            "callText:\(TraceText.firstUser)",
            "prepare:\(TraceText.secondUser)",
            "callText:\(TraceText.secondUser)",
        ]

        #expect(actualMarkerTrace == expectedMarkerTrace)
    }
}

private enum TraceText {
    static let system = "TRACE_SYSTEM_6f3a"
    static let weatherTool = "trace_weather_6f3a"
    static let timeTool = "trace_time_6f3a"
    static let firstUser = "TRACE_USER_SF_6f3a"
    static let secondUser = "TRACE_USER_NYC_6f3a"
}

private struct PromptMarker: Sendable {
    let name: String
    let tokens: [Int]

    static func encode(
        _ markerNames: [String],
        tokenizer: any MLXLMCommon.Tokenizer
    ) -> [PromptMarker] {
        markerNames.map {
            PromptMarker(
                name: $0,
                tokens: tokenizer.encode(text: $0, addSpecialTokens: false))
        }
    }
}

private final class TracingLanguageModel: Module, LanguageModel {
    private let base: any LanguageModel
    private let lock = NSLock()
    private var events = [ModelInputEvent]()

    static func install(on container: ModelContainer) async -> TracingLanguageModel {
        nonisolated(unsafe) var installedTrace: TracingLanguageModel?
        await container.update { context in
            let trace = TracingLanguageModel(base: context.model)
            context.model = trace
            installedTrace = trace
        }
        return installedTrace!
    }

    var recordedEvents: [ModelInputEvent] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }

    init(base: any LanguageModel) {
        self.base = base
        super.init()
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        base.sanitize(weights: weights)
    }

    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        base.sanitize(weights: weights, metadata: metadata)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return base.newCache(parameters: parameters)
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let tokenIds = input.text.tokens.asArray(Int.self)
        let result = try base.prepare(input, cache: cache, windowSize: windowSize)
        record(.init(kind: .prepare, tokens: tokenIds))
        return result
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let tokenIds = input.tokens.asArray(Int.self)
        let output = base(input, cache: cache, state: state)
        record(.init(kind: .callText, tokens: tokenIds))
        return output
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenIds = inputs.asArray(Int.self)
        let output = base(inputs, cache: cache)
        record(.init(kind: .callArray, tokens: tokenIds))
        return output
    }

    func markerTrace(for markers: [PromptMarker]) -> [String] {
        recordedEvents.flatMap { event in
            event.tokens.indices.flatMap { offset in
                markers.compactMap { marker in
                    event.tokens[offset...].starts(with: marker.tokens)
                        ? "\(event.kind.rawValue):\(marker.name)"
                        : nil
                }
            }
        }
    }

    private func record(_ event: ModelInputEvent) {
        lock.lock()
        events.append(event)
        lock.unlock()
    }
}

private struct ModelInputEvent: Sendable {
    enum Kind: String, Sendable {
        case prepare
        case callText
        case callArray
    }

    let kind: Kind
    let tokens: [Int]
}

private func tool(named name: String, description: String) -> ToolSpec {
    [
        "type": "function",
        "function": [
            "name": name,
            "description": description,
            "parameters": [
                "type": "object",
                "properties": [:] as [String: any Sendable],
                "required": [] as [String],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]
}

private extension ChatSession {
    func send(_ message: Chat.Message) async throws {
        for try await generation in streamDetails(
            to: message.content,
            role: message.role,
            images: message.images,
            videos: message.videos)
        {
            _ = generation
        }
    }
}
