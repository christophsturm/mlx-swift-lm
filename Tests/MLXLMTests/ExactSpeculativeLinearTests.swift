import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("Exact speculative linear projections", .serialized)
struct ExactSpeculativeLinearTests {
    @Test("Dense verification kernel matches one-token projections exactly")
    func denseBlockMatchesSingletons() {
        MLXRandom.seed(30)
        let linear = Linear(
            weight: MLXRandom.normal([64, 512]).asType(.bfloat16))
        let input = MLXRandom.normal([2, 4, 512]).asType(.bfloat16)

        let reference = concatenated(
            (0 ..< input.dim(1)).map { linear(input[0..., $0 ..< ($0 + 1), 0...]) },
            axis: 1)
        let actual = exactSpeculativeLinear(linear, input)

        eval(reference, actual)
        #expect(arrayEqual(reference, actual).item(Bool.self))
    }

    @Test("Four-bit streamed verification matches one-token projections exactly")
    func fourBitBlockMatchesSingletons() {
        MLXRandom.seed(31)
        let source = QuantizedLinear(512, 16, bias: false, groupSize: 64, bits: 4)
        let linear = QuantizedLinear(
            weight: source.weight,
            scales: source.scales.asType(.bfloat16),
            biases: source.biases?.asType(.bfloat16),
            groupSize: source.groupSize,
            bits: source.bits,
            mode: source.mode)
        let input = MLXRandom.normal([1, 6, 512]).asType(.bfloat16)

        let reference = concatenated(
            (0 ..< input.dim(1)).map { linear(input[0..., $0 ..< ($0 + 1), 0...]) },
            axis: 1)
        let actual = exactSpeculativeLinear(linear, input)

        eval(reference, actual)
        #expect(allClose(reference, actual, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test("Five-bit verification and argmax match exact singleton projections")
    func fiveBitBlockMatchesSingletons() throws {
        MLXRandom.seed(38)
        let source = QuantizedLinear(512, 16, bias: false, groupSize: 64, bits: 5)
        let linear = QuantizedLinear(
            weight: source.weight,
            scales: source.scales.asType(.bfloat16),
            biases: source.biases?.asType(.bfloat16),
            groupSize: source.groupSize,
            bits: source.bits,
            mode: source.mode)
        let input = MLXRandom.normal([1, 4, 512]).asType(.bfloat16)

        let reference = concatenated(
            (0 ..< input.dim(1)).map { linear(input[0..., $0 ..< ($0 + 1), 0...]) },
            axis: 1)
        let actual = exactSpeculativeLinear(linear, input)
        let expectedTokens = argMax(reference, axis: -1)
        let actualTokens = try #require(exactSpeculativeQuantizedArgMax(linear, input))

        eval(reference, actual, expectedTokens, actualTokens)
        #expect(allClose(reference, actual, rtol: 0, atol: 0).item(Bool.self))
        #expect(arrayEqual(expectedTokens, actualTokens).item(Bool.self))
    }

    @Test("Eight-bit verification matches one-token quantized projections exactly")
    func eightBitBlockMatchesSingletons() {
        MLXRandom.seed(32)
        let source = QuantizedLinear(512, 16, bias: false, groupSize: 64, bits: 8)
        let linear = QuantizedLinear(
            weight: source.weight,
            scales: source.scales.asType(.bfloat16),
            biases: source.biases?.asType(.bfloat16),
            groupSize: source.groupSize,
            bits: source.bits,
            mode: source.mode)
        let input = MLXRandom.normal([1, 4, 512]).asType(.bfloat16)

        let reference = concatenated(
            (0 ..< input.dim(1)).map { linear(input[0..., $0 ..< ($0 + 1), 0...]) },
            axis: 1)
        let actual = exactSpeculativeLinear(linear, input)

        eval(reference, actual)
        #expect(allClose(reference, actual, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test("Related four-bit projections fuse without changing singleton results")
    func fourBitProjectionGroupMatchesSingletons() {
        MLXRandom.seed(33)
        let linears: [Linear] = [16, 24, 32].map { outputSize in
            let source = QuantizedLinear(
                512, outputSize, bias: false, groupSize: 64, bits: 4)
            return QuantizedLinear(
                weight: source.weight,
                scales: source.scales.asType(.bfloat16),
                biases: source.biases?.asType(.bfloat16),
                groupSize: source.groupSize,
                bits: source.bits,
                mode: source.mode)
        }
        let input = MLXRandom.normal([1, 4, 512]).asType(.bfloat16)
        let reference = linears.map { linear in
            concatenated(
                (0 ..< input.dim(1)).map {
                    linear(input[0..., $0 ..< ($0 + 1), 0...])
                },
                axis: 1)
        }
        let actual = exactSpeculativeLinears(linears, input)

        eval(reference, actual)
        #expect(
            zip(reference, actual).allSatisfy {
                arrayEqual($0, $1).item(Bool.self)
            })
    }

    @Test("Four-bit streamed argmax matches exact verification logits")
    func fourBitArgMaxMatchesExactLogits() throws {
        MLXRandom.seed(34)
        let source = QuantizedLinear(512, 16, bias: false, groupSize: 64, bits: 4)
        let linear = QuantizedLinear(
            weight: source.weight,
            scales: source.scales.asType(.bfloat16),
            biases: source.biases?.asType(.bfloat16),
            groupSize: source.groupSize,
            bits: source.bits,
            mode: source.mode)
        let input = MLXRandom.normal([1, 6, 512]).asType(.bfloat16)

        let reference = argMax(exactSpeculativeLinear(linear, input), axis: -1)
        let actual = try #require(exactSpeculativeQuantizedArgMax(linear, input))

        eval(reference, actual)
        #expect(arrayEqual(reference, actual).item(Bool.self))
    }

    @Test("Four-bit token tiling remains exact beyond the streamed-width range")
    func fourBitTokenTiledWideBlockMatchesSingletons() throws {
        MLXRandom.seed(39)
        let source = QuantizedLinear(512, 16, bias: false, groupSize: 64, bits: 4)
        let linear = QuantizedLinear(
            weight: source.weight,
            scales: source.scales.asType(.bfloat16),
            biases: source.biases?.asType(.bfloat16),
            groupSize: source.groupSize,
            bits: source.bits,
            mode: source.mode)
        let input = MLXRandom.normal([1, 9, 512]).asType(.bfloat16)
        let reference = concatenated(
            (0 ..< input.dim(1)).map { linear(input[0..., $0 ..< ($0 + 1), 0...]) },
            axis: 1)
        let actual = exactSpeculativeLinear(linear, input)
        let expectedTokens = argMax(reference, axis: -1)
        let actualTokens = try #require(exactSpeculativeQuantizedArgMax(linear, input))

        eval(reference, actual, expectedTokens, actualTokens)
        #expect(arrayEqual(reference, actual).item(Bool.self))
        #expect(arrayEqual(expectedTokens, actualTokens).item(Bool.self))
    }

    @Test("Eight-bit block argmax matches exact verification logits")
    func eightBitArgMaxMatchesExactLogits() throws {
        MLXRandom.seed(35)
        let source = QuantizedLinear(512, 16, bias: false, groupSize: 64, bits: 8)
        let linear = QuantizedLinear(
            weight: source.weight,
            scales: source.scales.asType(.bfloat16),
            biases: source.biases?.asType(.bfloat16),
            groupSize: source.groupSize,
            bits: source.bits,
            mode: source.mode)
        let input = MLXRandom.normal([1, 4, 512]).asType(.bfloat16)

        let reference = argMax(exactSpeculativeLinear(linear, input), axis: -1)
        let actual = try #require(exactSpeculativeQuantizedArgMax(linear, input))

        eval(reference, actual)
        #expect(arrayEqual(reference, actual).item(Bool.self))
    }

    @Test("Small decode batches reuse the verification token dimension exactly")
    func batchAsTimeArgMaxMatchesRows() throws {
        MLXRandom.seed(37)
        let source = QuantizedLinear(512, 16, bias: false, groupSize: 64, bits: 4)
        let linear = QuantizedLinear(
            weight: source.weight,
            scales: source.scales.asType(.bfloat16),
            biases: source.biases?.asType(.bfloat16),
            groupSize: source.groupSize,
            bits: source.bits,
            mode: source.mode)
        let input = MLXRandom.normal([3, 1, 512]).asType(.bfloat16)
        let referenceRows = try (0 ..< 3).map { row in
            try #require(
                exactSpeculativeQuantizedArgMax(
                    linear, input[row ..< (row + 1), 0..., 0...]))
        }
        let reference = concatenated(referenceRows, axis: 0)
        let actual = try #require(exactSpeculativeQuantizedArgMax(linear, input))

        eval(reference, actual)
        #expect(arrayEqual(reference, actual).item(Bool.self))
    }

    @Test("Packed token masks constrain argmax and keep padded vocabulary rows disabled")
    func maskedArgMaxUsesEachTokenMaskRow() throws {
        MLXRandom.seed(36)
        let source = QuantizedLinear(512, 16, bias: false, groupSize: 64, bits: 4)
        let linear = QuantizedLinear(
            weight: source.weight,
            scales: source.scales.asType(.bfloat16),
            biases: source.biases?.asType(.bfloat16),
            groupSize: source.groupSize,
            bits: source.bits,
            mode: source.mode)
        let input = MLXRandom.normal([1, 6, 512]).asType(.bfloat16)
        let expected = MLXArray([1, 3, 5, 7, 9, 11]).reshaped(1, 6)
        let tokenMask = MLXArray(
            [1, 3, 5, 7, 9, 11].map { Int32(1) << Int32($0) }
        ).reshaped(6, 1)

        let paddedMask = padExactSpeculativeTokenMask(tokenMask, outputSize: 48)
        let actual = try #require(
            exactSpeculativeQuantizedArgMax(linear, input, tokenMask: tokenMask))

        eval(paddedMask, actual)
        #expect(paddedMask.shape == [6, 2])
        #expect(arrayEqual(paddedMask[0..., 1], MLXArray.zeros([6], dtype: .int32)).item(Bool.self))
        #expect(arrayEqual(expected, actual).item(Bool.self))
    }
}
