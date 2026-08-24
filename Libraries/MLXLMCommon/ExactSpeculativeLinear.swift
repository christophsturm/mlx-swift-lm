import Foundation
import MLX
import MLXNN

private struct ExactSpeculativeQLinearKernelKey: Hashable {
    let bits: Int
    let groupSize: Int
    let dtype: String
    let tokenCount: Int
    let inputSize: Int
    let outputSize: Int
    let tokenTiled: Bool
    let streamed: Bool
}

private struct ExactSpeculativeFusedQLinearKernelKey: Hashable {
    let groupSize: Int
    let dtype: String
    let tokenCount: Int
    let inputSize: Int
    let outputSizes: [Int]
}

private struct ExactSpeculativeQArgMaxKernelKey: Hashable {
    let bits: Int
    let groupSize: Int
    let dtype: String
    let tokenCount: Int
    let inputSize: Int
    let outputSize: Int
    let tokenTiled: Bool
    let streamed: Bool
    let masked: Bool
}

private enum ExactSpeculativeLinearKernels {
    nonisolated(unsafe) private static var quantizedKernels:
        [ExactSpeculativeQLinearKernelKey: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var fusedQuantizedKernels:
        [ExactSpeculativeFusedQLinearKernelKey: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var quantizedArgMaxKernels:
        [ExactSpeculativeQArgMaxKernelKey: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()

    static let dense = MLXFast.metalKernel(
        name: "exact_speculative_verify_gemv",
        inputNames: ["x", "weight"],
        outputNames: ["out"],
        source: denseSource,
        header: "#include <metal_simdgroup>\nusing namespace metal;\n")

    static func quantized(
        bits: Int,
        groupSize: Int,
        dtype: DType,
        tokenCount: Int,
        inputSize: Int,
        outputSize: Int,
        tokenTiled: Bool,
        streamed: Bool
    ) -> MLXFast.MLXFastKernel {
        let key = ExactSpeculativeQLinearKernelKey(
            bits: bits,
            groupSize: groupSize,
            dtype: String(describing: dtype),
            tokenCount: tokenCount,
            inputSize: inputSize,
            outputSize: outputSize,
            tokenTiled: tokenTiled,
            streamed: streamed)

        return lock.withLock {
            if let kernel = quantizedKernels[key] {
                return kernel
            }

            let kernel = MLXFast.metalKernel(
                name:
                    "exact_speculative_qmv_b\(bits)_gs\(groupSize)_t\(tokenCount)_k\(inputSize)_n\(outputSize)_\(streamed ? "streamed" : tokenTiled ? "tiled" : "block")",
                inputNames: ["x", "w", "scales", "biases"],
                outputNames: ["y"],
                source: streamed
                    ? quantizedStreamedSource
                    : tokenTiled ? quantizedTokenTiledSource : quantizedSource,
                header: quantizedHeader(
                    bits: bits,
                    groupSize: groupSize,
                    resultsPerSIMDGroup: streamed ? 1 : 4))
            quantizedKernels[key] = kernel
            return kernel
        }
    }

    static func fusedQuantized(
        groupSize: Int,
        dtype: DType,
        tokenCount: Int,
        inputSize: Int,
        outputSizes: [Int]
    ) -> MLXFast.MLXFastKernel {
        let key = ExactSpeculativeFusedQLinearKernelKey(
            groupSize: groupSize,
            dtype: String(describing: dtype),
            tokenCount: tokenCount,
            inputSize: inputSize,
            outputSizes: outputSizes)

        return lock.withLock {
            if let kernel = fusedQuantizedKernels[key] {
                return kernel
            }

            let inputNames =
                ["x"]
                + outputSizes.indices.flatMap {
                    ["w\($0)", "scales\($0)", "biases\($0)"]
                }
            let shapeName = outputSizes.map(String.init).joined(separator: "_")
            let kernel = MLXFast.metalKernel(
                name:
                    "exact_speculative_fused_qmv_b4_gs\(groupSize)_t\(tokenCount)_k\(inputSize)_n\(shapeName)",
                inputNames: inputNames,
                outputNames: ["y"],
                source: fusedQuantizedSource(outputSizes: outputSizes),
                header: quantizedHeader(bits: 4, groupSize: groupSize))
            fusedQuantizedKernels[key] = kernel
            return kernel
        }
    }

    static func quantizedArgMax(
        bits: Int,
        groupSize: Int,
        dtype: DType,
        tokenCount: Int,
        inputSize: Int,
        outputSize: Int,
        tokenTiled: Bool,
        streamed: Bool,
        masked: Bool
    ) -> MLXFast.MLXFastKernel {
        let key = ExactSpeculativeQArgMaxKernelKey(
            bits: bits,
            groupSize: groupSize,
            dtype: String(describing: dtype),
            tokenCount: tokenCount,
            inputSize: inputSize,
            outputSize: outputSize,
            tokenTiled: tokenTiled,
            streamed: streamed,
            masked: masked)

        return lock.withLock {
            if let kernel = quantizedArgMaxKernels[key] {
                return kernel
            }
            let inputNames = ["x", "w", "scales", "biases"] + (masked ? ["mask"] : [])
            let kernel = MLXFast.metalKernel(
                name:
                    "exact_speculative_qargmax_b\(bits)_gs\(groupSize)_t\(tokenCount)_k\(inputSize)_n\(outputSize)_\(streamed ? "streamed" : tokenTiled ? "tiled" : "block")_\(masked ? "masked" : "plain")",
                inputNames: inputNames,
                outputNames: ["tile_values", "tile_indices"],
                source: quantizedArgMaxSource(
                    tokenTiled: tokenTiled, streamed: streamed, masked: masked),
                header: quantizedHeader(
                    bits: bits,
                    groupSize: groupSize,
                    resultsPerSIMDGroup: streamed ? 1 : 4))
            quantizedArgMaxKernels[key] = kernel
            return kernel
        }
    }

    private static let denseSource = """
        uint lane = thread_position_in_grid.x;
        uint out_block = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;

        constexpr int TM = 4;
        constexpr int TN = 4;
        constexpr int SN = 32;
        constexpr int blockN = SN * TN;

        if (row >= R) return;
        int out_row = int(out_block * TM);
        if (out_row >= O) return;

        const device T* in_vec = x + row * K;
        const device T* mat = weight + out_row * K;
        float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
        int col = int(lane * TN);
        int n_iter = K / blockN;
        int leftover = K - blockN * n_iter;

        for (int iter = 0; iter < n_iter; ++iter) {
          float v[TN];
          for (int tn = 0; tn < TN; ++tn) v[tn] = float(in_vec[col + tn]);
          for (int tm = 0; tm < TM; ++tm) {
            for (int tn = 0; tn < TN; ++tn) {
              result[tm] += float(mat[tm * K + col + tn]) * v[tn];
            }
          }
          col += blockN;
        }

        if (leftover > 0) {
          float v[TN];
          for (int tn = 0; tn < TN; ++tn) {
            v[tn] = col + tn < K ? float(in_vec[col + tn]) : 0.0f;
          }
          for (int tm = 0; tm < TM; ++tm) {
            for (int tn = 0; tn < TN; ++tn) {
              T m = col + tn < K ? mat[tm * K + col + tn] : T(0);
              result[tm] += float(m) * v[tn];
            }
          }
        }

        for (int tm = 0; tm < TM; ++tm) {
          for (ushort sn = SN / 2; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
          }
        }
        if (lane == 0) {
          for (int tm = 0; tm < TM; ++tm) {
            out[row * O + out_row + tm] = T(result[tm]);
          }
        }
        """

    private static func quantizedHeader(
        bits: Int,
        groupSize: Int,
        resultsPerSIMDGroup: Int = 4
    ) -> String {
        """
        using namespace metal;

        constant constexpr int BITS = \(bits);
        constant constexpr int GS = \(groupSize);
        constant constexpr int PACK_FACTOR = (BITS == 5 ? 8 : 32 / BITS);
        constant constexpr int BYTES_PER_PACK = (BITS == 5 ? 5 : 32 / 8);
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
        constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
        constant constexpr int RESULTS_PER_SIMDGROUP = \(resultsPerSIMDGroup);
        constant constexpr int NUM_SIMDGROUPS = 2;
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

        template <typename T>
        inline float load_vector_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          if (BITS == 4) {
            for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1];
              x_thread[i + 2] = x[i + 2];
              x_thread[i + 3] = x[i + 3];
            }
          } else if (BITS == 5) {
            for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
                  x[i + 6] + x[i + 7];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 32.0f;
              x_thread[i + 2] = x[i + 2] / 4.0f;
              x_thread[i + 3] = x[i + 3] / 128.0f;
              x_thread[i + 4] = x[i + 4] / 16.0f;
              x_thread[i + 5] = x[i + 5] / 2.0f;
              x_thread[i + 6] = x[i + 6] / 64.0f;
              x_thread[i + 7] = x[i + 7] / 8.0f;
            }
          } else if (BITS == 8) {
            for (int i = 0; i < VALUES_PER_THREAD; i++) {
              sum += x[i];
              x_thread[i] = x[i];
            }
          }
          return sum;
        }

        inline float qdot_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          float accum = 0.0f;
          if (BITS == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
              uint packed = ws[i];
              accum +=
                  (x_thread[4 * i] * (packed & 0x000f) +
                   x_thread[4 * i + 1] * ((packed >> 4) & 0x000f) +
                   x_thread[4 * i + 2] * ((packed >> 8) & 0x000f) +
                   x_thread[4 * i + 3] * ((packed >> 12) & 0x000f));
            }
          } else if (BITS == 5) {
            for (int i = 0; i < (VALUES_PER_THREAD / 8); i++) {
              const thread float* xt = x_thread + 8 * i;
              const device uint8_t* wb = w + 5 * i;

              accum += (wb[0] & 0x1f) * xt[0];
              accum += (wb[0] & 0xe0) * xt[1];
              accum += (wb[1] & 0x3) * (xt[1] * 256.0f);
              accum += (wb[1] & 0x7c) * xt[2];
              accum += (wb[1] & 0x80) * xt[3];
              accum += (wb[2] & 0xf) * (xt[3] * 256.0f);
              accum += (wb[2] & 0xf0) * xt[4];
              accum += (wb[3] & 0x1) * (xt[4] * 256.0f);
              accum += (wb[3] & 0x3e) * xt[5];
              accum += (wb[3] & 0xc0) * xt[6];
              accum += (wb[4] & 0x7) * (xt[6] * 256.0f);
              accum += (wb[4] & 0xf8) * xt[7];
            }
          } else if (BITS == 8) {
            for (int i = 0; i < VALUES_PER_THREAD; i++) {
              accum += x_thread[i] * w[i];
            }
          }
          return scale * accum + sum * bias;
        }

        inline float qdot_exact(
            const thread uint16_t* ws,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          float accum = 0.0f;
          if (BITS == 4) {
            for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
              uint packed = ws[i];
              accum +=
                  (x_thread[4 * i] * (packed & 0x000f) +
                   x_thread[4 * i + 1] * ((packed >> 4) & 0x000f) +
                   x_thread[4 * i + 2] * ((packed >> 8) & 0x000f) +
                   x_thread[4 * i + 3] * ((packed >> 12) & 0x000f));
            }
          }
          return scale * accum + sum * bias;
        }
        """
    }

    private static let quantizedSource = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint b_idx = threadgroup_position_in_grid.z;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K_SIZE / GS;

        const device uint8_t* ws =
            (const device uint8_t*)w + out_row * in_vec_size_w +
            int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sc =
            scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bs =
            biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* xk =
            x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

        float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
        float x_thread[VERIFY_T][VALUES_PER_THREAD];
        for (int t = 0; t < VERIFY_T; ++t) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            result[t][row] = 0.0f;
          }
        }

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
          float sums[VERIFY_T];
          for (int t = 0; t < VERIFY_T; ++t) {
            sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
          }
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            const device T* sl = sc + row * in_vec_size_g;
            const device T* bl = bs + row * in_vec_size_g;
            float s = float(sl[0]);
            float b = float(bl[0]);
            for (int t = 0; t < VERIFY_T; ++t) {
              result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
            }
          }

          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          xk += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          for (int t = 0; t < VERIFY_T; ++t) {
            float r = simd_sum(result[t][row]);
            if (simd_lid == 0) {
              y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(r);
            }
          }
        }
        """

    private static let quantizedTokenTiledSource = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint token_tile = threadgroup_position_in_grid.z;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        constexpr int TOKEN_TILE = 2;
        constexpr int NUM_TOKEN_TILES = (VERIFY_T + TOKEN_TILE - 1) / TOKEN_TILE;
        int b_idx = int(token_tile) / NUM_TOKEN_TILES;
        int token_tile_idx = int(token_tile) - b_idx * NUM_TOKEN_TILES;
        int token_start = token_tile_idx * TOKEN_TILE;
        int token_count = min(TOKEN_TILE, VERIFY_T - token_start);
        int flat_token_start = b_idx * VERIFY_T + token_start;
        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K_SIZE / GS;

        const device uint8_t* ws =
            (const device uint8_t*)w + out_row * in_vec_size_w +
            int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sc =
            scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bs =
            biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* xk =
            x + flat_token_start * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

        float result[TOKEN_TILE][RESULTS_PER_SIMDGROUP];
        float x_thread[TOKEN_TILE][VALUES_PER_THREAD];
        for (int t = 0; t < TOKEN_TILE; ++t) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            result[t][row] = 0.0f;
          }
        }

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
          float sums[TOKEN_TILE];
          for (int t = 0; t < token_count; ++t) {
            sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
          }
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            const device T* sl = sc + row * in_vec_size_g;
            const device T* bl = bs + row * in_vec_size_g;
            float s = float(sl[0]);
            float b = float(bl[0]);
            for (int t = 0; t < token_count; ++t) {
              result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
            }
          }

          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          xk += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          for (int t = 0; t < token_count; ++t) {
            float r = simd_sum(result[t][row]);
            if (simd_lid == 0) {
              y[(flat_token_start + t) * N_SIZE + n] = T(r);
            }
          }
        }
        """

    private static let quantizedStreamedSource = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint b_idx = threadgroup_position_in_grid.z;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K_SIZE / GS;

        const device uint8_t* ws =
            (const device uint8_t*)w + out_row * in_vec_size_w +
            int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sc =
            scales + out_row * in_vec_size_g +
            int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bs =
            biases + out_row * in_vec_size_g +
            int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* xk =
            x + int(b_idx) * VERIFY_T * K_SIZE +
            int(simd_lid) * VALUES_PER_THREAD;

        float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
        for (int t = 0; t < VERIFY_T; ++t) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            result[t][row] = 0.0f;
          }
        }

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            const device uint16_t* source =
                (const device uint16_t*)(ws + row * in_vec_size_w);
            uint16_t packed[VALUES_PER_THREAD / 4];
            for (int index = 0; index < VALUES_PER_THREAD / 4; ++index) {
              packed[index] = source[index];
            }
            float scale = float(sc[row * in_vec_size_g]);
            float bias = float(bs[row * in_vec_size_g]);
            for (int t = 0; t < VERIFY_T; ++t) {
              float x_thread[VALUES_PER_THREAD];
              float sum = load_vector_exact<T>(xk + t * K_SIZE, x_thread);
              result[t][row] += qdot_exact(packed, x_thread, scale, bias, sum);
            }
          }

          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          xk += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          for (int t = 0; t < VERIFY_T; ++t) {
            float value = simd_sum(result[t][row]);
            if (simd_lid == 0) {
              y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(value);
            }
          }
        }
        """

    private static func fusedQuantizedSource(outputSizes: [Int]) -> String {
        var offset = outputSizes[0]
        let selections = outputSizes.dropFirst().enumerated().map { index, outputSize in
            defer { offset += outputSize }
            let linear = index + 1
            return """
                    if (global_out >= \(offset)) {
                      local_out = global_out - \(offset);
                      selected_w = (const device uint8_t*)w\(linear);
                      selected_scales = scales\(linear);
                      selected_biases = biases\(linear);
                    }
                """
        }.joined(separator: "\n")

        return """
            uint n_tile = threadgroup_position_in_grid.y;
            uint b_idx = threadgroup_position_in_grid.z;
            uint simd_gid = simdgroup_index_in_threadgroup;
            uint simd_lid = thread_index_in_simdgroup;

            int global_out = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
            int local_out = global_out;
            int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
            int in_vec_size_g = K_SIZE / GS;
            const device uint8_t* selected_w = (const device uint8_t*)w0;
            const device T* selected_scales = scales0;
            const device T* selected_biases = biases0;
            \(selections)

            const device uint8_t* ws =
                selected_w + local_out * in_vec_size_w +
                int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
            const device T* sc =
                selected_scales + local_out * in_vec_size_g +
                int(simd_lid) / SCALE_STEP_PER_THREAD;
            const device T* bs =
                selected_biases + local_out * in_vec_size_g +
                int(simd_lid) / SCALE_STEP_PER_THREAD;
            const device T* xk =
                x + int(b_idx) * VERIFY_T * K_SIZE +
                int(simd_lid) * VALUES_PER_THREAD;

            float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
            float x_thread[VERIFY_T][VALUES_PER_THREAD];
            for (int t = 0; t < VERIFY_T; ++t) {
              for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) result[t][row] = 0.0f;
            }

            for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
              float sums[VERIFY_T];
              for (int t = 0; t < VERIFY_T; ++t) {
                sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
              }
              for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                const device uint8_t* wl = ws + row * in_vec_size_w;
                const device T* sl = sc + row * in_vec_size_g;
                const device T* bl = bs + row * in_vec_size_g;
                float s = float(sl[0]);
                float b = float(bl[0]);
                for (int t = 0; t < VERIFY_T; ++t) {
                  result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
                }
              }
              ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
              sc += BLOCK_SIZE / GS;
              bs += BLOCK_SIZE / GS;
              xk += BLOCK_SIZE;
            }

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
              int n = global_out + row;
              for (int t = 0; t < VERIFY_T; ++t) {
                float r = simd_sum(result[t][row]);
                if (simd_lid == 0) {
                  y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(r);
                }
              }
            }
            """
    }

    private static func quantizedArgMaxSource(
        tokenTiled: Bool,
        streamed: Bool,
        masked: Bool
    ) -> String {
        let source =
            streamed
            ? quantizedArgMaxStreamedSource
            : tokenTiled ? quantizedArgMaxTokenTiledSource : quantizedArgMaxBlockSource
        guard masked else { return source }
        if streamed {
            return source.replacingOccurrences(
                of: "if (n < N_SIZE && rounded > best_value) {",
                with:
                    "if (n < N_SIZE && ((as_type<uint>(mask[(int(b_idx) * VERIFY_T + t) * MASK_WORDS + (n >> 5)]) >> (n & 31)) & 1u) != 0u && rounded > best_value) {"
            )
        }
        let row = tokenTiled ? "flat_token_start + t" : "int(b_idx) * VERIFY_T + t"
        return source.replacingOccurrences(
            of: "if (n < N_SIZE) {",
            with:
                "if (n < N_SIZE && ((as_type<uint>(mask[(\(row)) * MASK_WORDS + (n >> 5)]) >> (n & 31)) & 1u) != 0u) {"
        )
    }

    private static let quantizedArgMaxBlockSource = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint b_idx = threadgroup_position_in_grid.z;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K_SIZE / GS;
        threadgroup float tile_best_values[VERIFY_T][NUM_SIMDGROUPS];
        threadgroup int tile_best_indices[VERIFY_T][NUM_SIMDGROUPS];

        const device uint8_t* ws =
            (const device uint8_t*)w + out_row * in_vec_size_w +
            int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sc =
            scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bs =
            biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* xk =
            x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

        float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
        float x_thread[VERIFY_T][VALUES_PER_THREAD];
        for (int t = 0; t < VERIFY_T; ++t) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) result[t][row] = 0.0f;
        }

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
          float sums[VERIFY_T];
          for (int t = 0; t < VERIFY_T; ++t) {
            sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
          }
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            const device T* sl = sc + row * in_vec_size_g;
            const device T* bl = bs + row * in_vec_size_g;
            float s = float(sl[0]);
            float b = float(bl[0]);
            for (int t = 0; t < VERIFY_T; ++t) {
              result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
            }
          }
          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          xk += BLOCK_SIZE;
        }

        for (int t = 0; t < VERIFY_T; ++t) {
          float best_value = -3.4028234663852886e38f;
          int best_index = 0;
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N_SIZE) {
              float rounded = float(T(simd_sum(result[t][row])));
              if (rounded > best_value) {
                best_value = rounded;
                best_index = n;
              }
            }
          }
          if (simd_lid == 0) {
            tile_best_values[t][simd_gid] = best_value;
            tile_best_indices[t][simd_gid] = best_index;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_gid == 0 && simd_lid == 0) {
          for (int t = 0; t < VERIFY_T; ++t) {
            float best = tile_best_values[t][0];
            int best_idx = tile_best_indices[t][0];
            for (int i = 1; i < NUM_SIMDGROUPS; ++i) {
              if (tile_best_values[t][i] > best) {
                best = tile_best_values[t][i];
                best_idx = tile_best_indices[t][i];
              }
            }
            int offset = (int(b_idx) * VERIFY_T + t) * NUM_TILES + int(n_tile);
            tile_values[offset] = T(best);
            tile_indices[offset] = best_idx;
          }
        }
        """

    private static let quantizedArgMaxTokenTiledSource = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint token_tile = threadgroup_position_in_grid.z;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        constexpr int TOKEN_TILE = 2;
        constexpr int NUM_TOKEN_TILES = (VERIFY_T + TOKEN_TILE - 1) / TOKEN_TILE;
        int b_idx = int(token_tile) / NUM_TOKEN_TILES;
        int token_tile_idx = int(token_tile) - b_idx * NUM_TOKEN_TILES;
        int token_start = token_tile_idx * TOKEN_TILE;
        int token_count = min(TOKEN_TILE, VERIFY_T - token_start);
        int flat_token_start = b_idx * VERIFY_T + token_start;
        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K_SIZE / GS;
        threadgroup float tile_best_values[TOKEN_TILE][NUM_SIMDGROUPS];
        threadgroup int tile_best_indices[TOKEN_TILE][NUM_SIMDGROUPS];

        const device uint8_t* ws =
            (const device uint8_t*)w + out_row * in_vec_size_w +
            int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sc =
            scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bs =
            biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* xk =
            x + flat_token_start * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

        float result[TOKEN_TILE][RESULTS_PER_SIMDGROUP];
        float x_thread[TOKEN_TILE][VALUES_PER_THREAD];
        for (int t = 0; t < TOKEN_TILE; ++t) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) result[t][row] = 0.0f;
        }

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
          float sums[TOKEN_TILE];
          for (int t = 0; t < token_count; ++t) {
            sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
          }
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            const device T* sl = sc + row * in_vec_size_g;
            const device T* bl = bs + row * in_vec_size_g;
            float s = float(sl[0]);
            float b = float(bl[0]);
            for (int t = 0; t < token_count; ++t) {
              result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
            }
          }
          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          xk += BLOCK_SIZE;
        }

        for (int t = 0; t < token_count; ++t) {
          float best_value = -3.4028234663852886e38f;
          int best_index = 0;
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N_SIZE) {
              float rounded = float(T(simd_sum(result[t][row])));
              if (rounded > best_value) {
                best_value = rounded;
                best_index = n;
              }
            }
          }
          if (simd_lid == 0) {
            tile_best_values[t][simd_gid] = best_value;
            tile_best_indices[t][simd_gid] = best_index;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_gid == 0 && simd_lid == 0) {
          for (int t = 0; t < token_count; ++t) {
            float best = tile_best_values[t][0];
            int best_idx = tile_best_indices[t][0];
            for (int i = 1; i < NUM_SIMDGROUPS; ++i) {
              if (tile_best_values[t][i] > best) {
                best = tile_best_values[t][i];
                best_idx = tile_best_indices[t][i];
              }
            }
            int offset = (flat_token_start + t) * NUM_TILES + int(n_tile);
            tile_values[offset] = T(best);
            tile_indices[offset] = best_idx;
          }
        }
        """

    private static let quantizedArgMaxStreamedSource = """
        uint n_tile = threadgroup_position_in_grid.y;
        uint b_idx = threadgroup_position_in_grid.z;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K_SIZE / GS;
        threadgroup float tile_best_values[VERIFY_T][NUM_SIMDGROUPS];
        threadgroup int tile_best_indices[VERIFY_T][NUM_SIMDGROUPS];

        const device uint8_t* ws =
            (const device uint8_t*)w + out_row * in_vec_size_w +
            int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sc =
            scales + out_row * in_vec_size_g +
            int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bs =
            biases + out_row * in_vec_size_g +
            int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* xk =
            x + int(b_idx) * VERIFY_T * K_SIZE +
            int(simd_lid) * VALUES_PER_THREAD;

        float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
        for (int t = 0; t < VERIFY_T; ++t) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            result[t][row] = 0.0f;
          }
        }

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            const device uint16_t* source =
                (const device uint16_t*)(ws + row * in_vec_size_w);
            uint16_t packed[VALUES_PER_THREAD / 4];
            for (int index = 0; index < VALUES_PER_THREAD / 4; ++index) {
              packed[index] = source[index];
            }
            float scale = float(sc[row * in_vec_size_g]);
            float bias = float(bs[row * in_vec_size_g]);
            for (int t = 0; t < VERIFY_T; ++t) {
              float x_thread[VALUES_PER_THREAD];
              float sum = load_vector_exact<T>(xk + t * K_SIZE, x_thread);
              result[t][row] += qdot_exact(packed, x_thread, scale, bias, sum);
            }
          }

          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          xk += BLOCK_SIZE;
        }

        for (int t = 0; t < VERIFY_T; ++t) {
          float best_value = -3.4028234663852886e38f;
          int best_index = 0;
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            float rounded = float(T(simd_sum(result[t][row])));
            if (n < N_SIZE && rounded > best_value) {
              best_value = rounded;
              best_index = n;
            }
          }

          if (simd_lid == 0) {
            tile_best_values[t][simd_gid] = best_value;
            tile_best_indices[t][simd_gid] = best_index;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0 && simd_lid == 0) {
          for (int t = 0; t < VERIFY_T; ++t) {
            float best = tile_best_values[t][0];
            int best_idx = tile_best_indices[t][0];
            for (int i = 1; i < NUM_SIMDGROUPS; ++i) {
              float candidate = tile_best_values[t][i];
              int candidate_idx = tile_best_indices[t][i];
              if (candidate > best) {
                best = candidate;
                best_idx = candidate_idx;
              }
            }
            int offset =
                (int(b_idx) * VERIFY_T + t) * NUM_TILES + int(n_tile);
            tile_values[offset] = T(best);
            tile_indices[offset] = best_idx;
          }
        }
        """
}

/// Applies a linear projection to a speculative token block with singleton-equivalent accumulation.
package func exactSpeculativeLinear(_ linear: Linear, _ input: MLXArray) -> MLXArray {
    guard input.ndim == 3, input.dim(1) > 1 else {
        return linear(input)
    }

    #if canImport(Metal)
    if let quantized = linear as? QuantizedLinear,
        let projected = exactQuantizedSpeculativeLinear(quantized, input)
    {
        return projected
    }
    if !(linear is QuantizedLinear), linear.bias == nil,
        let projected = exactDenseSpeculativeWeight(linear.weight, input)
    {
        return projected
    }
    #endif

    return exactSingletonLinear(linear, input)
}

/// Applies related projections through one exact quantized kernel when their layouts agree.
package func exactSpeculativeLinears(
    _ linears: [Linear], _ input: MLXArray
) -> [MLXArray] {
    guard input.ndim == 3, input.dim(1) > 1 else {
        return linears.map { $0(input) }
    }
    #if canImport(Metal)
    if let projected = exactFusedQuantizedSpeculativeLinears(linears, input) {
        return projected
    }
    #endif
    return linears.map { exactSpeculativeLinear($0, input) }
}

/// Reports whether a quantized projection can fuse affine dequantization with greedy sampling.
package func canExactSpeculativeQuantizedArgMax(_ linear: Linear) -> Bool {
    #if canImport(Metal)
    guard let quantized = linear as? QuantizedLinear,
        [4, 5, 8].contains(quantized.bits),
        quantized.mode == .affine,
        quantized.bias == nil,
        let quantizationBiases = quantized.biases,
        quantized.scales.dtype == .bfloat16 || quantized.scales.dtype == .float16,
        quantizationBiases.dtype == quantized.scales.dtype
    else {
        return false
    }
    let inputSize = quantized.weight.dim(1) * 32 / quantized.bits
    return inputSize.isMultiple(of: 512) && quantized.weight.dim(0).isMultiple(of: 8)
    #else
    return false
    #endif
}

/// Samples an affine quantized projection without materializing its full vocabulary logits.
package func exactSpeculativeQuantizedArgMax(
    _ linear: Linear,
    _ input: MLXArray,
    tokenMask providedTokenMask: MLXArray? = nil
) -> MLXArray? {
    #if canImport(Metal)
    guard canExactSpeculativeQuantizedArgMax(linear),
        let quantized = linear as? QuantizedLinear,
        let quantizationBiases = quantized.biases,
        input.ndim == 3,
        input.dtype == quantized.scales.dtype
    else {
        return nil
    }

    let batchSize = input.dim(0)
    let tokenCount = input.dim(1)
    let inputSize = quantized.weight.dim(1) * 32 / quantized.bits
    let outputSize = quantized.weight.dim(0)
    guard tokenCount > 0, input.dim(2) == inputSize else { return nil }

    var tokenMask = providedTokenMask
    if tokenMask?.ndim == 1 {
        tokenMask = tokenMask?.expandedDimensions(axis: 0)
    }
    if let tokenMask {
        guard tokenMask.dtype == .int32,
            tokenMask.dim(0) == batchSize * tokenCount,
            tokenMask.dim(1) >= (outputSize + 31) / 32
        else {
            return nil
        }
    }

    if tokenCount == 1, (2 ... 4).contains(batchSize),
        let transposed = exactSpeculativeQuantizedArgMax(
            linear, input.transposed(1, 0, 2), tokenMask: tokenMask)
    {
        return transposed.transposed(1, 0)
    }

    let streamed = quantized.bits == 4 && (6 ... 8).contains(tokenCount)
    let tokenTiled = quantized.bits == 4 && tokenCount >= 6 && !streamed
    let rowsPerThreadgroup = streamed ? 2 : 8
    let tileCount = outputSize / rowsPerThreadgroup
    let kernel = ExactSpeculativeLinearKernels.quantizedArgMax(
        bits: quantized.bits,
        groupSize: quantized.groupSize,
        dtype: input.dtype,
        tokenCount: tokenCount,
        inputSize: inputSize,
        outputSize: outputSize,
        tokenTiled: tokenTiled,
        streamed: streamed,
        masked: tokenMask != nil)
    var inputs = [input.contiguous(), quantized.weight, quantized.scales, quantizationBiases]
    if let tokenMask {
        inputs.append(tokenMask)
    }
    let output = kernel(
        inputs,
        template: [
            ("T", input.dtype),
            ("VERIFY_T", tokenCount),
            ("K_SIZE", inputSize),
            ("N_SIZE", outputSize),
            ("NUM_TILES", tileCount),
            ("MASK_WORDS", tokenMask?.dim(1) ?? 0),
        ],
        grid: (
            32,
            2 * tileCount,
            tokenTiled ? batchSize * ((tokenCount + 1) / 2) : batchSize
        ),
        threadGroup: (32, 2, 1),
        outputShapes: [
            [batchSize, tokenCount, tileCount],
            [batchSize, tokenCount, tileCount],
        ],
        outputDTypes: [input.dtype, .int32]
    )
    let bestTile = argMax(output[0], axis: -1)
    return takeAlong(output[1], bestTile[.ellipsis, .newAxis], axis: -1).squeezed(axis: -1)
    #else
    return nil
    #endif
}

/// Pads a tokenizer-sized packed bitmask so quantized vocabulary padding remains disallowed.
package func padExactSpeculativeTokenMask(
    _ tokenMask: MLXArray, outputSize: Int
) -> MLXArray {
    let mask = tokenMask.ndim == 1 ? tokenMask.expandedDimensions(axis: 0) : tokenMask
    let requiredWords = (outputSize + 31) / 32
    let missingWords = requiredWords - mask.dim(1)
    guard missingWords > 0 else { return mask }
    return concatenated(
        [mask, MLXArray.zeros([mask.dim(0), missingWords], dtype: .int32)], axis: 1)
}

/// Projects tied embeddings without changing decode-time accumulation order.
package func exactSpeculativeEmbeddingProjection(
    _ embedding: Embedding, _ input: MLXArray
) -> MLXArray {
    guard input.ndim == 3, input.dim(1) > 1 else {
        return embedding.asLinear(input)
    }
    #if canImport(Metal)
    if let projected = exactDenseSpeculativeWeight(embedding.weight, input) {
        return projected
    }
    #endif
    return concatenated(
        (0 ..< input.dim(1)).map { token in
            embedding.asLinear(input[0..., token ..< (token + 1), 0...])
        },
        axis: 1)
}

#if canImport(Metal)
private func exactDenseSpeculativeWeight(
    _ weight: MLXArray,
    _ input: MLXArray
) -> MLXArray? {
    guard weight.ndim == 2, input.ndim == 3, weight.dtype == input.dtype,
        input.dtype == .bfloat16 || input.dtype == .float16,
        weight.dim(1) == input.dim(2)
    else {
        return nil
    }

    let batchSize = input.dim(0)
    let tokenCount = input.dim(1)
    let inputSize = input.dim(2)
    let outputSize = weight.dim(0)
    guard outputSize >= 4, outputSize.isMultiple(of: 4), inputSize < 16 * outputSize else {
        return nil
    }

    let rows = batchSize * tokenCount
    let paddedRows = ((rows + 7) / 8) * 8
    let projected = ExactSpeculativeLinearKernels.dense(
        [input.reshaped(rows, inputSize), weight],
        template: [
            ("T", input.dtype),
            ("K", inputSize),
            ("O", outputSize),
            ("R", rows),
        ],
        grid: (32, outputSize / 4, paddedRows),
        threadGroup: (32, 1, 8),
        outputShapes: [[rows, outputSize]],
        outputDTypes: [input.dtype]
    )[0]
    return projected.reshaped(batchSize, tokenCount, outputSize)
}

private func exactFusedQuantizedSpeculativeLinears(
    _ linears: [Linear],
    _ input: MLXArray
) -> [MLXArray]? {
    guard (2 ... 4).contains(linears.count), (2 ..< 6).contains(input.dim(1)) else {
        return nil
    }
    let quantized = linears.compactMap { $0 as? QuantizedLinear }
    guard quantized.count == linears.count, let first = quantized.first,
        first.bits == 4, first.mode == .affine, first.bias == nil,
        quantized.allSatisfy({ linear in
            linear.bits == 4 && linear.groupSize == first.groupSize
                && linear.mode == first.mode && linear.bias == nil
                && linear.biases != nil
                && linear.scales.dtype == input.dtype
                && linear.biases?.dtype == input.dtype
                && input.dim(2) == linear.weight.dim(1) * 8
                && input.dim(2).isMultiple(of: 512)
                && linear.weight.dim(0).isMultiple(of: 8)
        })
    else {
        return nil
    }

    let batchSize = input.dim(0)
    let tokenCount = input.dim(1)
    let inputSize = input.dim(2)
    let outputSizes = quantized.map { $0.weight.dim(0) }
    let totalOutputSize = outputSizes.reduce(0, +)
    let kernel = ExactSpeculativeLinearKernels.fusedQuantized(
        groupSize: first.groupSize,
        dtype: input.dtype,
        tokenCount: tokenCount,
        inputSize: inputSize,
        outputSizes: outputSizes)
    var inputs = [input.contiguous()]
    for linear in quantized {
        inputs.append(linear.weight)
        inputs.append(linear.scales)
        inputs.append(linear.biases!)
    }
    let projected = kernel(
        inputs,
        template: [
            ("T", input.dtype),
            ("VERIFY_T", tokenCount),
            ("K_SIZE", inputSize),
            ("N_SIZE", totalOutputSize),
        ],
        grid: (32, 2 * (totalOutputSize / 8), batchSize),
        threadGroup: (32, 2, 1),
        outputShapes: [[batchSize, tokenCount, totalOutputSize]],
        outputDTypes: [input.dtype]
    )[0]
    var offset = 0
    let splitIndices = outputSizes.dropLast().map { size in
        offset += size
        return offset
    }
    return MLX.split(projected, indices: splitIndices, axis: -1)
}

private func exactQuantizedSpeculativeLinear(
    _ linear: QuantizedLinear,
    _ input: MLXArray
) -> MLXArray? {
    guard [4, 5, 8].contains(linear.bits),
        linear.mode == .affine,
        let quantizationBiases = linear.biases,
        linear.scales.dtype == .bfloat16 || linear.scales.dtype == .float16,
        quantizationBiases.dtype == linear.scales.dtype,
        input.dtype == linear.scales.dtype
    else {
        return nil
    }

    let batchSize = input.dim(0)
    let tokenCount = input.dim(1)
    let inputSize = linear.weight.dim(1) * 32 / linear.bits
    let outputSize = linear.weight.dim(0)
    guard input.dim(2) == inputSize, inputSize.isMultiple(of: 512),
        outputSize.isMultiple(of: 8)
    else {
        return nil
    }

    let streamed = linear.bits == 4 && (6 ... 8).contains(tokenCount)
    let tokenTiled = linear.bits == 4 && tokenCount >= 6 && !streamed
    let rowsPerThreadgroup = streamed ? 2 : 8
    let kernel = ExactSpeculativeLinearKernels.quantized(
        bits: linear.bits,
        groupSize: linear.groupSize,
        dtype: input.dtype,
        tokenCount: tokenCount,
        inputSize: inputSize,
        outputSize: outputSize,
        tokenTiled: tokenTiled,
        streamed: streamed)
    let projected = kernel(
        [input.contiguous(), linear.weight, linear.scales, quantizationBiases],
        template: [
            ("T", input.dtype),
            ("VERIFY_T", tokenCount),
            ("K_SIZE", inputSize),
            ("N_SIZE", outputSize),
        ],
        grid: (
            32,
            2 * (outputSize / rowsPerThreadgroup),
            tokenTiled ? batchSize * ((tokenCount + 1) / 2) : batchSize
        ),
        threadGroup: (32, 2, 1),
        outputShapes: [[batchSize, tokenCount, outputSize]],
        outputDTypes: [input.dtype]
    )[0]

    if let bias = linear.bias {
        return projected + bias
    }
    return projected
}
#endif

private func exactSingletonLinear(_ linear: Linear, _ input: MLXArray) -> MLXArray {
    let batches = (0 ..< input.dim(0)).map { batch in
        concatenated(
            (0 ..< input.dim(1)).map { token in
                linear(input[batch ..< (batch + 1), token ..< (token + 1), 0...])
            },
            axis: 1)
    }
    return concatenated(batches, axis: 0)
}
