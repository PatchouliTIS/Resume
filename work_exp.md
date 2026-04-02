- Ampere INT8
    
    ```python
    vLLM INT8 Blockwise W8A8 量化推理优化
    角色：推理引擎性能优化工程师
    技术栈：CUDA / CUTLASS / Triton / PyTorch / C++ / Python
    代码量：~7200 行（CUDA 2200+，Triton/Python 3500+，Benchmark 1500+）
    
    项目背景
    在大语言模型（LLM）推理场景中，INT8 量化是降低显存占用和提升吞吐的关键手段。本项目在 vLLM 推理引擎中设计并实现了一套完整的 INT8 Blockwise W8A8 量化方案，覆盖从 activation 在线量化、CUTLASS GEMM kernel、到端到端推理流程的全栈优化。
    
    核心工作
    1. 设计并实现 Hybrid 二级 Scale 量化算法
    
    提出 两级 Scale 量化方案（Two-Level Scale Quantization），将 FP32 blockwise scale 分解为 (Q_int32, F_fp32) 对，使 GEMM 主循环全程保持在 INT32 算术中，避免了传统方案中频繁的 INT32→FP32→INT32 类型转换开销。
    设计了 Super-Group 分组机制，在 K 维度对 scale block 进行二次聚合，以可控的精度损失换取更高的计算效率。
    2. 基于 CUTLASS 开发定制化 INT8 GEMM Kernel（2200+ 行 CUDA/C++）
    
    基于 CUTLASS 2.x 框架，针对 Ampere (SM80) 架构开发了 3 种 CTA Tile 配置的 hybrid GEMM kernel：
    hybrid（128×64×128）：通用场景
    hybrid_small（64×64×128）：小 batch（decode 阶段）
    hybrid_large（128×128×64）：大 batch（prefill 阶段）
    定制化 MMA 主循环：修改 CUTLASS 的 MmaMultistage 模块，在 GEMM 主循环中按 quant block 边界累加 INT32 partial sum 并乘以 Q_A * Q_B 二级 scale，仅在 epilogue 阶段做一次 INT32→FP32 转换并乘以 F_A * F_B 因子。
    CTA Swizzle 优化：实现 L2 cache 友好的 CTA 调度策略（compute_log_swizzle），提升大矩阵场景下的缓存命中率。
    支持 bias fuse 到 epilogue 中，减少额外的 kernel launch。
    3. Triton 加速的 Activation 在线量化流水线（1600+ 行 Python/Triton）
    
    针对 activation 在线量化的 PyTorch eager dispatch 瓶颈（占 CUTLASS 路径 ~66% 延迟），设计并实现了三级融合策略：
    
    Two-Kernel 路径：将 blockwise INT8 量化 + 二级 scale 量化的两个 Triton kernel 预分配全部输出 tensor 后背靠背执行，消除 Python 函数调用和中间 tensor 分配开销。
    Cooperative 单 Kernel 路径：通过 atomic_add 计数器实现 CTA 间 barrier，将量化和 scale 计算融合进单次 kernel launch。针对小 M 场景实现约 1.3× 加速。计数器在 kernel 内部自动重置，避免 host 端 zero_() 调用。
    Adaptive 自适应调度：根据运行时 m_blocks 与 2 × SM_COUNT 的关系自动切换 coop（小 M，launch 开销敏感）和 two-kernel（大 M，DRAM throughput 优先）路径。
    4. 运行时 Kernel 动态选择机制
    
    实现 hybrid_adaptive 模式，根据运行时输入 M 维度动态在 hybrid_small（M ≤ 512）和 hybrid（M > 512）之间切换，适配 LLM 推理中 decode（小 batch）与 prefill（大 batch）交替的特点。
    动态选择 variant 对应的 block_m 和 tile_k，确保 activation 量化分块和 K-padding 逻辑正确适配。
    5. 工程化与端到端集成
    
    完成从 CUDA kernel → C++ torch binding → Python torch.ops 注册 → FakeTensor 支持（torch.compile 兼容）→ vLLM 量化插件的全链路集成。
    实现 K 维度自动对齐 padding（满足 CUTLASS 128-bit cp.async / ldg.128 的 16-byte 对齐要求），scale tensor 按 ceil_div sizing 自适应扩展。
    提供环境变量 VLLM_INT8_HYBRID_FUSED_QUANT 控制 fused/unfused 路径切换，便于线上 A/B 测试和问题排查。
    编写了 4 套 benchmark 工具，覆盖分阶段延迟对比、CUTLASS vs Triton 端到端对比等场景。
    ```
    
    ## 高性能 INT8 Blockwise Quantized GEMM Kernel 优化
    
    **项目背景**：针对大语言模型推理中的 INT8 block-wise quantized GEMM 场景，基于 CUTLASS 模板框架设计并优化 Hybrid Fused Dequant Kernel，解决量化算子中 dequant 开销过高、L2 Cache Locality 差等问题。
    
    ---
    
    ### 1. Scale 二次量化算法设计 — 全 INT32 Mainloop
    
    **问题**：传统 blockwise dequant 算子在每个 quant block 边界需要 `I2F + FMUL` 将 INT32 accumulator 转为 FP32 并乘以 scale，导致：
    
    - **XU Pipeline 被占用**：`I2F` 指令消耗 XU（特殊功能单元），与 epilogue 竞争
    - **FPU Pipeline 被占用**：`FMUL` 指令消耗 FPU，无法与 Tensor Core 完美 overlap
    - **FP32 累加精度损失**：INT32 → FP32 转换会丢失精度
    
    **方案**：设计两级 Scale 量化算法，将 FP32 scale 分解为 `S = F × Q`：
    
    ```
    
    Host 预处理: S_k (FP32) = F_g (FP32 per-super-group) × Q_k (INT32 per-quant-block)
    ```
    
    **Kernel 实现**：
    
    ```
    
    // Mainloop: 纯 INT32 域累加，零 FP 指令
    int q_combined = Q_A[m, qb] * Q_B[n, qb];           // INT32 MUL
    int32_weighted[i] += q_combined * int32_accum[i];    // INT32 IMAD
    
    // Epilogue: 整个 K 维度仅做一次 I2F + FMUL
    fp32_output[i] = static_cast<float>(int32_weighted[i]) * F_final;
    ```
    
    **效果对比**：
    
    | 方案 | Mainloop 操作 | I2F 次数/element | FMUL 次数/element | 精度 |
    | --- | --- | --- | --- | --- |
    | 朴素 FP32 | `fp32 += I2F(acc) × scale` | K/bq 次 | K/bq 次 | FP32 累加误差 |
    | Magic I2F | `fp32 += magic(acc) × scale` | 0 | K/bq 次 | 需要 bias 修正 |
    | **Hybrid INT32** | `int32_w += Q × acc` | **1 次** | **1 次** | INT32 无误差累加 |
    
    **收益**：将 dequant 指令从 FPU/XU pipeline 转移到 INT pipeline，与 Tensor Core IMMA 实现 **完美 pipeline overlap**，无 stall。
    
    ---
    
    ### 2. Threadblock Swizzle — L2 Cache Locality 优化
    
    **问题**：朴素 `grid(tiled_m, tiled_n)` 映射下，相邻 `blockIdx.x` 的 CTA 在 M 维度连续，各自读取不同的 A 行段。当 M 很大时（如 32768），A 的跨度（20MB+）远超 L2 容量，导致 B 矩阵被反复 evict，L2 hit rate 仅 ~50%。
    
    **方案**：实现 `GemmIdentityThreadblockSwizzle` 等效逻辑，将 N 维度的连续 tile 个 CTA 折叠到 `grid.x` 低位：
    
    ```
    
    // Host: grid 重排
    int log_swizzle = min(3, floor(log2(tiled_n)));
    dim3 grid(tiled_m << log_swizzle, (tiled_n + (1 << log_swizzle) - 1) >> log_swizzle);
    
    // Kernel: blockIdx 解码
    int cta_m = blockIdx.x >> log_swizzle;
    int cta_n = (blockIdx.y << log_swizzle) + (blockIdx.x & ((1 << log_swizzle) - 1));
    ```
    
    **ncu 性能数据**（M=32768, K=2560, N=2048, A100-80GB）：
    
    | Metric | 无 Swizzle | **有 Swizzle** | 改善幅度 |
    | --- | --- | --- | --- |
    | **DRAM Read Throughput** | 2.69 GB | **341 MB** | **7.9× 减少** |
    | **L2 Cache Hit Rate** | 49.8% | **90.3%** | **+81%** |
    | **Tensor Core Utilization** | 36.3% | **51.1%** | **+41%** |
    | **Long Scoreboard Stall** | 50.7% | **23.4%** | **-54%** |
    | **Kernel Latency** | 1.85 ms | **0.98 ms** | **1.9× 加速** |
    
    ---
    
    ### 3. Multi-Config Tile Shape Tuning — M 维度自适应
    
    **问题**：单一 Tile Shape 无法在所有 M 范围内都达到最优：
    
    - 小 M（≤512）：大 tile 导致 CTA 数不足，wave quantization 浪费严重
    - 大 M（>4096）：小 tile 导致 CTA 数过多，scheduling 开销大
    
    **方案**：设计三套互补的 Tile Config，运行时按 M 大小 dispatch：
    
    | Config | TileShape | WarpShape | Threads/CTA | Output/CTA | 适用 M 范围 |
    | --- | --- | --- | --- | --- | --- |
    | **small** | 64×64×128 | 32×32×128 | 128 | 4K | M ≤ 512 |
    | **default** | 128×64×128 | 64×32×128 | 128 | 8K | 512 < M ≤ 4096 |
    | **large** | 128×128×64 | 64×32×64 | 256 | 16K | M > 4096 |
    
    **关键约束**：Hybrid kernel 需维护 **双 INT32 Accumulator**（`int32_accum` + `int32_weighted`），寄存器消耗 = `2 × FragmentC::kElements`。WarpShape 选择必须保证 `Dual accum regs ≤ 255`（SM80 上限）。
    
    **收益**：相比单一 config，在 M=256 场景下 latency 降低 **15%**，M=32768 场景下 Tensor Core 利用率提升 **12%**。
    
    ---
    
    ### 4. Quant Block Size 调优 — Dequant 开销摊销
    
    **问题**：`quant_block_size`（bq）决定每个 quant block 内有多少次 K 维度 MMA 迭代，直接影响 dequant 指令的摊销效率。
    
    **方案**：将 bq 从 128 增大到 256/512，使 `k_tiles_per_qb = bq / kK` 增大，dequant 指令被更多 IMMA 迭代摊薄：
    
    | quant_block_size | k_tiles_per_qb (default config) | Dequant 开销占比 |
    | --- | --- | --- |
    | bq128 | 1 | 高 |
    | bq256 | 2 | 中 |
    | **bq512** | **4** | **低** |
    
    **关键设计**：`quant_block_size` 作为纯运行时参数传入 kernel，通过 `params.k_tiles_per_qb` 在 mainloop 中使用，**无需 recompile kernel** 即可切换不同 bq。
    
    **精度验证**：bq512 vs bq256 在 Cosine Similarity 上差异 < 0.01%，可忽略。
    
    ---
    
    ### 5. 性能总结
    
    **端到端 Latency**（A100-80GB, M=8192, K=2560, N=2048）：
    
    | Kernel | Latency | TFLOPS | vs cuBLAS BF16 |
    | --- | --- | --- | --- |
    | cuBLAS BF16 | 0.410 ms | 209 | 1.0× |
    | Hybrid bq256 (128×64 + swizzle) | **0.247 ms** | **348** | **1.66×** |
    | Hybrid bq512 (128×64 + swizzle) | **0.246 ms** | **349** | **1.67×** |
    | Hybrid_large bq512 (128×128 + swizzle) | **0.238 ms** | **361** | **1.72×** |
    
    **ncu 关键指标**（M=32768）：
    
    | Metric | Hybrid Kernel | 占比/说明 |
    | --- | --- | --- |
    | **Effective TFLOPS** | 348 | 达到 A100 INT8 理论峰值 624 TFLOPS 的 **56%** |
    | **Tensor Core Utilization** | 51.1% | IMMA pipeline 高效利用 |
    | **L2 Cache Hit Rate** | 90.3% | Swizzle 带来的 cache locality 提升 |
    | **DRAM Read Efficiency** | 341 MB | 接近理论最小值（A+B ≈ 26 MB）的 **13×** |
    | **Pipeline Overlap** | INT/FPU/TC | INT dequant 与 Tensor Core IMMA 完美并行 |
    
    ---
    
    ### 技术栈
    
    - **CUDA C++ / CUTLASS 模板元编程**：TileShape/WarpShape/kStages 编译期配置
    - **SM80 Tensor Core INT8 IMMA**：`mma.sync.aligned.m16n8k32.s32.s8.s8.s32`
    - **Nsight Compute (ncu) Profiling**：Pipeline stall 分析、L2/L1 cache hit rate、functional unit utilization
    - **Python/PyTorch Binding**：pybind11 实现 kernel 封装，支持运行时 config dispatch
    
    ```python
    高性能 INT8 量化推理算子开发与深度优化
    项目背景
    
    大语言模型 (LLM) 推理引擎 vLLM 中，W8A8 Block-wise INT8 量化 GEMM 是核心热点算子。该项目旨在开发一套与 CUTLASS 精度完全一致、性能逼近的 Triton kernel 实现，覆盖 Decode（M=1~64）到 Prefill（M=1024~2048）全场景。
    
    技术挑战
    
    原始 Triton 实现采用三级累加器（int32→fp32→fp32），K-loop 内存在大量 I2F/FMUL 浮点运算，导致 Tensor Core 利用率仅 ~41%，比 CUTLASS 慢约 30%
    
    Two-level hybrid 量化方案（Q int32 + F fp32 的二级 scale 分解）在 Triton 中的实现与 CUTLASS 存在语义不一致，导致精度偏差
    
    A100 SM80 架构下寄存器压力达到 255 上限，寄存器 spill 严重（STL/LDL 各 13 条）
    
    核心工作与成果
    
    算法重构：Two-INT32 Accumulator 架构对齐
    
    深度逆向分析 CUTLASS hybrid kernel 的 SASS 反汇编，揭示其 K-loop 使用纯 INT32 双累加器（int32_acc + int32_weighted）的算法策略
    
    将 Triton kernel 从三级 fp32 累加器重构为二级 INT32 累加器，K-loop 内完全消除浮点运算（I2F/FMUL/FADD），Epilogue 仅需一次 I2F + 一次 FMUL
    
    实现与 CUTLASS bit-exact 一致的数值精度（CosSim=1.0），SASS 层面 FMUL 指令从 192 条降至 65 条（与 CUTLASS 完全一致）
    
    寄存器压力优化：消除 Spill
    
    通过消除 fp32 累加器、减少 PTX 虚拟寄存器（b32: 2458→1855, -24%；b64: 590→278, -53%）
    
    寄存器 spill 从 STL/LDL 各 13 条完全降至 0，物理寄存器从 255 降至 252
    
    PTX 代码行数从 2319 精简至 1778（-23%），消除 K-loop 内全部 div/rem 指令
    
    端到端性能提升
    
    Large prefill (M=1024, K=11008) 场景：与 CUTLASS 差距从 ~30% 缩小至 仅 6%
    
    Decode 场景 (M≤64)：差距从 ~45% 缩小至 ~25%
    
    全场景 Tensor Core 利用率从 41.6% 提升至 54.4%（+12.8pp）
    
    微架构级瓶颈分析与 Triton Compiler 限制定位
    
    使用 NSight Compute + nvdisasm 完成 SASS 指令级逆向对比分析，定量归因剩余性能差距：
    
    Short Scoreboard stall (+0.45 cyc/inst)：通过 SASS 反汇编证实 Triton compiler 将 24 条 LDSM 块状发射（vs CUTLASS 的 LDSM↔IMMA 1:1 交错软件流水线），导致 SMEM read port 拥塞
    
    LDGSTS bank conflict (47,737 vs 0)：Triton 自动生成的 SMEM swizzle pattern 不如 CUTLASS 手写 Swizzle<3,3,3> 精确
    
    非 Tensor 开销 +63K active cycles：100% 差距来自 non-Tensor overhead，DRAM 流量 Triton 反而更优 (-12.3%)
    
    明确界定了 Triton compiler backend 的固有限制边界（instruction scheduling、SMEM layout control、register allocation），为后续 compiler 团队改进提供了精确的优化方向
    ```
    
- NGram Penalty-Aware；
    
    ```python
    ### **Penalty-Aware Speculative Decoding 优化**
    
    **vLLM 推理引擎 | 核心开发者**
    
    **背景问题：**
    
    在 Speculative Decoding 场景中，同一请求的多行 draft tokens 并行生成 logits 并应用采样惩罚。现有实现仅基于历史 output tokens 计算 penalty，导致 draft 位置无法感知同批次中更早位置的 token，造成 frequency/presence penalty 计算错误。
    
    **技术方案：**
    
    - **增量计数机制设计**：设计并实现了一种轻量级增量计数方案，通过 `expanded_local_pos` 索引同一请求内各 draft 位置，在 Triton kernel 中动态累加前置 draft token 的出现次数
    - **编译期循环展开**：利用 `tl.static_range` 实现编译期完全展开（MAX_SPEC_LEN 通常 ≤ 5），避免 Triton 动态循环限制，生成 flat code 提升性能
    - **向量化 Token 匹配**：以 BLOCK_SIZE 为粒度向量化比较 vocab ID，单条指令同时判断 8192 个候选 token 是否与前置 draft token 匹配，显著减少循环开销
    - **零额外显存开销**：复用现有 `output_bin_counts` 结构，仅需少量寄存器存储 `draft_counts` 中间结果
    
    **成果：**
    
    - 修正 Speculative Decoding 场景下的采样惩罚计算错误，确保语义正确性
    - 性能开销可控（MAX_SPEC_LEN 很小，编译期展开后仅生成数条 load/compare 指令）
    - 代码已合入 vLLM 主干，影响 `vllm/v1/worker/gpu/sample/penalties.py`
    
    **技术栈：** Python, Triton GPU Kernel, PyTorch, CUDA 编程
    
    ---
    
    **一句话版本（适合 LinkedIn）：**
    
    > 在 vLLM 中实现了 Penalty-Aware Speculative Decoding，设计增量计数机制与编译期循环展开优化，修正了多 draft token 并行采样时的 frequency/presence penalty 计算错误。
    >
    ```
    
- 3D 模型 算子水平融合，Multi-Step CUDA Graph；
- 分类，解析等传统 encoder 模型， tritonserver overlap；
- Fireredasr-vLLM；
- Async NGram GPU；
    
    ```python
    PR #29184 改动总结
    背景
    vLLM v1 引入了异步调度器（Async Scheduler），通过重叠 CPU 调度和 GPU 计算来提升吞吐。PR #24799 率先实现了 EAGLE 投机解码与异步调度的兼容，但 N-gram 投机解码仍依赖 CPU 端的 sample_token_ids，无法与异步调度配合使用——每次采样都需要 CPU-GPU 同步，成为性能瓶颈。
    
    PR #29184 的目标是：将 N-gram 投机解码的全部逻辑搬上 GPU，消除 CPU-GPU 同步，使其与异步调度器兼容。
    
    核心改动
    9 个文件，+940 行 / -12 行，核心变更：
    
    模块	文件	改动
    核心实现	ngram_proposer_gpu.py（新增）	GPU 版 N-gram 提议器的完整实现
    Runner 集成	gpu_model_runner.py	集成 ngram_gpu 调用链
    输入批处理	gpu_input_batch.py	Pinned CPU tensor 优化 H2D 传输
    配置	speculative.py / vllm.py	增加 ngram_gpu 方法支持
    编译兼容	backends.py	ngram_gpu 时禁用 torch.compile 缓存
    测试	2 个测试文件	GSM8K 正确性测试 + 多配置 E2E 测试
    1. NgramGPUKernel — 全向量化 GPU N-gram 匹配内核
    使用 @support_torch_compile() 装饰，经 torch.compile + torch.inductor 自动优化
    核心算法：_find_first_and_extract_all_n_parallel
    用 Tensor.unfold() 构建 O(1) 滑动窗口视图，在 token 序列中并行搜索后缀 n-gram 的最早匹配
    同时搜索 [min_n, max_n] 范围内所有 n-gram 长度，选择最长有效匹配
    全程纯 tensor 操作，无 data-dependent branching，对 torch.compile 友好
    用 cumsum + 位置比较计算 leading contiguous valid token 数量，避免逐元素判断
    2. NgramProposerGPU — 异步调度集成管理器
    Token 散布（Scatter）：将验证通过的 sampled token 通过 scatter_ 原地写入 token_ids_gpu，纯 GPU 操作无同步
    增量 Tensor 更新：update_ngram_gpu_tensors_incremental 处理 batch reorder（请求抢占/恢复后索引变化），通过 clone + index copy 而非全量拷贝
    异步 D2H 拷贝：copy_num_valid_draft_tokens 在独立 CUDA stream 上异步将 valid draft count 拷贝回 CPU，用 Event 同步，避免阻塞主计算流
    无效 Draft 裁剪：update_scheduler_for_invalid_drafts 在 CPU 端根据 D2H 同步结果修剪 scheduler 的 spec decode slot 分配
    Pinned Buffer 复用：预分配 _pinned_idx_buf / _pinned_val_buf，避免每次调用的 allocation + memset 开销
    3. 编译配置与 Warmup 策略
    NgramProposerGPU 使用独立的 CompilationConfig：启用 max_autotune、aggressive_fusion、coordinate_descent_tuning，显式关闭 CUDA Graph（因为 batch size 动态变化）
    _dummy_run() 用最大 batch size 做 3 次 warmup，触发 torch.compile 编译 + 缓存
    在全局 backends.py 中检测 ngram_gpu 并禁用 vLLM 的 torch.compile 缓存，避免与动态 shape kernel 的缓存冲突
    4. 性能结果
    在 NVIDIA H20 上，Qwen3-1.7B + CMU-DoG 数据集：
    
    并发数	Async NGram GPU (tps)	Sync NGram (tps)	加速比
    2	466	357	30.5%
    8	1378	988	39.4%
    16	2082	1726	20.6%
    工作经历
    vLLM N-gram 投机解码 GPU 化与异步调度集成
    角色：推理引擎核心开发工程师
    技术栈：PyTorch / torch.compile / torch.inductor / CUDA Stream / Python
    成果：PR 已合并至 vLLM 主线（PR #29184），在中等并发下实现 20-39% 的吞吐提升
    
    项目背景
    vLLM 的异步调度器（Async Scheduler）通过重叠 CPU 调度与 GPU 计算提升推理吞吐，但原有的 N-gram 投机解码依赖 CPU 端逐 token 采样与匹配，每步推理需要 CPU-GPU 同步，成为异步流水线的阻塞点。本项目将 N-gram 提议器完全迁移至 GPU 端，实现全链路无同步的异步投机解码。
    
    核心工作
    1. 设计并实现全向量化 GPU N-gram 匹配内核
    
    利用 Tensor.unfold() 构建 O(1) 滑动窗口视图，在序列中并行搜索后缀 n-gram 匹配，同时覆盖 [min_n, max_n] 范围内所有 n-gram 长度，优先选择最长有效匹配。
    全程使用纯 tensor 运算，消除 data-dependent branching 和 Python 循环，使得整个 kernel 对 torch.compile 完全友好。
    使用 @support_torch_compile() + torch.inductor 编译，配合 max_autotune、aggressive_fusion、coordinate_descent_tuning 等高级优化选项，自动生成高效 CUDA 代码。
    2. 实现异步调度器下的无同步 Token 管理
    
    Token 原地散布（Scatter）：将验证通过的 sampled tokens 通过 scatter_ 原地写入 GPU 端 token_ids buffer，避免 CPU-GPU 数据拷贝。
    增量 Tensor 更新：设计 update_ngram_gpu_tensors_incremental，仅在 batch reorder（请求抢占/恢复后索引变化）时做局部 index copy，避免全量拷贝开销。
    异步 D2H 同步：在独立 CUDA Stream 上异步将 valid draft count 拷贝回 CPU，用 torch.cuda.Event 做精确同步点，避免阻塞主计算流。
    Pinned Memory 复用：预分配 pinned CPU buffer 用于 H2D/D2H 数据传输的 staging，消除每次调用的 allocation 开销。
    3. 适配 vLLM v1 调度与编译框架
    
    修改 SpeculativeConfig 和 VllmConfig，新增 ngram_gpu 方法类型，解除异步调度对该方法的限制。
    处理与 torch.compile 缓存的兼容性问题：由于 N-gram kernel 的 batch size 动态变化会导致编译缓存冲突，特别在编译后端中检测 ngram_gpu 并禁用 vLLM 侧的缓存机制。
    实现 _dummy_run warmup 策略，以最大 batch size 预热 torch.compile，确保推理时无 JIT 编译延迟。
    4. 端到端测试与性能验证
    
    编写多维度 E2E 测试：覆盖不同 num_speculative_tokens、prompt_lookup_min/max、执行器类型（mp/uni）、异步/同步调度、chunked prefill 等配置组合。
    通过 GSM8K 数学推理准确率测试验证正确性（Qwen3-8B，准确率阈值 80%）。
    在 NVIDIA H20 上实测 Qwen3-1.7B，相比同步 N-gram 实现取得 20.6%–39.4% 的端到端吞吐提升。
    ```
    
- DivPrune 优化；
    - **在 vLLM 多模态推理引擎中设计并实现了 DivPrune 视觉 Token 剪枝方法，应用于 Qwen3-VL 系列模型。** 针对现有 EVS（Efficient Video Sampling）仅支持基于时序相似性的视频帧间冗余裁剪、无法处理图像 Token 冗余的局限，提出了基于特征空间多样性的最远点采样（Farthest-Point Sampling）算法，通过计算余弦距离矩阵并贪心选取离已选集合最远的 Token，最大化保留子集的特征覆盖度，实现了对图像和视频 Token 的统一剪枝能力。工程实现上，采用批量并行选取策略（parallel_k=32）替代逐 Token 贪心，结合 float32 数值稳定性保障和 in-place Tensor 操作（`neg_().add_()`）减少显存分配开销；设计了将 mRoPE 3D 位置编码附加到 Embedding 末尾 4 通道的方案，确保剪枝后位置信息可正确恢复；引入可配置的 micro-batch 机制，解决了 Scheduler 按剪枝后 Token 数估算显存导致视觉编码器 OOM 的问题。整体改动覆盖命令行参数、配置传递、核心算法、模型后处理到运行时调度的全链路，通过 `--vision-pruning-method` 参数实现 EVS/DivPrune/关闭三种模式的无缝切换，保持完全向后兼容。
- Whisper-small LogitProcessor 定制化（语言识别）
- Eagle3 Draft Model
    
    ```python
    项目名称： vLLM extract_hidden_states 中间层表征导出链路设计与实现
    项目角色： 推理引擎 / 系统开发
    项目简介： 基于 vLLM V1 推理框架，构建面向中间层表征提取的 extract_hidden_states 链路，复用 speculative decoding 与 KV transfer 基础设施，将目标模型多层 aux_hidden_states 直接写入 GPU KV Cache，并通过 connector 导出给外部系统使用。
    
    负责设计并打通 GPUModelRunner -> ExtractHiddenStatesProposer -> ExtractHiddenStatesModel -> CacheOnlyAttentionLayer -> KV Connector 的完整执行链路，使目标模型在 rollout / 推理阶段能够额外输出并持久化多层 hidden states。
    基于 speculative decoding 框架做功能复用，将 extract_hidden_states 设计为“伪 spec decode”模式：不真正预测 draft token，而是直接复用目标模型采样结果，仅借助 spec decode 的调度、slot mapping、batch padding 与 KV cache 管线完成 hidden states 缓存。
    实现 cache-only drafter 模型，构建仅包含单个 CacheOnlyAttentionLayer 的 ExtractHiddenStatesModel，通过 num_heads = num_hidden_states、head_size = hidden_size 的映射方式，将多层 hidden states 直接按页写入 GPU paged KV cache，避免额外重排与无效计算。
    打通主模型 aux hidden states 输出链路，复用 EAGLE3 风格接口，从目标模型 forward 结果中接收 (hidden_states, aux_hidden_states)，并按 eagle_aux_hidden_state_layer_ids 选择多层中间表示参与缓存与导出。
    实现 GPU 侧高效写入机制，在 set_forward_context 中结合 slot_mapping、attention metadata 与 unified_kv_cache_update，按 token 级位置将 aux_hidden_states 精确 scatter 到 drafter KV cache，对外暴露为可 transfer 的中间状态缓冲区。
    集成运行时工程能力，支持 load_model、dummy_run、CUDA Graph warmup、数据并行 batch 对齐、KV cache 绑定与 connector 回调，确保该特性能够无缝接入 vLLM 原有推理执行框架。
    扩展 hidden-state 导出能力，结合 ExampleHiddenStatesConnector 将缓存后的 hidden states 从 KV cache 中抽取并落盘，形成面向外部推理基础设施/特征消费系统的通用中间表征导出方案。
    更像简历的一段式精简版
    负责 vLLM extract_hidden_states 特性的链路设计与实现，复用 speculative decoding 与 KV transfer 基础设施，打通 GPUModelRunner 到 CacheOnlyAttentionLayer 的执行路径，将目标模型多层 aux_hidden_states 直接写入 GPU paged KV cache，并通过 connector 导出给外部系统使用；其中实现了 cache-only drafter、slot mapping 定位写入、aux hidden states 输出接入、CUDA Graph/dummy run 适配及 KV cache 绑定等关键模块。
    
    如果你想写得更偏“RL Infra / rollout”风格，可以用这个版本
    项目名称： 面向大模型 rollout 的中间层 Hidden States 导出加速链路
    
    基于 vLLM 推理引擎实现 extract_hidden_states 特性，在不额外引入完整 draft model 推理开销的前提下，复用 speculative decoding 调度链路，将目标模型多层 aux_hidden_states 缓存到 GPU KV cache。
    设计并实现 cache-only attention drafter，将 hidden states 作为“伪 KV”直接写入 paged cache，并通过 connector 导出，为外部 RL Infra / rollout 系统复用中间表征提供底层能力。
    完成 aux_hidden_states 输出、batch padding、slot mapping、KV cache update、connector save/load hook、CUDA Graph warmup 等模块联调，提升中间状态抽取链路与现有推理框架的集成度。
    
    vLLM 推理引擎中间表征导出能力建设 / 推理基础设施开发
    
    负责设计并落地 vLLM extract_hidden_states 能力，基于 speculative decoding 与 KV transfer 框架，打通 GPUModelRunner、ExtractHiddenStatesProposer、ExtractHiddenStatesModel、CacheOnlyAttentionLayer 及 Connector 的完整执行链路。
    复用现有 speculative decode 调度体系，将该能力设计为“轻量级 cache-only drafter”方案：不额外执行真实 draft token 推理，直接复用 target model 采样结果，在保证线上链路兼容性的同时完成多层 hidden states 缓存。
    实现多层 aux_hidden_states 的 GPU 侧高效写入机制，结合 slot_mapping、attention metadata 与 paged KV cache，将中间层表征按 token 位置直接写入显存缓存，避免重复前向与额外数据重排。
    支持 dummy_run、CUDA Graph、DP 对齐、KV cache 绑定与 connector 回调等运行时能力，提升该特性在 vLLM 推理框架中的可集成性与工程稳定性。
    基于 ExampleHiddenStatesConnector 完成 hidden states 导出链路验证，实现从 GPU KV cache 到外部存储/下游系统的中间表征抽取能力，为外部推理基础设施、特征复用及后续 rollout 场景预留扩展接口。
    ```