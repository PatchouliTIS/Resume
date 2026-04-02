### 陈选安

::: left
icon:phone 17826507955
:::

::: right
icon:weixin deoppressoliber
:::

::: left
icon:email [623894679@qq.com](mailto:pyramkar@qq.com)
:::

::: right
icon:github github.com/PatchouliTIS
:::

## 教育背景

:::left
**中国科学技术大学 - 软件工程（硕士）**
:::

:::right
**2022.09 - 2025.06**
:::

GPA  3.4/4.3  (TOP 25%)    研究生二等奖学金

*<u>主修课程</u>*：软件工程、Python程序设计、分布式系统与云计算、智能计算、深度学习、机器学习

:::left
**长安大学 - 道路桥梁与渡河工程**
:::
:::right
**2015.09 - 2019.06**（本科）   **2019.09 - 2020.10**（硕士肄业） 
:::
GPA 3.7/5.0  (TOP 10%)  院级一等奖学金
*<u>主修课程</u>*：C++程序设计、Python程序设计、数据结构

## 实习经历

:::left
**微软**  Bing Core Search Intern
:::

:::right
**2023.10 - 2024.12**

:::

工作描述：

- **KV Cache 压缩实现**： 基于 <u>注意力分数</u>的 KV Cache 压缩机制，从<u>稀疏计算</u>和<u>KV Block整合</u>两个方向对 Mistral-7B-Instruction 模型实现推理优化：使用 Triton 框架实现了 Prefill 阶段 Block-Sparse Attention 注意力点积计算，通过<u>分块并行运算</u>以及<u>算子融合</u>等方式加速张量运算；通过编写CUDA内存搬运函数，与vLLM框架中CacheEngine结合，实现了Prefill阶段Sparse KV Cache的动态压缩存储，<u>与全量KV缓存的方案相比达到了60%压缩率</u>；<u>相比 Dense Attention 的情况，在20k~30k tokens prompts 推理任务中达到了4x kernel 加速比</u>，业务场景下在A100上推理任务的P95时延从9.8s优化到6.3s，在 Infinite Benchmark，PPL，RULE 等基准测试中达到全量模型 99.8% 的性能，同时在一些特定任务上性能超过全量模型 20% ；

- **Online TopK 算子融合实现**：使用 Cutlass 与 CuTe 框架，针对<u>在线稀疏注意力</u>的场景，运用 Tensor Core ，异步IO，计算重叠流水线，并行 Bitonic TopK 等技术实现了<u>张量点积运算算子和 Online TopK算子的融合</u>，相比原有的 PyTorch 中 Matmul 与 Topk 算子独立运行的情况，将原始的140ms算子运行时延压缩到11ms，达到了 12x ~ 15x kernel 加速比；

- **Labeling 模型量化部署**：根据在线与离线推理情境下 batch size 不同的场景采用不同的量化方案实现模型的推理加速：采用SmoothQuant针对大batch size的compute-bound场景进行优化，采用GPTQ优化小batch size的memory-bound场景；<u>将Labeling模型服务的P95延迟从6.7秒降至4.4秒</u>，提升系统响应速度和用户体验；

- **搜索排序模型优化**：在基于 BERT 架构的排序模型的知识蒸馏训练过程中，<u>引入 RoPE 来强化模型对长序列上下文的理解能力，将最大输入长度提升到 386 tokens</u>；添加额外的多任务MLP头与交叉熵损失函数来实现多任务推理；<u>使用DeepSpeed ZeRO-2 策略和FP16混合精度方法在8xA100 集群上进行两阶段蒸馏，启动DeepSpeed中的 FusedTransformerKernel 替换原始模型中的forward方法，达到基线蒸馏的 8x 训练速度</u>；

- **必应跨语言搜索缓存优化**：根据搜索日志构建跨语言搜索缓存，将TOP5搜索结果向量化后存储于Faiss向量数据库中，用<u>当前用户Query与离线搜索日志做ANN搜索匹配，提前返回搜索结果，将响应时间缩短至原方案的80%</u>，显著提升了系统性能；使用 Azure DevOps 构建完整的 CI/CD 工作流，部署至线上服务； 

:::left 

**腾讯**   CSIG 软件开发实习生
:::

:::right

**2023.06 - 2023.09**

:::

工作描述：

- **构建数据传输服务**：基于MySQL主从复制原理，使用Canal增量订阅组件<u>从0到1完成MySQL-Canal-Kafka数据传输服务</u>的构建工作，替换掉之前基于JDBC驱动的Kafka Connect数据同步方案，<u>使服务QPS提升至原来的176%</u>；
## 项目经历

:::left

**基于VGG19的神经网络训练以及寒武纪 MLU 算子实现**
:::
:::right

**2023.04 - 2023.05**

:::

- **工作内容**
  
  - 使用Tensorflow框架，基于VGG19神经网络结构搭建图像转换网络以及特征提取网络进行训练，在此基础上使用最小化损失函数对网络训练进行优化，实现了图像的实时风格迁移；
  
  - 实现多个寒武纪 MLU 的算子， 使用 MLU 提供的 SIMD 函数和内存搬运函数， 实现多个算子例如SoftMax，矩阵乘等；在此基础上在MLU实现双流水线并行，将自定义算子融合进TensorFlow计算图中，加速比达到2.4x。

:::left

**基于RAG的大语言模型私人知识库机器人**

:::

:::right

**2023.12 - 2024.03**

:::

- **项目描述**：将Wikipedia上关于计算机科学相关的文本经过分片向量化后存储到本地Postgres向量数据库，对大语言模型应用量化，局部参数微调等技术构建大语言模型问答机器人。

- **工作内容：**
  
  - 使用AutoGPTQ对Llama-2-7B-hf模型进行weight only 4bit量化，使用vllm框架进行推理；
  - 使用额外的闭源模型ChatGPT-4o对用户Query进行改写，提炼核心问题；
  - 使用Sentence-Transformer微调BGE模型，构造Query-Chunk-Tag数据对，增强BGE模型上下文感知能力和文本理解能力，相比原本的嵌入模型其IR质量提高了10%；
  - 使用基于随机森林算法的多分类算法根据用户Query所属的tag对文本结果进行重排序，在原有模块的基础上提高了24%的回答质量。


## 专业技能

- **编程语言基础**：熟悉 C++ 面向对象编程与现代C++语言特性；熟悉Linux系统开发；熟悉Python与Golang；

- **AI相关**：熟悉框架Pytorch，Tensorflow；熟悉大模型推理框架vLLM，TensorRT-LLM，掌握深度学习大规模分布式系统中，张量并行，模型并行，序列并行等概念；拥有CUDA/Triton算子开发经验，掌握常用的神经网络推理优化方法。

- **开发组件**：熟悉消息队列Kafka的部署和使用，掌握分布式搜索引擎ElasticSearch的使用，熟悉Docker容器的原理与使用，掌握CMake的使用，了解容器编排工具Kubernetes的基本概念；



