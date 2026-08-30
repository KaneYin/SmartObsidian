# Weft 项目真实简历亮点提取与技术文档

> **项目名称**: Weft (Local-First Agent & Knowledge Graph RAG System over Obsidian Vault)  
> **验证状态**: 288 项 pytest 单元测试 100% 通过（离线测试驱动），所有技术亮点均有真实源码与单元测试支撑。

---

## 1. 项目简介 (Project Overview)

**Weft** 是一款面向 Obsidian Markdown 笔记库的本地优先 (Local-First)、高隐私、图感知增强型 RAG (Retrieval-Augmented Generation) 知识库与 Agent 记忆系统。项目融合了向量检索、手写 BM25 词法检索、$N$ 路倒数排名融合 (RRF)、Cross-Encoder 重排序、双向 Wikilink 图扩充召回、持久化 Agent 记忆推演以及零信任本地安全沙箱机制，支持在纯离线本地开源模型 (Ollama) 与云端 API (Claude / OpenAI) 之间自适应切换。

---

## 2. 简历亮点分类提炼 (Resume Highlights)

在简历中，可根据所投递的目标岗位（如 **RAG / AI Agent 系统专家**、**LLM 应用基础设施工程师**、**AI 搜索/检索增强工程师**），选择以下维度进行针对性呈现：

### 维度一：多路召回与自适应融合检索架构 (Hybrid Search & Multi-Query RRF)

**简历描述示例**：
- **设计并实现了 $N$ 路 Reciprocal Rank Fusion (RRF) 混合检索与重排管道**：手写 Okapi BM25 词法检索与 Dense Vector 向量检索引擎，结合基于多轮对话上下文改写的 Context Query 及 Agent 内存 Query 进行三路并行召回，解决单一检索范式的词法/语义匹配盲区。
- **引入两阶段检索范式 (Two-Stage Retrieval)**：一阶段基于 RRF 召回候选池 ($k=20\sim 30$)，二阶段集成 Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) 进行语义重排序。设计 `fast` / `balanced` / `best` 意图驱动的三挡检索模式，平衡低延迟与高精度的系统诉求。

**源码与测试支撑**：
- 核心融合算法：[src/weft/fusion.py:L10-L21](file:///Users/kane/Dev/AgentDevelopment/src/weft/fusion.py#L10-L21) (`reciprocal_rank_fusion` 公式: $score = \sum \frac{1}{k + rank + 1}$)
- 手写 BM25 索引：[src/weft/bm25.py:L23-L60](file:///Users/kane/Dev/AgentDevelopment/src/weft/bm25.py#L23-L60) ($k_1=1.5, b=0.75$)
- Cross-Encoder 重排器：[src/weft/rerank.py:L21-L36](file:///Users/kane/Dev/AgentDevelopment/src/weft/rerank.py#L21-L36)
- 检索配置与意图分级：[src/weft/retrieval_config.py:L47-L102](file:///Users/kane/Dev/AgentDevelopment/src/weft/retrieval_config.py#L47-L102)
- 对应单测：`tests/test_fusion.py`, `tests/test_bm25.py`, `tests/test_rerank.py`, `tests/test_agent_hybrid.py`, `tests/test_agent_rerank.py`

---

### 维度二：知识图谱拓扑增强与分块策略 (Graph-Aware & Advanced Chunking)

**简历描述示例**：
- **开发了 Obsidian Wikilink 双向图拓扑扩展召回机制**：自动解析笔记间的 `[[wikilink]]` 链接生成图邻接表，在向量/BM25 召回 Top-K 节点后，通过 1-hop 邻居图遍历拉取关联笔记，提升关联知识点的全景召回率。
- **构建 Parent-Child 与 Sliding Contextual 分块增强算法**：实现段落级 Child 节点比对与父级 Heading / Section 节点完整上下文回溯的 Parent-Child Chunking；设计 Sliding Window (1000 字符，200 字符重叠) 与 LLM 预生成 Context Sentence 注入 `embed_text` 的 Contextual Chunking 策略，显著提升嵌入向量的表达质量。

**源码与测试支撑**：
- Wikilink 图解析与拓扑检索：[src/weft/graph.py:L17-L50](file:///Users/kane/Dev/AgentDevelopment/src/weft/graph.py#L17-L50)
- Parent-Child 与 Sliding/Contextual Chunking：[src/weft/chunking.py:L55-L101](file:///Users/kane/Dev/AgentDevelopment/src/weft/chunking.py#L55-L101)
- 对应单测：`tests/test_graph.py`, `tests/test_agent_graph.py`, `tests/test_chunking.py`, `tests/test_agent_parent.py`

---

### 维度三：持久化 Agent 记忆与推演机制 (Durable Agent Memory & Inference)

**简历描述示例**：
- **架构了双轨 Agent 记忆系统**：支持 `preference` / `fact` / `decision` / `task` 四类持久化语义记忆与 `episodes.jsonl` 情景日志。采用 Append-only + Latest-record-wins 策略与 Tombstone 压缩机制，保证内存更新的原子性与透明审计。
- **实现反思推演与只读镜像同步 (Memory Suggest & Mirror)**：设计基于 LLM 的情景日志反思提炼引擎，自动挖掘未显式指定的规则与事实，生成提案供用户 Accept/Reject 审核，并实时同步更新 Vault 内部的只读 `_memory.md` 镜像文件。

**源码与测试支撑**：
- 记忆存储与压缩机制：[src/weft/memory.py:L70-L138](file:///Users/kane/Dev/AgentDevelopment/src/weft/memory.py#L70-L138)
- 记忆召回与嵌入比对：[src/weft/memory.py:L158-L176](file:///Users/kane/Dev/AgentDevelopment/src/weft/memory.py#L158-L176)
- 记忆反思推演：[src/weft/memory_infer.py](file:///Users/kane/Dev/AgentDevelopment/src/weft/memory_infer.py)
- 只读 Mirror 同步：[src/weft/memory_mirror.py](file:///Users/kane/Dev/AgentDevelopment/src/weft/memory_mirror.py)
- 对应单测：`tests/test_memory.py`, `tests/test_memory_infer.py`, `tests/test_memory_mirror.py`, `tests/test_agent_memory.py`

---

### 维度四：零信任安全与本地沙箱防御 (Zero-Trust Security & Path Confinement)

**简历描述示例**：
- **设计零信任本地安全沙箱**：防护符号链接攻击 (Symlink Path Traversal Attack)，在读取与写入路径中递归校验符号链接，严禁跨出 Vault 规范根目录。
- **实现高密数据防护与脱敏管道**：构建带 Glob 包含/排除与正则脱敏 (`--redact`) 的隐私引擎；所有向量索引、API Log (`.weft/api-log.jsonl`) 及配置文件写操作均强制通过临时文件原子落盘，并将文件权限收紧至 `0600`、目录权限收紧至 `0700`；内置 ANSI 终端控制字符清洗机制，防止控制代码注入。

**源码与测试支撑**：
- 符号链接防御与原子写：[src/weft/security.py:L43-L184](file:///Users/kane/Dev/AgentDevelopment/src/weft/security.py#L43-L184)
- 隐私过滤与正则脱敏：[src/weft/privacy.py:L31-L68](file:///Users/kane/Dev/AgentDevelopment/src/weft/privacy.py#L31-L68)
- 安全加固文档：[docs/security/2026-08-08-security-hardening.md](file:///Users/kane/Dev/AgentDevelopment/docs/security/2026-08-08-security-hardening.md)
- 对应单测：`tests/test_security.py`, `tests/test_privacy.py`

---

### 维度五：硬件感知与异构 LLM Provider 架构 (Hardware-Aware Provider & Fallback)

**简历描述示例**：
- **开发了硬件自适应模型调度器与 Provider 降级机制**：自动探测 macOS (Unified Memory) 及 Linux (Nvidia CUDA VRAM) 显存开销，智能分配模型量化阶梯（8GB: 3B / 16GB: 7B / 24GB+: 14B）。
- **解耦式 Provider 统一抽象**：统一 Ollama (纯离线)、Anthropic Claude API 及 OpenAI API 的调用协议，实现层级化配置解析（CLI > Env > Config File > Hardware Default）与无缝 Fallback 降级机制。

**源码与测试支撑**：
- 显存/硬件探测：[src/weft/hardware.py:L31-L53](file:///Users/kane/Dev/AgentDevelopment/src/weft/hardware.py#L31-L53)
- Provider 注册与 Fallback：[src/weft/providers.py:L50-L164](file:///Users/kane/Dev/AgentDevelopment/src/weft/providers.py#L50-L164)
- 对应单测：`tests/test_hardware.py`, `tests/test_providers.py`, `tests/test_ollama_client.py`, `tests/test_openai_client.py`

---

### 维度六：基准测试与自动化工程 (CRAG Benchmark & Test Automation)

**简历描述示例**：
- **实现 CRAG (Comprehensive RAG) Task 1 自动化基准测试适配器**：对接 CRAG 网页抓取数据集，实现无数据污染的临时向量索引构建与 Ollama 本地模型答案生成，并通过轻量级 Runner 实现自动化离线 RAG 评估。
- **全系统 100% 离线单测覆盖**：编写 288 项单元测试，涵盖模拟 LLM (FakeLLM)、模拟重排器 (FakeReranker) 及端到端 CLI / REST API / TUI 测试，测试套件运行时间 < 8 秒。

**源码与测试支撑**：
- CRAG 评测适配器：[src/weft/benchmarks/crag.py](file:///Users/kane/Dev/AgentDevelopment/src/weft/benchmarks/crag.py), [src/weft/benchmarks/crag_runner.py](file:///Users/kane/Dev/AgentDevelopment/src/weft/benchmarks/crag_runner.py)
- 对应单测：`tests/test_crag_benchmark.py`, `tests/test_crag_runner.py`

---

## 3. 简历项目描述精炼版 (Resume Bullet Points for Quick Copy)

```markdown
项目名称：Weft - 本地图感知 RAG 与 Agent 知识库系统 (Python / LangGraph / SentenceTransformers)
项目描述：基于 Obsidian Markdown 笔记库的本地优先、高隐私、图感知与持久化 Agent 记忆系统。
主要贡献：
1. 检索架构：设计并手写 $N$ 路 Reciprocal Rank Fusion (RRF) 算法，融合密集向量 (Vector)、手写 BM25 词法索引、上下文改写 Query 与 Agent 记忆 Query 三路召回，并结合 Cross-Encoder 实现两阶段重排 (Reranking)，提供 fast/balanced/best 意图驱动模式。
2. 拓扑与分块：基于 Wikilink 双向链接构建加权知识图谱，实现 1-hop 图扩展召回；开发 Parent-Child 段落分块与 Sliding Contextual 块级上下文注入算法，大幅提升语义召回精度。
3. 记忆系统：架构 preference/fact/decision/task 四类语义记忆与情景日志 (Episodic Log)，实现追加写 collapse、Tombstone 压缩与 LLM 自动化记忆反思推演。
4. 安全沙箱：设计零信任防护机制，递归校验路径排除 Symlink 目录穿透攻击；实现包含/排除 Glob 过滤与正则脱敏；审计日志与索引文件强制采取 0600/0700 权限与原子落盘。
5. 硬件与基准：开发基于 macOS/CUDA 显存探测的硬件自适应模型挑选器与多 Provider 自动 Fallback 机制；集成 CRAG Task 1 跑分适配器。全系统通过 288 项自动化单元测试。
```
