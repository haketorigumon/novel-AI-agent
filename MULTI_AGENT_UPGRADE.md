# 多智能体系统升级说明

本文档描述了Novel AI Agent从专注于小说生成到通用多智能体系统的升级过程和架构变化。

## 升级概述

Novel AI Agent已经从一个专注于小说生成的系统升级为一个通用的多智能体系统，可以处理各种任务。这次升级借鉴了[OpenManus](https://github.com/FoundationAgents/OpenManus)、[OpenGPTs](https://github.com/langchain-ai/opengpts)和[CAMEL](https://github.com/camel-ai/camel)等先进多智能体框架的设计理念。

## 新增功能

### 1. 多种专业智能体类型

系统现在支持多种专业智能体类型，每种类型都有其特定的能力和专长：

- **任务型智能体 (TaskAgent)**: 专注于完成特定任务，能够分解任务并使用工具
- **助手型智能体 (AssistantAgent)**: 提供通用帮助和回答问题，保持对话历史
- **专家型智能体 (ExpertAgent)**: 在特定领域提供专业知识和咨询
- **创意型智能体 (CreativeAgent)**: 生成各种创意内容，如故事、诗歌、设计等

### 2. 智能体通信系统

- 实现了结构化的消息传递系统，支持不同类型的消息
- 支持一对一通信和广播通信
- 支持对话管理和消息历史记录

### 3. 记忆管理系统

- 为智能体提供长期记忆存储
- 支持不同类型的记忆（情景记忆、语义记忆等）
- 基于相关性的记忆检索机制
- 记忆重要性评分和衰减机制

### 4. 工具使用能力

- 实现了工具注册和执行系统
- 智能体可以根据自身能力使用不同工具
- 内置基础工具（计算器、时间查询等）
- 可扩展的工具注册机制

### 5. 智能体编排系统

- 实现了智能体协作的编排机制
- 支持任务分解和分配
- 支持不同的编排策略（层级式、点对点、混合式）
- 任务状态跟踪和结果汇总

### 6. 交互式界面

- 命令行交互式聊天界面
- 支持多种LLM提供商
- 任务状态和结果可视化（规划中）

## 架构变化

### 新增目录结构

```
src/
├── core/multi_agent/       # 多智能体系统核心
├── agents/types/           # 专业智能体类型
├── communication/          # 智能体通信
├── memory/                 # 记忆管理
├── tools/                  # 工具系统
└── orchestration/          # 智能体编排
```

### 主要组件

1. **EnhancedBaseAgent**: 增强的基础智能体类，提供通信、记忆、工具使用等能力
2. **Message**: 智能体间通信的消息类，支持不同类型的消息和元数据
3. **MemoryManager**: 智能体记忆管理器，提供存储和检索功能
4. **ToolRegistry**: 工具注册表，管理可用工具和执行
5. **Orchestrator**: 智能体编排器，协调多个智能体的协作
6. **MultiAgentSystem**: 多智能体系统的主入口，管理所有组件

## 配置变化

在`config.yaml`中新增了多智能体系统的配置部分：

```yaml
multi_agent:
  enabled: true
  max_agents: 20
  default_agent_types:
    - "assistant"
    - "expert"
    - "creative"
    - "task"
  orchestration_strategy: "hierarchical"
  # 更多配置...
```

## 使用方法

### 启动多智能体系统

```bash
python main.py multi_agent
```

### 交互式聊天

```bash
python main.py agent_chat
```

## 未来发展方向

1. **Web界面**: 开发专门的多智能体系统Web界面，可视化智能体交互和任务执行
2. **更多智能体类型**: 添加更多专业智能体类型，如研究型、分析型、教学型等
3. **高级工具集成**: 集成更多高级工具，如网络搜索、数据分析、代码执行等
4. **记忆优化**: 改进记忆检索算法，使用嵌入和语义搜索
5. **自适应编排**: 实现基于任务特性的自适应编排策略
6. **多模态支持**: 添加图像、音频等多模态内容处理能力

## 参考资源

- [OpenManus](https://github.com/FoundationAgents/OpenManus)
- [OpenGPTs](https://github.com/langchain-ai/opengpts)
- [CAMEL](https://github.com/camel-ai/camel)