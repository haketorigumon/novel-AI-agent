# Novel AI Agent 🤖

Novel AI Agent is an advanced AI agent system designed for generating long-form novels (over 5 million words) with self-evolution and continuous improvement capabilities. It has now been upgraded to a powerful multi-agent system that can collaborate on various complex tasks. The system integrates several core technologies and has been enhanced with features inspired by the Agent Zero framework.

[![Docker Support](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Multi-Agent](https://img.shields.io/badge/Multi--Agent-System-green?style=for-the-badge&logo=robot&logoColor=white)](https://github.com/agent0ai/agent-zero)
[![Vector Memory](https://img.shields.io/badge/Vector-Memory-purple?style=for-the-badge&logo=database&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Modern UI](https://img.shields.io/badge/Modern-UI-orange?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)

## 🌟 Core Technologies

### 1. Dynamic World Story Simulation
Based on [Dynamic-World-Story-using-LLM-Agent-Based-Simulation](https://github.com/JackRipper01/Dynamic-World-Story-using-LLM-Agent-Based-Simulation):
- 🎭 Multiple character agents, each with unique personalities and goals
- 🎬 Director agent that guides story development through environmental changes
- 🌍 Dynamic world simulation with environmental changes and event generation
- 📚 Intelligent narrative synthesis that integrates multiple character contributions into a coherent story

### 2. Darwin-Godel Machine Evolution
Based on [Darwin-Godel-Machine](https://github.com/mmtmn/Darwin-Godel-Machine):
- 🧬 Automatic code evolution based on story generation performance
- 🔄 Continuous self-improvement and optimization
- 📊 Performance evaluation and quality analysis
- 🔒 Safe code backup and rollback mechanisms

### 3. Multi-Agent System
Based on multi-agent collaboration frameworks like [OpenManus](https://github.com/FoundationAgents/OpenManus), [OpenGPTs](https://github.com/langchain-ai/opengpts), and [CAMEL](https://github.com/camel-ai/camel):
- 🧠 Multiple specialist agent types (task, assistant, expert, creative)
- 💬 Structured agent communication protocol
- 🛠️ Tool usage capabilities
- 🗃️ Long-term memory management
- 🎯 Task decomposition and assignment
- 🔄 Flexible orchestration strategies

### 4. Agent Zero Inspired Features
New features inspired by the [Agent Zero](https://github.com/agent0ai/agent-zero) framework:
- 🧠 Vector-based memory retrieval using FAISS
- 🔄 Enhanced multi-agent communication
- 🌐 Modern web interface with real-time updates
- 🐳 Docker support for easy deployment
- 📊 Improved embedding capabilities

## 🚀 功能特性

### 小说生成
- **长篇小说生成**: 目标生成500万字以上的长篇小说
- **角色智能体协作**: 角色、导演、世界模拟器协同工作
- **动态世界**: 实时环境变化和事件生成
- **自我进化**: 基于性能自动改进代码架构

### 通用多智能体系统
- **多种智能体类型**: 任务型、助手型、专家型、创意型智能体
- **动态智能体创建**: 根据任务需求按需创建专业智能体
- **智能体通信**: 智能体之间的结构化消息传递
- **记忆管理**: 智能体长期记忆与相关性检索
- **工具集成**: 智能体可使用工具完成任务
- **智能编排**: 多智能体的层级协调

### 通用功能
- **Web界面**: 实时监控和控制面板
- **交互式聊天**: 与多智能体系统进行对话
- **多种LLM支持**: 支持Ollama、OpenAI、Anthropic、Google等14种LLM提供商

## 📦 Installation

### Prerequisites
- Python 3.8+ (Python 3.9 or higher recommended)
- [Ollama](https://ollama.ai/) (for local LLM, optional)
- Docker (for containerized deployment, optional)

### Quick Installation

#### Option 1: Standard Installation
```bash
# Clone the repository
git clone https://github.com/ineverxxx-max/novel-AI-agent.git
cd novel-AI-agent

# Run the installation script
chmod +x install.sh
./install.sh

# Or install manually
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Option 2: Docker Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/ineverxxx-max/novel-AI-agent.git
cd novel-AI-agent

# Build and run with Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t novel-ai-agent .
docker run -p 12000:80 novel-ai-agent
```

### 配置LLM提供商

系统支持多种LLM提供商，选择其中一种即可：

#### 选项1: Ollama (本地免费)
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 启动服务
ollama serve

# 下载模型
ollama pull llama3
```

#### 选项2: OpenAI
```bash
export OPENAI_API_KEY="your_api_key_here"
```

#### 选项3: Anthropic Claude
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

#### 选项4: Google Gemini
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

#### 选项5: 其他提供商
支持Groq、Together AI、DeepSeek、Moonshot等，详见[提供商配置指南](PROVIDER_GUIDE.md)

**配置方法**:
1. 设置环境变量（推荐）
2. 编辑`config.yaml`文件
3. 使用`.env`文件

## 🎯 使用方法

### 命令行界面

所有命令都通过 [`main.py`](main.py) 文件执行。

#### 小说生成

```bash
# 激活虚拟环境
source venv/bin/activate

# 生成小说（命令行模式）
python main.py generate

# 启动Web界面
python main.py web

# 生成小说并启用Web界面
python main.py generate --web-interface

# 手动触发代码演进
python main.py evolve --generations 3
```

#### 多智能体系统

```bash
# 启动多智能体系统
python main.py multi_agent

# 使用特定LLM提供商
python main.py multi_agent --provider openai --model gpt-4

# 指定输出目录
python main.py multi_agent --output_dir my_agents

# 禁用Web界面
python main.py multi_agent --web_interface false
```

#### 交互式聊天

```bash
# 启动与多智能体系统的交互式聊天
python main.py agent_chat

# 使用特定LLM提供商
python main.py agent_chat --provider openai --model gpt-4
```

#### 其他命令

```bash
# 列出所有支持的LLM提供商
python main.py providers

# 测试与LLM提供商的连接
python main.py test_connection --provider openai --model gpt-4

# 查看帮助
python main.py --help
```

### Web界面

启动Web界面后，在浏览器中访问 http://localhost:12000 查看实时仪表板：

#### 小说生成界面
- 📊 **故事进度**: 实时查看字数、章节、生成进度
- 👥 **角色管理**: 监控角色状态和关系
- 🌍 **世界状态**: 查看世界模拟和事件
- 🧬 **代码演进**: 监控自我改进过程
- 📝 **内容预览**: 查看最新生成的故事内容

#### 多智能体系统界面 (开发中)
- 🧠 **智能体管理**: 创建、监控和管理智能体
- 🔄 **任务监控**: 查看任务状态和结果
- 💬 **对话可视化**: 查看智能体之间的通信
- 🛠️ **工具使用**: 监控智能体工具使用情况

## ⚙️ 配置

编辑 [`config.yaml`](config.yaml) 文件自定义设置：

```yaml
# LLM配置
llm:
  provider: "ollama"
  model: "llama3"
  base_url: "http://localhost:11434"
  temperature: 0.8
  max_tokens: 4096

# 故事配置
story:
  target_length: 5000000  # 目标字数
  chapter_length: 5000    # 每章字数
  output_dir: "output"

# 小说智能体配置
agents:
  max_agents: 10
  director_enabled: true
  character_types:
    - "protagonist"
    - "antagonist" 
    - "supporting"
    - "narrator"

# 演进配置
evolution:
  enabled: true
  mutation_rate: 0.1
  evaluation_interval: 10000

# Web界面
web_interface:
  host: "0.0.0.0"
  port: 12000
  enable_cors: true

# 多智能体系统配置
multi_agent:
  enabled: true
  max_agents: 20
  default_agent_types:
    - "assistant"
    - "expert"
    - "creative"
    - "task"
  orchestration_strategy: "hierarchical"  # hierarchical, peer-to-peer, hybrid
  
  # 智能体创建设置
  auto_create_agents: true
  agent_creation_threshold: 0.7
  
  # 任务管理设置
  task_decomposition_enabled: true
  max_concurrent_tasks: 10
  
  # 通信设置
  broadcast_enabled: true
  direct_communication_enabled: true
  
  # 记忆设置
  memory_enabled: true
  memory_retention_days: 30
  
  # 工具设置
  tool_usage_enabled: true
  default_tools_enabled: true
```

## 🏗️ Architecture

```text
novel-AI-agent/
├── src/
│   ├── core/                # Core system
│   │   ├── novel_agent.py   # Novel generation system
│   │   └── multi_agent/     # Multi-agent system
│   │       ├── multi_agent_system.py
│   │       └── config.py
│   ├── agents/              # Agent system
│   │   ├── base_agent.py    # Base agent (novel)
│   │   ├── enhanced_base_agent.py  # Enhanced base agent (general)
│   │   ├── director.py      # Director agent
│   │   ├── character.py     # Character agent
│   │   └── types/           # Specialist agent types
│   │       ├── task_agent.py
│   │       ├── assistant_agent.py
│   │       ├── expert_agent.py
│   │       └── creative_agent.py
│   ├── communication/       # Agent communication
│   │   └── message.py
│   ├── memory/              # Memory management
│   │   └── memory_manager.py
│   ├── tools/               # Tool system
│   │   └── tool_registry.py
│   ├── orchestration/       # Agent orchestration
│   │   └── orchestrator.py
│   ├── simulation/          # World simulation
│   │   └── world.py
│   ├── evolution/           # Code evolution
│   │   └── code_evolver.py
│   ├── web/                 # Web interface
│   │   └── server.py
│   └── utils/               # Utilities
│       ├── config.py
│       └── llm_client.py
├── templates/               # HTML templates
├── output/                  # Output directory
├── backups/                 # Code backups
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
└── config.yaml              # Configuration file
```

## 🚀 Agent Zero Improvements

The Novel AI Agent has been enhanced with several features inspired by the [Agent Zero](https://github.com/agent0ai/agent-zero) framework. For detailed information about these improvements, see [AGENT_ZERO_IMPROVEMENTS.md](AGENT_ZERO_IMPROVEMENTS.md).

### Vector-Based Memory

The memory system now uses FAISS for efficient vector-based retrieval:

```python
# Example: Retrieving relevant memories
memories = await memory_manager.retrieve_relevant_memories("What happened with the protagonist?")
```

Key features:
- Semantic search using embeddings
- Multiple memory types (episodic, semantic, procedural, system)
- Memory consolidation and summarization
- Automatic embedding of memory content

### Enhanced Multi-Agent Communication

The multi-agent system has been improved with better communication:

```python
# Example: Agent communication
response = await agent.process_message(
    sender_id="user_1",
    content="Can you analyze this character's motivation?",
    message_type="query"
)
```

Key features:
- Hierarchical agent structure
- Improved message passing
- Task decomposition
- Specialized agent roles

### Modern Web Interface

The web interface has been completely redesigned:

- Real-time updates via WebSockets
- Interactive chat interface
- Tabbed navigation
- Status indicators and visualizations
- Mobile-responsive design

### Docker Support

Added Docker support for easy deployment:

```bash
# Run with Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t novel-ai-agent .
docker run -p 12000:80 novel-ai-agent
```

### Embedding Capabilities

Enhanced the LLM client with embedding capabilities:

```python
# Example: Generating embeddings
embedding = await llm_client.generate_embedding("Text to embed")
```

Key features:
- Support for multiple providers (OpenAI, Cohere, local)
- Fallback mechanisms
- Batch processing

## 🔧 开发

### 小说生成系统

#### 添加新的角色智能体类型
```python
# 在 src/agents/ 中创建新的智能体类
class CustomAgent(BaseAgent):
    async def _generate_personality(self):
        # 实现个性生成逻辑
        pass
    
    async def contribute_to_story(self, story_state, world_context, scene_plan):
        # 实现故事贡献逻辑
        pass
```

#### 扩展世界模拟
```python
# 在 src/simulation/world.py 中添加新功能
async def add_custom_event_type(self, event_data):
    # 添加自定义事件类型
    pass
```

#### 自定义演进策略
```python
# 在 src/evolution/code_evolver.py 中修改演进逻辑
async def custom_improvement_strategy(self, performance_metrics):
    # 实现自定义改进策略
    pass
```

### 多智能体系统

#### 添加新的专业智能体类型
```python
# 在 src/agents/types/ 中创建新的智能体类
from ..enhanced_base_agent import EnhancedBaseAgent

class CustomSpecialistAgent(EnhancedBaseAgent):
    async def _generate_personality(self):
        # 实现个性生成逻辑
        self.personality = {
            "analytical": 0.8,
            "creative": 0.7,
            "detail_oriented": 0.9
        }
    
    async def _generate_initial_goals(self):
        # 实现目标生成逻辑
        self.goals = [
            "提供专业领域知识",
            "解决复杂问题",
            "持续学习和改进"
        ]
    
    async def _setup_capabilities(self):
        # 设置智能体能力
        self.capabilities = {
            "basic_reasoning",
            "specialized_knowledge",
            "problem_solving"
        }
        
        # 获取可用工具
        if self.tool_registry:
            self.available_tools = await self.tool_registry.get_available_tools(self.capabilities)
```

#### 添加新工具
```python
# 在 src/tools/tool_registry.py 中注册新工具
def register_custom_tools(self):
    self.register_tool(
        name="custom_tool",
        description="执行自定义操作",
        function=self._custom_tool_function,
        required_capabilities={"specialized_capability"}
    )

def _custom_tool_function(self, param1, param2):
    # 实现工具功能
    return f"处理 {param1} 和 {param2} 的结果"
```

#### 自定义编排策略
```python
# 在 src/orchestration/orchestrator.py 中添加新的编排策略
async def custom_orchestration_strategy(self, task, available_agents):
    # 实现自定义编排逻辑
    selected_agents = []
    
    # 基于任务需求选择合适的智能体
    for agent_id, agent in available_agents.items():
        if self._is_agent_suitable(agent, task):
            selected_agents.append(agent_id)
    
    return selected_agents
```

## 📊 性能监控

系统提供多种性能指标：

### 小说生成指标
- **故事质量**: LLM评估的内容质量分数
- **生成效率**: 每分钟生成的字数
- **系统稳定性**: 错误率和崩溃频率
- **代码复杂度**: 代码结构和可维护性
- **角色一致性**: 角色行为的连贯性

### 多智能体系统指标
- **任务完成率**: 成功完成的任务百分比
- **任务完成时间**: 完成任务所需的平均时间
- **智能体协作效率**: 多智能体协作的效率评分
- **资源利用率**: 系统资源使用情况
- **工具使用频率**: 各工具的使用频率和效果
- **记忆检索准确率**: 记忆检索的相关性和准确性
- **用户满意度**: 用户对系统响应的满意度评分

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Dynamic-World-Story-using-LLM-Agent-Based-Simulation](https://github.com/JackRipper01/Dynamic-World-Story-using-LLM-Agent-Based-Simulation) - 多智能体故事模拟
- [Darwin-Godel-Machine](https://github.com/mmtmn/Darwin-Godel-Machine) - 自我改进AI系统
- [OpenManus](https://github.com/FoundationAgents/OpenManus) - 多智能体协作框架
- [OpenGPTs](https://github.com/langchain-ai/opengpts) - 可定制GPT助手框架
- [CAMEL](https://github.com/camel-ai/camel) - 通信智能体框架
- [Ollama](https://ollama.ai/) - 本地LLM运行环境
- [Llama 3](https://llama.meta.com/) - 基础语言模型

## 📞 支持

如有问题或建议，欢迎通过以下方式联系我们：
- 创建 [Issue](https://github.com/ineverxxx-max/novel-AI-agent/issues) 报告问题或提出功能建议
- 发送邮件至项目维护者
- 查看 [Wiki](https://github.com/ineverxxx-max/novel-AI-agent/wiki) 获取更多详细文档

## 🚀 快速开始

1. 克隆项目：`git clone https://github.com/ineverxxx-max/novel-AI-agent.git`
2. 安装依赖：`./install.sh` 或手动安装
3. 配置LLM提供商（推荐先使用 Ollama 本地模式）
4. 运行：`python main.py generate` 开始生成小说
5. 访问 Web 界面：http://localhost:12000

---

**Novel AI Agent** - 让AI创作无限可能 ✨