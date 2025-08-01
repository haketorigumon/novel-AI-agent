# Novel AI Agent 🤖

Novel AI Agent是一款先进的 AI 代理，旨在通过代码演进生成长篇小说（五百万字以上），同时不断改进自身架构。现在它已升级为通用多智能体系统，可以处理各种任务。它集成了三项关键技术：

## 🌟 核心技术

### 1. Dynamic World Story Simulation
基于 [Dynamic-World-Story-using-LLM-Agent-Based-Simulation](https://github.com/JackRipper01/Dynamic-World-Story-using-LLM-Agent-Based-Simulation) 的多智能体模拟系统：
- 🎭 多个角色智能体，每个都有独特的个性和目标
- 🎬 导演智能体，通过环境变化引导故事发展
- 🌍 动态世界模拟，支持环境变化和事件生成
- 📚 智能叙事合成，将多个角色贡献整合为连贯故事

### 2. Darwin-Godel Machine Evolution
基于 [Darwin-Godel-Machine](https://github.com/mmtmn/Darwin-Godel-Machine) 的自我改进系统：
- 🧬 自动代码演进，基于故事生成性能
- 🔄 持续自我改进和优化
- 📊 性能评估和质量分析
- 🔒 安全的代码备份和回滚机制

### 3. 通用多智能体系统
借鉴 [OpenManus](https://github.com/FoundationAgents/OpenManus)、[OpenGPTs](https://github.com/langchain-ai/opengpts) 和 [CAMEL](https://github.com/camel-ai/camel) 的多智能体协作框架：
- 🧠 多种专业智能体类型（任务型、助手型、专家型、创意型）
- 💬 结构化智能体通信协议
- 🛠️ 工具使用能力
- 🗃️ 长期记忆管理
- 🎯 任务分解与分配
- 🔄 灵活的编排策略

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

## 📦 安装

### 前置要求
- Python 3.8+
- [Ollama](https://ollama.ai/) (用于本地LLM)

### 快速安装
```bash
# 克隆仓库
git clone https://github.com/ineverxxx-max/novel-AI-agent.git
cd novel-AI-agent

# 运行安装脚本
chmod +x install.sh
./install.sh

# 或手动安装
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
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

启动Web界面后，访问 http://localhost:12000 查看实时仪表板：

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

编辑 `config.yaml` 文件自定义设置：

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

## 🏗️ 架构

```
novel-AI-agent/
├── src/
│   ├── core/                # 核心系统
│   │   ├── novel_agent.py   # 小说生成系统
│   │   └── multi_agent/     # 多智能体系统
│   │       ├── multi_agent_system.py
│   │       └── config.py
│   ├── agents/              # 智能体系统
│   │   ├── base_agent.py    # 基础智能体（小说）
│   │   ├── enhanced_base_agent.py  # 增强基础智能体（通用）
│   │   ├── director.py      # 导演智能体
│   │   ├── character.py     # 角色智能体
│   │   └── types/           # 专业智能体类型
│   │       ├── task_agent.py
│   │       ├── assistant_agent.py
│   │       ├── expert_agent.py
│   │       └── creative_agent.py
│   ├── communication/       # 智能体通信
│   │   └── message.py
│   ├── memory/              # 记忆管理
│   │   └── memory_manager.py
│   ├── tools/               # 工具系统
│   │   └── tool_registry.py
│   ├── orchestration/       # 智能体编排
│   │   └── orchestrator.py
│   ├── simulation/          # 世界模拟
│   │   └── world.py
│   ├── evolution/           # 代码演进
│   │   └── code_evolver.py
│   ├── web/                 # Web界面
│   │   └── server.py
│   └── utils/               # 工具类
│       ├── config.py
│       └── llm_client.py
├── templates/               # HTML模板
├── output/                  # 输出目录
├── backups/                 # 代码备份
└── config.yaml              # 配置文件
```

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

如有问题或建议，请：
- 创建 [Issue](https://github.com/ineverxxx-max/novel-AI-agent/issues)
- 发送邮件至项目维护者
- 查看 [Wiki](https://github.com/ineverxxx-max/novel-AI-agent/wiki) 获取更多文档

---

**Novel AI Agent** - 让AI创作无限可能 ✨