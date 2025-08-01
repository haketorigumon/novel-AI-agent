# Novel AI Agent 🤖

Novel AI Agent是一款先进的 AI 代理，旨在通过代码演进生成长篇小说（五百万字以上），同时不断改进自身架构。它集成了两项关键技术：

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

## 🚀 功能特性

- **长篇小说生成**: 目标生成500万字以上的长篇小说
- **多智能体协作**: 角色、导演、世界模拟器协同工作
- **动态世界**: 实时环境变化和事件生成
- **自我进化**: 基于性能自动改进代码架构
- **Web界面**: 实时监控和控制面板
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

# 查看帮助
python main.py --help
```

### Web界面

启动Web界面后，访问 http://localhost:12000 查看实时仪表板：

- 📊 **故事进度**: 实时查看字数、章节、生成进度
- 👥 **角色管理**: 监控角色状态和关系
- 🌍 **世界状态**: 查看世界模拟和事件
- 🧬 **代码演进**: 监控自我改进过程
- 📝 **内容预览**: 查看最新生成的故事内容

## ⚙️ 配置

编辑 `config.yaml` 文件自定义设置：

```yaml
# LLM配置
llm:
  provider: "ollama"
  model: "llama3"
  base_url: "http://localhost:11434"
  temperature: 0.8

# 故事配置
story:
  target_length: 5000000  # 目标字数
  chapter_length: 5000    # 每章字数
  output_dir: "output"

# 智能体配置
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
```

## 🏗️ 架构

```
novel-AI-agent/
├── src/
│   ├── core/           # 核心系统
│   │   └── novel_agent.py
│   ├── agents/         # 智能体系统
│   │   ├── base_agent.py
│   │   ├── director.py
│   │   └── character.py
│   ├── simulation/     # 世界模拟
│   │   └── world.py
│   ├── evolution/      # 代码演进
│   │   └── code_evolver.py
│   ├── web/           # Web界面
│   │   └── server.py
│   └── utils/         # 工具类
│       ├── config.py
│       └── llm_client.py
├── templates/         # HTML模板
├── output/           # 生成的小说
├── backups/          # 代码备份
└── config.yaml       # 配置文件
```

## 🔧 开发

### 添加新的智能体类型
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

### 扩展世界模拟
```python
# 在 src/simulation/world.py 中添加新功能
async def add_custom_event_type(self, event_data):
    # 添加自定义事件类型
    pass
```

### 自定义演进策略
```python
# 在 src/evolution/code_evolver.py 中修改演进逻辑
async def custom_improvement_strategy(self, performance_metrics):
    # 实现自定义改进策略
    pass
```

## 📊 性能监控

系统提供多种性能指标：

- **故事质量**: LLM评估的内容质量分数
- **生成效率**: 每分钟生成的字数
- **系统稳定性**: 错误率和崩溃频率
- **代码复杂度**: 代码结构和可维护性
- **角色一致性**: 角色行为的连贯性

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
- [Ollama](https://ollama.ai/) - 本地LLM运行环境
- [Llama 3](https://llama.meta.com/) - 基础语言模型

## 📞 支持

如有问题或建议，请：
- 创建 [Issue](https://github.com/ineverxxx-max/novel-AI-agent/issues)
- 发送邮件至项目维护者
- 查看 [Wiki](https://github.com/ineverxxx-max/novel-AI-agent/wiki) 获取更多文档

---

**Novel AI Agent** - 让AI创作无限可能 ✨