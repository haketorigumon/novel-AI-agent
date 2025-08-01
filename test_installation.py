#!/usr/bin/env python3
"""Test script to verify Novel AI Agent installation"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from src.utils.config import Config
        print("✅ Config module imported successfully")
        
        from src.utils.llm_client import LLMClient
        print("✅ LLM Client module imported successfully")
        
        from src.core.novel_agent import NovelAIAgent
        print("✅ Novel Agent module imported successfully")
        
        from src.agents.base_agent import BaseAgent
        from src.agents.director import DirectorAgent
        from src.agents.character import CharacterAgent
        print("✅ Agent modules imported successfully")
        
        from src.simulation.world import WorldSimulation
        print("✅ World Simulation module imported successfully")
        
        from src.evolution.code_evolver import CodeEvolver
        print("✅ Code Evolver module imported successfully")
        
        from src.web.server import WebServer
        print("✅ Web Server module imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

async def test_config():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from src.utils.config import Config
        
        # Test default config
        config = Config()
        print("✅ Default configuration created")
        
        # Test config loading from file
        if Path("config.yaml").exists():
            config = Config.load("config.yaml")
            print("✅ Configuration loaded from config.yaml")
        else:
            print("⚠️ config.yaml not found, using defaults")
        
        # Validate config structure
        assert hasattr(config, 'llm')
        assert hasattr(config, 'story')
        assert hasattr(config, 'agents')
        assert hasattr(config, 'evolution')
        assert hasattr(config, 'simulation')
        assert hasattr(config, 'web_interface')
        print("✅ Configuration structure is valid")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

async def test_llm_client():
    """Test LLM client initialization"""
    print("\n🤖 Testing LLM client...")
    
    try:
        from src.utils.llm_client import LLMClient
        from src.utils.config import Config
        
        config = Config()
        client = LLMClient(
            provider=config.llm.provider,
            model=config.llm.model,
            base_url=config.llm.base_url
        )
        print("✅ LLM Client initialized")
        
        # Test provider support
        providers = LLMClient.get_supported_providers()
        assert len(providers) >= 14, f"Expected at least 14 providers, got {len(providers)}"
        assert "ollama" in providers
        assert "openai" in providers
        assert "anthropic" in providers
        print(f"✅ Provider support verified ({len(providers)} providers)")
        
        # Test provider info
        ollama_info = LLMClient.get_provider_info("ollama")
        assert ollama_info["requires_api_key"] == False
        openai_info = LLMClient.get_provider_info("openai")
        assert openai_info["requires_api_key"] == True
        print("✅ Provider information correct")
        
        # Test connection (this will fail if Ollama is not running, but that's OK)
        async with client as c:
            connected = await c.check_connection()
            if connected:
                print("✅ LLM service is available")
            else:
                print("⚠️ LLM service not available (Ollama may not be running)")
        
        return True
    except Exception as e:
        print(f"❌ LLM Client error: {e}")
        return False

async def test_agent_creation():
    """Test agent creation"""
    print("\n👥 Testing agent creation...")
    
    try:
        from src.utils.config import Config
        from src.utils.llm_client import LLMClient
        from src.agents.character import CharacterAgent
        from src.simulation.world import WorldSimulation
        
        config = Config()
        llm_client = LLMClient(
            provider=config.llm.provider,
            model=config.llm.model,
            base_url=config.llm.base_url
        )
        
        # Create world simulation
        world_sim = WorldSimulation(config, llm_client)
        print("✅ World simulation created")
        
        # Create character agent
        character = CharacterAgent(
            agent_id="test_character",
            character_type="protagonist",
            config=config,
            llm_client=llm_client,
            world_simulation=world_sim
        )
        print("✅ Character agent created")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        return False

async def test_directory_structure():
    """Test that all required directories exist"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "src",
        "src/core",
        "src/agents", 
        "src/simulation",
        "src/evolution",
        "src/web",
        "src/utils",
        "templates",
        "output",
        "backups"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            all_exist = False
    
    return all_exist

async def main():
    """Run all tests"""
    print("🚀 Novel AI Agent Installation Test\n")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("LLM Client", test_llm_client),
        ("Agent Creation", test_agent_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Installation appears to be successful.")
        print("\nNext steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Download Llama 3 model: ollama pull llama3")
        print("3. Start the web interface: python main.py web")
        print("4. Or generate a novel: python main.py generate")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Please check the installation.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)