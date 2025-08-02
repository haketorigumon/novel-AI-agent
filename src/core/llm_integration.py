"""
LLM Integration for Minimal Core System
Provides intelligent processing capabilities while maintaining architectural flexibility
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass

from .minimal_core import MinimalAgent, PromptEngine, Memory, MemoryLayer
from ..utils.llm_client import LLMClient


@dataclass
class LLMResponse:
    """Structured LLM response"""
    content: str
    metadata: Dict[str, Any]
    confidence: float
    reasoning: Optional[str] = None
    suggestions: Optional[List[str]] = None
    follow_up_questions: Optional[List[str]] = None


class IntelligentAgent(MinimalAgent):
    """Enhanced agent with full LLM integration"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], 
                 communication_hub, state_manager, prompt_engine: PromptEngine,
                 plugin_loader, llm_client: LLMClient):
        super().__init__(agent_id, config, communication_hub, state_manager, 
                        prompt_engine, plugin_loader)
        self.llm_client = llm_client
        self.conversation_history: List[Dict[str, Any]] = []
        self.learning_enabled = config.get('learning_enabled', True)
        self.creativity_level = config.get('creativity_level', 0.7)
        self.specialization_score = 0.0
        
    async def _process_with_llm(self, prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Enhanced LLM processing with intelligence features"""
        async with self.llm_client as client:
            # Add context and conversation history
            enhanced_prompt = await self._enhance_prompt(prompt, context or {})
            
            # Generate response
            response = await client.generate(
                enhanced_prompt,
                system_prompt=await self._get_system_prompt()
            )
            
            if not response:
                return LLMResponse(
                    content="I apologize, but I couldn't process that request.",
                    metadata={"error": "No response from LLM"},
                    confidence=0.0
                )
            
            # Analyze response quality and extract metadata
            analysis = await self._analyze_response(response, prompt)
            
            # Store in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
                "analysis": analysis,
                "context": context
            })
            
            # Learn from interaction if enabled
            if self.learning_enabled:
                await self._learn_from_interaction(prompt, response, analysis)
            
            return LLMResponse(
                content=response,
                metadata=analysis,
                confidence=analysis.get("confidence", 0.5),
                reasoning=analysis.get("reasoning"),
                suggestions=analysis.get("suggestions", []),
                follow_up_questions=analysis.get("follow_up_questions", [])
            )
    
    async def _enhance_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Enhance prompt with context and agent state"""
        # Get relevant memories
        relevant_memories = await self._get_relevant_memories(prompt)
        
        # Build enhanced prompt
        enhanced = f"""
AGENT CONTEXT:
- ID: {self.agent_id}
- Role: {self.role}
- Capabilities: {', '.join(self.capabilities)}
- Specialization Score: {self.specialization_score:.2f}
- Creativity Level: {self.creativity_level}

RELEVANT MEMORIES:
{self._format_memories(relevant_memories)}

CURRENT CONTEXT:
{json.dumps(context, indent=2)}

CONVERSATION HISTORY (last 3 interactions):
{self._format_conversation_history()}

MAIN REQUEST:
{prompt}

Please provide a thoughtful, contextually aware response that leverages your role, capabilities, and accumulated knowledge.
"""
        return enhanced
    
    async def _get_system_prompt(self) -> str:
        """Generate dynamic system prompt based on agent state"""
        return f"""You are {self.agent_id}, an intelligent AI agent with the following characteristics:

Role: {self.role}
Capabilities: {', '.join(self.capabilities)}
Personality Traits: {json.dumps(self.execution_context.get('personality', {}), indent=2)}

You are part of a flexible, adaptive AI system. Your responses should be:
1. Contextually aware and relevant
2. Consistent with your role and capabilities
3. Helpful and constructive
4. Adaptive to the user's needs
5. Honest about your limitations

You can collaborate with other agents, use tools, store memories, and adapt your behavior based on experience.
Always strive to provide the most helpful and accurate response possible."""
    
    async def _analyze_response(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """Analyze response quality and extract metadata"""
        async with self.llm_client as client:
            analysis_prompt = f"""
Analyze this AI response for quality and extract metadata:

Original Prompt: "{original_prompt}"
AI Response: "{response}"

Provide analysis in JSON format:
{{
    "confidence": 0.0-1.0,
    "relevance": 0.0-1.0,
    "completeness": 0.0-1.0,
    "creativity": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "helpfulness": 0.0-1.0,
    "reasoning": "explanation of the response approach",
    "strengths": ["list", "of", "strengths"],
    "weaknesses": ["list", "of", "weaknesses"],
    "suggestions": ["improvement", "suggestions"],
    "follow_up_questions": ["relevant", "follow", "up", "questions"],
    "topics": ["main", "topics", "covered"],
    "sentiment": "positive|neutral|negative",
    "complexity": "low|medium|high"
}}
"""
            
            analysis = await client.generate_structured(
                analysis_prompt,
                {
                    "confidence": "number",
                    "relevance": "number",
                    "completeness": "number",
                    "creativity": "number",
                    "accuracy": "number",
                    "helpfulness": "number",
                    "reasoning": "string",
                    "strengths": "array",
                    "weaknesses": "array",
                    "suggestions": "array",
                    "follow_up_questions": "array",
                    "topics": "array",
                    "sentiment": "string",
                    "complexity": "string"
                },
                "You are a response quality analyzer. Provide objective analysis of AI responses."
            )
            
            return analysis or {
                "confidence": 0.5,
                "relevance": 0.5,
                "completeness": 0.5,
                "reasoning": "Analysis unavailable"
            }
    
    async def _learn_from_interaction(self, prompt: str, response: str, analysis: Dict[str, Any]):
        """Learn and adapt from interactions"""
        # Update specialization score based on performance
        performance_score = (
            analysis.get("confidence", 0.5) * 0.3 +
            analysis.get("relevance", 0.5) * 0.3 +
            analysis.get("helpfulness", 0.5) * 0.4
        )
        
        # Exponential moving average for specialization
        self.specialization_score = (
            0.9 * self.specialization_score + 0.1 * performance_score
        )
        
        # Adapt creativity level based on task type
        if analysis.get("complexity") == "high" and analysis.get("creativity", 0.5) > 0.7:
            self.creativity_level = min(1.0, self.creativity_level + 0.05)
        elif analysis.get("accuracy", 0.5) < 0.6:
            self.creativity_level = max(0.3, self.creativity_level - 0.05)
        
        # Store learning memory
        await self._store_memory(
            content={
                "interaction": {
                    "prompt": prompt,
                    "response": response,
                    "performance": performance_score
                },
                "learning": {
                    "specialization_change": performance_score - self.specialization_score,
                    "creativity_adjustment": analysis.get("creativity", 0.5),
                    "topics_learned": analysis.get("topics", [])
                }
            },
            layer=MemoryLayer.EPISODIC,
            importance=performance_score,
            metadata={
                "type": "learning",
                "performance_score": performance_score,
                "topics": analysis.get("topics", [])
            }
        )
        
        # Update execution context
        self.execution_context.update({
            "specialization_score": self.specialization_score,
            "creativity_level": self.creativity_level,
            "last_performance": performance_score,
            "interaction_count": self.execution_context.get("interaction_count", 0) + 1
        })
        
        await self._save_state()
    
    async def _get_relevant_memories(self, query: str, limit: int = 5) -> List[Memory]:
        """Get memories relevant to the current query"""
        all_memories = await self.state_manager.load_memories(self.agent_id)
        
        if not all_memories:
            return []
        
        # Simple relevance scoring (in a full implementation, this would use embeddings)
        query_words = set(query.lower().split())
        scored_memories = []
        
        for memory in all_memories:
            content_str = str(memory.content).lower()
            content_words = set(content_str.split())
            
            # Calculate relevance score
            overlap = len(query_words.intersection(content_words))
            relevance = overlap / max(len(query_words), 1)
            
            # Boost score based on importance and recency
            age_factor = max(0.1, 1.0 - (datetime.now() - memory.last_accessed).days / 30)
            final_score = relevance * memory.importance * age_factor
            
            scored_memories.append((final_score, memory))
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]
    
    def _format_memories(self, memories: List[Memory]) -> str:
        """Format memories for prompt inclusion"""
        if not memories:
            return "No relevant memories found."
        
        formatted = []
        for memory in memories:
            formatted.append(f"- [{memory.layer.value}] {memory.content} (importance: {memory.importance:.2f})")
        
        return "\n".join(formatted)
    
    def _format_conversation_history(self) -> str:
        """Format recent conversation history"""
        if not self.conversation_history:
            return "No previous conversations."
        
        recent = self.conversation_history[-3:]
        formatted = []
        
        for i, conv in enumerate(recent, 1):
            formatted.append(f"{i}. Prompt: {conv['prompt'][:100]}...")
            formatted.append(f"   Response: {conv['response'][:100]}...")
            formatted.append(f"   Performance: {conv['analysis'].get('confidence', 0.5):.2f}")
        
        return "\n".join(formatted)
    
    async def collaborate_with_agent(self, other_agent_id: str, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Collaborate with another agent on a task"""
        collaboration_prompt = await self.prompt_engine.generate_prompt(
            "agent_collaboration",
            task=task,
            other_agent=other_agent_id,
            my_capabilities=self.capabilities,
            context=context or {},
            collaboration_history=await self._get_collaboration_history(other_agent_id)
        )
        
        response = await self._process_with_llm(collaboration_prompt, context)
        
        # Store collaboration memory
        await self._store_memory(
            content={
                "type": "collaboration",
                "partner": other_agent_id,
                "task": task,
                "my_contribution": response.content,
                "success": response.confidence > 0.7
            },
            layer=MemoryLayer.EPISODIC,
            importance=0.8,
            metadata={"collaboration_partner": other_agent_id}
        )
        
        return {
            "contribution": response.content,
            "confidence": response.confidence,
            "reasoning": response.reasoning,
            "suggestions": response.suggestions
        }
    
    async def _get_collaboration_history(self, other_agent_id: str) -> List[Dict[str, Any]]:
        """Get history of collaborations with specific agent"""
        memories = await self.state_manager.load_memories(self.agent_id)
        collaborations = []
        
        for memory in memories:
            if (isinstance(memory.content, dict) and 
                memory.content.get("type") == "collaboration" and
                memory.content.get("partner") == other_agent_id):
                collaborations.append(memory.content)
        
        return collaborations[-5:]  # Last 5 collaborations
    
    async def generate_plugin(self, requirement: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a plugin based on requirements"""
        plugin_prompt = await self.prompt_engine.generate_prompt(
            "plugin_generation",
            requirement=requirement,
            context=context or {},
            my_capabilities=self.capabilities,
            available_interfaces=["PluginInterface", "AsyncPlugin", "ToolPlugin"]
        )
        
        response = await self._process_with_llm(plugin_prompt, context)
        
        # Extract code from response (simplified)
        plugin_code = self._extract_code_from_response(response.content)
        
        if plugin_code:
            # Save plugin to file
            plugin_file = await self.plugin_loader.save_generated_plugin(
                plugin_code, 
                f"generated_by_{self.agent_id}_{uuid.uuid4().hex[:8]}"
            )
            
            # Store plugin creation memory
            await self._store_memory(
                content={
                    "type": "plugin_creation",
                    "requirement": requirement,
                    "plugin_file": plugin_file,
                    "success": True
                },
                layer=MemoryLayer.SEMANTIC,
                importance=0.9,
                metadata={"plugin_file": plugin_file}
            )
            
            return {
                "success": True,
                "plugin_file": plugin_file,
                "code": plugin_code,
                "confidence": response.confidence
            }
        else:
            return {
                "success": False,
                "error": "Could not extract valid plugin code",
                "raw_response": response.content
            }
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response"""
        import re
        
        # Look for code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # Look for class definitions
        class_match = re.search(r'(class \w+.*?(?=\n\n|\n$|\Z))', response, re.DOTALL)
        if class_match:
            return class_match.group(1)
        
        return None
    
    async def self_improve(self) -> Dict[str, Any]:
        """Analyze performance and suggest self-improvements"""
        improvement_prompt = f"""
Analyze my performance and suggest improvements:

Agent ID: {self.agent_id}
Role: {self.role}
Current Specialization Score: {self.specialization_score:.2f}
Creativity Level: {self.creativity_level:.2f}
Total Interactions: {self.execution_context.get('interaction_count', 0)}

Recent Performance Data:
{json.dumps(self.conversation_history[-5:], indent=2, default=str)}

Current Capabilities: {', '.join(self.capabilities)}

Suggest improvements in JSON format:
{{
    "performance_analysis": "overall performance assessment",
    "strengths": ["identified", "strengths"],
    "weaknesses": ["identified", "weaknesses"],
    "improvement_suggestions": [
        {{
            "area": "area to improve",
            "suggestion": "specific suggestion",
            "priority": "high|medium|low",
            "implementation": "how to implement"
        }}
    ],
    "new_capabilities": ["suggested", "new", "capabilities"],
    "parameter_adjustments": {{
        "creativity_level": 0.0-1.0,
        "specialization_focus": "area to focus on"
    }}
}}
"""
        
        response = await self._process_with_llm(improvement_prompt)
        
        try:
            improvements = json.loads(response.content)
            
            # Apply parameter adjustments if suggested
            if "parameter_adjustments" in improvements:
                adjustments = improvements["parameter_adjustments"]
                if "creativity_level" in adjustments:
                    self.creativity_level = max(0.0, min(1.0, adjustments["creativity_level"]))
            
            # Store self-improvement memory
            await self._store_memory(
                content={
                    "type": "self_improvement",
                    "analysis": improvements,
                    "applied_changes": {
                        "creativity_level": self.creativity_level
                    }
                },
                layer=MemoryLayer.META,
                importance=1.0,
                metadata={"improvement_session": datetime.now().isoformat()}
            )
            
            await self._save_state()
            
            return improvements
            
        except json.JSONDecodeError:
            return {
                "error": "Could not parse improvement suggestions",
                "raw_response": response.content
            }
    
    async def get_agent_insights(self) -> Dict[str, Any]:
        """Get insights about agent performance and behavior"""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "specialization_score": self.specialization_score,
            "creativity_level": self.creativity_level,
            "total_interactions": self.execution_context.get("interaction_count", 0),
            "memory_count": len(await self.state_manager.load_memories(self.agent_id)),
            "recent_performance": [
                conv["analysis"].get("confidence", 0.5) 
                for conv in self.conversation_history[-10:]
            ],
            "top_topics": self._get_top_topics(),
            "collaboration_partners": self._get_collaboration_partners(),
            "learning_trajectory": self._get_learning_trajectory()
        }
    
    def _get_top_topics(self) -> List[str]:
        """Get most frequently discussed topics"""
        topic_counts = {}
        for conv in self.conversation_history:
            for topic in conv.get("analysis", {}).get("topics", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return sorted(topic_counts.keys(), key=topic_counts.get, reverse=True)[:5]
    
    def _get_collaboration_partners(self) -> List[str]:
        """Get list of agents this agent has collaborated with"""
        partners = set()
        for conv in self.conversation_history:
            if "collaboration_partner" in conv.get("context", {}):
                partners.add(conv["context"]["collaboration_partner"])
        
        return list(partners)
    
    def _get_learning_trajectory(self) -> List[Dict[str, Any]]:
        """Get learning trajectory over time"""
        trajectory = []
        for i, conv in enumerate(self.conversation_history):
            trajectory.append({
                "interaction": i + 1,
                "confidence": conv["analysis"].get("confidence", 0.5),
                "topics": conv["analysis"].get("topics", []),
                "timestamp": conv["timestamp"]
            })
        
        return trajectory[-20:]  # Last 20 interactions