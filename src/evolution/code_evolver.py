"""Code Evolver - Self-improving system based on Darwin-Godel Machine principles"""

import os
import json
import shutil
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..utils.config import Config
from ..utils.llm_client import LLMClient

class CodeEvolver:
    """Implements self-improving code evolution based on story generation performance"""
    
    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client
        self.evolution_history = []
        self.current_generation = 0
        self.performance_metrics = {}
        self.backup_dir = Path("backups")
        self.src_dir = Path("src")
        
    async def initialize(self):
        """Initialize the code evolution system"""
        self.backup_dir.mkdir(exist_ok=True)
        await self._load_evolution_history()
        await self._analyze_current_codebase()
    
    async def _load_evolution_history(self):
        """Load previous evolution history"""
        history_file = self.backup_dir / "evolution_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.evolution_history = data.get("history", [])
                    self.current_generation = data.get("current_generation", 0)
            except Exception as e:
                print(f"Warning: Could not load evolution history: {e}")
    
    async def _save_evolution_history(self):
        """Save evolution history"""
        history_file = self.backup_dir / "evolution_history.json"
        data = {
            "history": self.evolution_history,
            "current_generation": self.current_generation,
            "last_updated": datetime.now().isoformat()
        }
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def _analyze_current_codebase(self):
        """Analyze the current codebase structure and quality"""
        analysis = {
            "total_files": 0,
            "total_lines": 0,
            "complexity_score": 0,
            "modularity_score": 0,
            "file_structure": {}
        }
        
        if self.src_dir.exists():
            for py_file in self.src_dir.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = len(content.splitlines())
                        analysis["total_files"] += 1
                        analysis["total_lines"] += lines
                        
                        # Simple complexity analysis
                        complexity = self._calculate_file_complexity(content)
                        analysis["file_structure"][str(py_file)] = {
                            "lines": lines,
                            "complexity": complexity
                        }
                except Exception:
                    continue
        
        # Calculate overall scores
        if analysis["total_files"] > 0:
            avg_complexity = sum(f["complexity"] for f in analysis["file_structure"].values()) / analysis["total_files"]
            analysis["complexity_score"] = min(1.0, avg_complexity / 100)  # Normalize
            analysis["modularity_score"] = min(1.0, analysis["total_files"] / 20)  # More files = better modularity
        
        self.performance_metrics["codebase_analysis"] = analysis
    
    def _calculate_file_complexity(self, content: str) -> float:
        """Calculate a simple complexity score for a file"""
        lines = content.splitlines()
        complexity = 0
        
        for line in lines:
            line = line.strip()
            # Count complexity indicators
            if line.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'with ')):
                complexity += 1
            elif line.startswith(('def ', 'class ', 'async def ')):
                complexity += 2
            elif 'lambda' in line:
                complexity += 1
        
        return complexity
    
    async def evolve_system(self, story_state: Dict[str, Any]):
        """Main evolution process"""
        print(f"Starting evolution generation {self.current_generation + 1}")
        
        # Evaluate current performance
        performance = await self._evaluate_performance(story_state)
        
        # Identify areas for improvement
        improvements = await self._identify_improvements(performance, story_state)
        
        if not improvements:
            print("No improvements identified, skipping evolution")
            return
        
        # Create backup of current generation
        await self._create_backup()
        
        # Apply improvements
        success = await self._apply_improvements(improvements)
        
        if success:
            self.current_generation += 1
            evolution_record = {
                "generation": self.current_generation,
                "timestamp": datetime.now().isoformat(),
                "performance_before": performance,
                "improvements_applied": improvements,
                "success": True
            }
            self.evolution_history.append(evolution_record)
            await self._save_evolution_history()
            print(f"Evolution generation {self.current_generation} completed successfully")
        else:
            # Rollback on failure
            await self._rollback_changes()
            print("Evolution failed, rolled back changes")
    
    async def _evaluate_performance(self, story_state: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate current system performance"""
        performance = {}
        
        # Story generation metrics
        word_count = story_state.get("current_word_count", 0)
        target_length = self.config.story.target_length
        
        performance["story_progress"] = min(1.0, word_count / target_length)
        performance["generation_efficiency"] = self._calculate_generation_efficiency(story_state)
        performance["story_quality"] = await self._estimate_story_quality(story_state)
        performance["system_stability"] = self._assess_system_stability()
        
        # Code quality metrics
        performance["code_complexity"] = self.performance_metrics.get("codebase_analysis", {}).get("complexity_score", 0.5)
        performance["code_modularity"] = self.performance_metrics.get("codebase_analysis", {}).get("modularity_score", 0.5)
        
        return performance
    
    def _calculate_generation_efficiency(self, story_state: Dict[str, Any]) -> float:
        """Calculate how efficiently the system generates story content"""
        # Simple metric based on word count vs time (if we had timing data)
        # For now, return a baseline score
        return 0.7
    
    async def _estimate_story_quality(self, story_state: Dict[str, Any]) -> float:
        """Estimate story quality using LLM analysis"""
        if not story_state.get("story_content"):
            return 0.5
        
        # Analyze recent story content
        recent_content = story_state["story_content"][-3:] if story_state["story_content"] else []
        if not recent_content:
            return 0.5
        
        sample_content = "\n\n".join(recent_content)
        
        async with self.llm_client as client:
            quality_prompt = f"""
Analyze this story content for quality on a scale of 0.0 to 1.0:

{sample_content[:2000]}...

Rate the content on:
1. Narrative coherence
2. Character development
3. Engaging prose
4. Plot advancement
5. Overall readability

Respond with just a number between 0.0 and 1.0 representing overall quality.
"""
            
            response = await client.generate(quality_prompt, 
                "You are a literary critic evaluating story quality.")
            
            try:
                # Extract numeric score
                import re
                match = re.search(r'0\.\d+|1\.0|0|1', response)
                if match:
                    return float(match.group())
            except:
                pass
        
        return 0.6  # Default score
    
    def _assess_system_stability(self) -> float:
        """Assess system stability based on error history"""
        # For now, return a baseline stability score
        # In a real implementation, this would track errors and crashes
        return 0.8
    
    async def _identify_improvements(self, performance: Dict[str, float], story_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific improvements to make"""
        improvements = []
        
        # Analyze performance bottlenecks
        for metric, score in performance.items():
            if score < 0.6:  # Threshold for improvement
                improvement = await self._suggest_improvement(metric, score, story_state)
                if improvement:
                    improvements.append(improvement)
        
        # Limit improvements per generation
        return improvements[:3]
    
    async def _suggest_improvement(self, metric: str, score: float, story_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Suggest specific improvement for a metric"""
        async with self.llm_client as client:
            improvement_prompt = f"""
The story generation system has a low score ({score:.2f}) for metric: {metric}

Current story state: {json.dumps({
                "word_count": story_state.get("current_word_count", 0),
                "chapter": story_state.get("current_chapter", 1),
                "content_samples": len(story_state.get("story_content", []))
            }, indent=2)}

Suggest a specific code improvement that could address this issue.
Focus on:
1. Concrete changes to existing code
2. New features or algorithms
3. Better prompting strategies
4. Improved data structures

Respond with JSON:
{{
    "improvement_type": "code_modification/new_feature/algorithm_change",
    "target_file": "path/to/file.py",
    "description": "what to change and why",
    "expected_impact": "how this will improve the metric",
    "implementation_complexity": "low/medium/high",
    "code_suggestion": "specific code changes or additions"
}}
"""
            
            improvement = await client.generate_structured(
                improvement_prompt,
                {
                    "improvement_type": "string",
                    "target_file": "string",
                    "description": "string",
                    "expected_impact": "string",
                    "implementation_complexity": "string",
                    "code_suggestion": "string"
                },
                "You are a senior software engineer optimizing an AI story generation system."
            )
            
            if improvement and improvement.get("implementation_complexity") != "high":
                improvement["metric"] = metric
                improvement["current_score"] = score
                return improvement
        
        return None
    
    async def _create_backup(self):
        """Create backup of current codebase"""
        backup_path = self.backup_dir / f"generation_{self.current_generation}"
        
        if backup_path.exists():
            shutil.rmtree(backup_path)
        
        # Copy source directory
        if self.src_dir.exists():
            shutil.copytree(self.src_dir, backup_path / "src")
        
        # Copy config files
        for config_file in ["config.yaml", "requirements.txt"]:
            if Path(config_file).exists():
                shutil.copy2(config_file, backup_path / config_file)
        
        print(f"Created backup at {backup_path}")
    
    async def _apply_improvements(self, improvements: List[Dict[str, Any]]) -> bool:
        """Apply the suggested improvements"""
        success_count = 0
        
        for improvement in improvements:
            try:
                success = await self._apply_single_improvement(improvement)
                if success:
                    success_count += 1
                    print(f"Applied improvement: {improvement['description']}")
                else:
                    print(f"Failed to apply improvement: {improvement['description']}")
            except Exception as e:
                print(f"Error applying improvement: {e}")
        
        # Consider successful if at least half of improvements were applied
        return success_count >= len(improvements) // 2
    
    async def _apply_single_improvement(self, improvement: Dict[str, Any]) -> bool:
        """Apply a single improvement"""
        target_file = Path(improvement.get("target_file", ""))
        
        if not target_file.exists():
            print(f"Target file does not exist: {target_file}")
            return False
        
        improvement_type = improvement.get("improvement_type", "")
        code_suggestion = improvement.get("code_suggestion", "")
        
        if improvement_type == "code_modification":
            return await self._modify_existing_code(target_file, code_suggestion, improvement)
        elif improvement_type == "new_feature":
            return await self._add_new_feature(target_file, code_suggestion, improvement)
        elif improvement_type == "algorithm_change":
            return await self._change_algorithm(target_file, code_suggestion, improvement)
        
        return False
    
    async def _modify_existing_code(self, target_file: Path, code_suggestion: str, improvement: Dict) -> bool:
        """Modify existing code in a file"""
        try:
            # Read current file
            with open(target_file, 'r', encoding='utf-8') as f:
                current_content = f.read()
            
            # Use LLM to apply the modification
            async with self.llm_client as client:
                modification_prompt = f"""
Apply this code improvement to the existing file:

Current file content:
{current_content}

Improvement description: {improvement['description']}
Code suggestion: {code_suggestion}

Provide the complete modified file content. Make minimal, targeted changes.
Ensure the code remains syntactically correct and maintains existing functionality.
"""
                
                modified_content = await client.generate(modification_prompt,
                    "You are a senior Python developer making careful code improvements.")
                
                # Basic validation
                if len(modified_content) > 100 and "def " in modified_content:
                    # Write modified content
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    return True
        
        except Exception as e:
            print(f"Error modifying code: {e}")
        
        return False
    
    async def _add_new_feature(self, target_file: Path, code_suggestion: str, improvement: Dict) -> bool:
        """Add a new feature to existing code"""
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                current_content = f.read()
            
            # Add the new feature at the end of the class or file
            if "class " in current_content:
                # Find the last method in the class and add after it
                lines = current_content.splitlines()
                insert_point = len(lines)
                
                # Find a good insertion point
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() and not lines[i].startswith(' '):
                        insert_point = i + 1
                        break
                
                # Insert the new feature
                new_lines = lines[:insert_point] + ["", f"    # New feature: {improvement['description']}", code_suggestion] + lines[insert_point:]
                modified_content = "\n".join(new_lines)
                
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                return True
        
        except Exception as e:
            print(f"Error adding new feature: {e}")
        
        return False
    
    async def _change_algorithm(self, target_file: Path, code_suggestion: str, improvement: Dict) -> bool:
        """Change an algorithm in existing code"""
        # For now, treat this the same as code modification
        return await self._modify_existing_code(target_file, code_suggestion, improvement)
    
    async def _rollback_changes(self):
        """Rollback to previous generation"""
        backup_path = self.backup_dir / f"generation_{self.current_generation}"
        
        if backup_path.exists():
            # Remove current src directory
            if self.src_dir.exists():
                shutil.rmtree(self.src_dir)
            
            # Restore from backup
            if (backup_path / "src").exists():
                shutil.copytree(backup_path / "src", self.src_dir)
            
            print("Rolled back to previous generation")
        else:
            print("No backup found for rollback")
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status"""
        return {
            "current_generation": self.current_generation,
            "total_evolutions": len(self.evolution_history),
            "last_evolution": self.evolution_history[-1] if self.evolution_history else None,
            "performance_metrics": self.performance_metrics,
            "backup_count": len(list(self.backup_dir.glob("generation_*"))) if self.backup_dir.exists() else 0
        }