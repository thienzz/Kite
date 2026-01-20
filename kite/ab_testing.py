"""
A/B Testing Framework for Agents
Test different prompts, models, and configurations.
"""

import time
import random
import hashlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class Variant:
    """A/B test variant configuration."""
    name: str
    weight: float  # 0-1, must sum to 1.0 across variants
    config: Dict[str, Any]
    
    # Metrics
    impressions: int = 0
    conversions: int = 0
    total_latency: float = 0
    errors: int = 0
    
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        return self.conversions / self.impressions if self.impressions > 0 else 0
    
    def avg_latency(self) -> float:
        """Calculate average latency."""
        return self.total_latency / self.impressions if self.impressions > 0 else 0
    
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.errors / self.impressions if self.impressions > 0 else 0


@dataclass
class Experiment:
    """A/B test experiment."""
    name: str
    description: str
    variants: List[Variant]
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    active: bool = True
    
    def __post_init__(self):
        # Validate weights sum to 1.0
        total_weight = sum(v.weight for v in self.variants)
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Variant weights must sum to 1.0, got {total_weight}")
    
    def get_variant(self, user_id: str) -> Variant:
        """
        Get variant for user (consistent assignment).
        Uses hash-based assignment for consistency.
        """
        # Hash user_id to get consistent assignment
        hash_value = int(hashlib.md5(
            f"{self.name}:{user_id}".encode()
        ).hexdigest(), 16)
        
        # Normalize to 0-1
        normalized = (hash_value % 10000) / 10000.0
        
        # Select variant based on cumulative weights
        cumulative = 0
        for variant in self.variants:
            cumulative += variant.weight
            if normalized <= cumulative:
                return variant
        
        return self.variants[-1]  # Fallback
    
    def record_impression(self, variant_name: str):
        """Record an impression."""
        for v in self.variants:
            if v.name == variant_name:
                v.impressions += 1
                break
    
    def record_conversion(self, variant_name: str):
        """Record a conversion."""
        for v in self.variants:
            if v.name == variant_name:
                v.conversions += 1
                break
    
    def record_latency(self, variant_name: str, latency: float):
        """Record latency."""
        for v in self.variants:
            if v.name == variant_name:
                v.total_latency += latency
                break
    
    def record_error(self, variant_name: str):
        """Record an error."""
        for v in self.variants:
            if v.name == variant_name:
                v.errors += 1
                break
    
    def get_results(self) -> Dict:
        """Get experiment results."""
        results = {
            'name': self.name,
            'description': self.description,
            'duration': (self.end_time or time.time()) - self.start_time,
            'active': self.active,
            'variants': []
        }
        
        for v in self.variants:
            results['variants'].append({
                'name': v.name,
                'weight': v.weight,
                'impressions': v.impressions,
                'conversions': v.conversions,
                'conversion_rate': v.conversion_rate(),
                'avg_latency': v.avg_latency(),
                'error_rate': v.error_rate()
            })
        
        # Calculate winner
        if all(v.impressions >= 100 for v in self.variants):
            winner = max(self.variants, key=lambda v: v.conversion_rate())
            results['winner'] = winner.name
            results['confidence'] = self._calculate_confidence(winner)
        
        return results
    
    def _calculate_confidence(self, winner: Variant) -> float:
        """Simple confidence calculation."""
        # Simplified - in production use proper statistical tests
        if winner.impressions < 100:
            return 0.0
        
        others_avg = sum(
            v.conversion_rate() 
            for v in self.variants 
            if v != winner
        ) / (len(self.variants) - 1)
        
        improvement = (winner.conversion_rate() - others_avg) / others_avg if others_avg > 0 else 0
        
        return min(improvement * 10, 1.0)  # Simplified


class ABTestManager:
    """
    Manage multiple A/B tests.
    
    Example:
        manager = ABTestManager()
        
        # Create experiment
        exp = manager.create_experiment(
            name="prompt_test",
            description="Test different system prompts",
            variants=[
                Variant("control", 0.5, {"prompt": "You are helpful"}),
                Variant("friendly", 0.5, {"prompt": "You are very friendly"})
            ]
        )
        
        # Get variant for user
        variant = manager.get_variant("prompt_test", user_id)
        
        # Use variant config
        response = llm.chat(system_prompt=variant.config['prompt'])
        
        # Record metrics
        manager.record_impression("prompt_test", variant.name)
        manager.record_conversion("prompt_test", variant.name)
    """
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
    
    def create_experiment(self, name: str, description: str,
                         variants: List[Variant]) -> Experiment:
        """Create new experiment."""
        if name in self.experiments:
            raise ValueError(f"Experiment {name} already exists")
        
        exp = Experiment(name, description, variants)
        self.experiments[name] = exp
        return exp
    
    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get experiment by name."""
        return self.experiments.get(name)
    
    def get_variant(self, experiment_name: str, user_id: str) -> Optional[Variant]:
        """Get variant for user."""
        exp = self.experiments.get(experiment_name)
        if exp and exp.active:
            return exp.get_variant(user_id)
        return None
    
    def record_impression(self, experiment_name: str, variant_name: str):
        """Record impression."""
        exp = self.experiments.get(experiment_name)
        if exp:
            exp.record_impression(variant_name)
    
    def record_conversion(self, experiment_name: str, variant_name: str):
        """Record conversion."""
        exp = self.experiments.get(experiment_name)
        if exp:
            exp.record_conversion(variant_name)
    
    def record_latency(self, experiment_name: str, variant_name: str, 
                      latency: float):
        """Record latency."""
        exp = self.experiments.get(experiment_name)
        if exp:
            exp.record_latency(variant_name, latency)
    
    def record_error(self, experiment_name: str, variant_name: str):
        """Record error."""
        exp = self.experiments.get(experiment_name)
        if exp:
            exp.record_error(variant_name)
    
    def stop_experiment(self, name: str):
        """Stop experiment."""
        exp = self.experiments.get(name)
        if exp:
            exp.active = False
            exp.end_time = time.time()
    
    def get_results(self, name: str) -> Optional[Dict]:
        """Get experiment results."""
        exp = self.experiments.get(name)
        return exp.get_results() if exp else None
    
    def list_experiments(self) -> List[str]:
        """List all experiments."""
        return list(self.experiments.keys())


class PromptVersionManager:
    """
    Manage prompt versions with A/B testing.
    
    Example:
        manager = PromptVersionManager()
        
        # Register prompts
        manager.register("system_prompt", "v1", "You are helpful")
        manager.register("system_prompt", "v2", "You are very helpful")
        
        # Create A/B test
        manager.create_test(
            "system_prompt",
            versions=["v1", "v2"],
            weights=[0.5, 0.5]
        )
        
        # Get prompt for user
        prompt = manager.get_prompt("system_prompt", user_id)
    """
    
    def __init__(self):
        self.prompts: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.ab_manager = ABTestManager()
    
    def register(self, prompt_name: str, version: str, content: str):
        """Register a prompt version."""
        self.prompts[prompt_name][version] = content
    
    def get_versions(self, prompt_name: str) -> List[str]:
        """Get all versions of a prompt."""
        return list(self.prompts[prompt_name].keys())
    
    def create_test(self, prompt_name: str, versions: List[str],
                   weights: List[float], description: str = ""):
        """Create A/B test for prompt versions."""
        variants = []
        for version, weight in zip(versions, weights):
            if version not in self.prompts[prompt_name]:
                raise ValueError(f"Version {version} not found")
            
            variants.append(Variant(
                name=version,
                weight=weight,
                config={'prompt': self.prompts[prompt_name][version]}
            ))
        
        self.ab_manager.create_experiment(
            name=f"prompt_{prompt_name}",
            description=description or f"Test {prompt_name} versions",
            variants=variants
        )
    
    def get_prompt(self, prompt_name: str, user_id: str) -> str:
        """Get prompt for user (with A/B test if active)."""
        exp_name = f"prompt_{prompt_name}"
        variant = self.ab_manager.get_variant(exp_name, user_id)
        
        if variant:
            self.ab_manager.record_impression(exp_name, variant.name)
            return variant.config['prompt']
        
        # Fallback to latest version
        versions = self.get_versions(prompt_name)
        if versions:
            return self.prompts[prompt_name][versions[-1]]
        
        return ""
    
    def record_success(self, prompt_name: str, user_id: str):
        """Record successful use of prompt."""
        exp_name = f"prompt_{prompt_name}"
        variant = self.ab_manager.get_variant(exp_name, user_id)
        if variant:
            self.ab_manager.record_conversion(exp_name, variant.name)
    
    def get_results(self, prompt_name: str) -> Optional[Dict]:
        """Get A/B test results for prompt."""
        return self.ab_manager.get_results(f"prompt_{prompt_name}")


if __name__ == "__main__":
    print("A/B Testing Framework Example\n")
    
    # Example 1: Basic A/B test
    print("1. Basic A/B Test")
    manager = ABTestManager()
    
    exp = manager.create_experiment(
        name="model_test",
        description="Test GPT-4 vs Claude",
        variants=[
            Variant("gpt4", 0.5, {"model": "gpt-4"}),
            Variant("claude", 0.5, {"model": "claude-3"})
        ]
    )
    
    # Simulate usage
    for i in range(200):
        user_id = f"user_{i}"
        variant = manager.get_variant("model_test", user_id)
        
        manager.record_impression("model_test", variant.name)
        
        # Simulate conversion
        if random.random() < 0.3:
            manager.record_conversion("model_test", variant.name)
    
    results = manager.get_results("model_test")
    print(f"   Results: {results['variants'][0]['conversion_rate']:.1%} vs {results['variants'][1]['conversion_rate']:.1%}")
    
    # Example 2: Prompt versioning
    print("\n2. Prompt Version Test")
    prompt_mgr = PromptVersionManager()
    
    prompt_mgr.register("greeting", "v1", "Hello, how can I help?")
    prompt_mgr.register("greeting", "v2", "Hi there! What can I do for you today?")
    
    prompt_mgr.create_test(
        "greeting",
        versions=["v1", "v2"],
        weights=[0.5, 0.5]
    )
    
    # Use it
    for i in range(100):
        user_id = f"user_{i}"
        prompt = prompt_mgr.get_prompt("greeting", user_id)
        
        # Simulate success
        if random.random() < 0.4:
            prompt_mgr.record_success("greeting", user_id)
    
    results = prompt_mgr.get_results("greeting")
    print(f"   Prompt test results: {len(results['variants'])} variants tested")
    
    print("\n[OK] A/B testing framework working")
