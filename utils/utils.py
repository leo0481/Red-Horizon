from enum import Enum


# ============================================================================
# VULNERABILITY TOPICS INCLUDING NEW CATEGORIES
# ============================================================================
class VulnerabilityTopic(Enum):
    """Extended vulnerability topics including labor rights."""

    # Original topics
    JAILBREAK = "jailbreak"
    INAPPROPRIATE_TOOL_USE = "inappropriate_tool_use"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    RAG_LEAK = "rag_leak"
    PROMPT_INJECTION = "prompt_injection"
