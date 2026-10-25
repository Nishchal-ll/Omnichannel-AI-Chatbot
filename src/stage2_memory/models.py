"""
src/stage2_memory/models.py
───────────────────────────
Strongly-typed data contracts for Stage 2 (Memory MESH) and Stage 3 (Brain).
Decouples memory orchestration from cognitive prompt execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class ChatMessage:
    role: str                       # 'user' | 'assistant' | 'system'
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class UserMemoryItem:
    id: str
    user_id: str
    memory_text: str
    category: str = "general"       # 'identity' | 'preference' | 'fact' | 'general'
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_text": self.memory_text,
            "category": self.category,
            "created_at": self.created_at,
        }


@dataclass
class ContextPackage:
    user_id: str
    original_query: str
    standalone_query: str
    chat_history: List[ChatMessage]
    user_memories: List[UserMemoryItem]
    retrieved_knowledge: str
    retrieved_chunks: List[Dict[str, Any]]
    intent_type: str = "information_query"      # 'fast_anchor' | 'memory_command' | 'information_query' | 'out_of_domain'
    fast_reply: Optional[str] = None

