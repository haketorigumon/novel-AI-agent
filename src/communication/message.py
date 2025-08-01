"""Message class for agent communication"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

class Message:
    """
    Message class for communication between agents
    
    Attributes:
        message_id: Unique identifier for the message
        sender_id: ID of the sending agent
        receiver_id: ID of the receiving agent
        content: Message content
        type: Message type (e.g., text, command, query, response)
        timestamp: Time the message was created
        metadata: Additional message metadata
        reference_message_id: ID of a message this is responding to
    """
    
    def __init__(
        self,
        sender_id: str,
        receiver_id: str,
        content: Any,
        type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        reference_message_id: Optional[str] = None,
        message_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.message_id = message_id or str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.content = content
        self.type = type
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.reference_message_id = reference_message_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "reference_message_id": self.reference_message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            content=data["content"],
            type=data["type"],
            metadata=data["metadata"],
            reference_message_id=data.get("reference_message_id"),
            timestamp=timestamp
        )
    
    def __str__(self) -> str:
        """String representation of the message"""
        return f"Message(id={self.message_id}, from={self.sender_id}, to={self.receiver_id}, type={self.type})"


class Conversation:
    """
    Represents a conversation between agents
    
    Attributes:
        conversation_id: Unique identifier for the conversation
        participants: List of agent IDs participating in the conversation
        messages: List of messages in the conversation
        metadata: Additional conversation metadata
    """
    
    def __init__(
        self,
        participants: List[str],
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.participants = participants
        self.messages: List[Message] = []
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
    
    def add_message(self, message: Message):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from the conversation, optionally limited to the most recent ones"""
        if limit is None:
            return self.messages
        return self.messages[-limit:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary"""
        return {
            "conversation_id": self.conversation_id,
            "participants": self.participants,
            "messages": [message.to_dict() for message in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create conversation from dictionary"""
        conversation = cls(
            conversation_id=data["conversation_id"],
            participants=data["participants"],
            metadata=data["metadata"]
        )
        
        conversation.messages = [Message.from_dict(msg_data) for msg_data in data["messages"]]
        conversation.created_at = datetime.fromisoformat(data["created_at"])
        conversation.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return conversation