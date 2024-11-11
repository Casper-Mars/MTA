from typing import List

from langchain_core.messages import BaseMessage, HumanMessage

from framework.core.worker import Worker


def format_members(members: List[Worker]) -> str:
    members_str = "[\n"
    for item in members:
        members_str += "- name: " + item.get_executor_name() + "\n  description: " + item.get_ability() + "\n"
    members_str += "]"
    return members_str


def format_messages(messages: List[BaseMessage]) -> str:
    res = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if msg.name:
                res += f"{msg.name}: {msg.content}\n"
            else:
                res += f"User: {msg.content}\n"
        else:
            res += f"AI: {msg.content}\n"
    return res
