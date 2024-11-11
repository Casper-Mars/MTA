from typing import Optional, Any, List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.runnables.utils import Input, Output

from framework.core.worker import Worker
from instance.utils.utils import format_messages


class GeneralAgent(Worker):
    llm: Runnable

    def __init__(self, llm: Runnable):
        p = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""你是一个聊天机器人。
{memory}

如下是短期对话历史：
```
{msgs_his}
```
            """),
            MessagesPlaceholder("messages"),
        ])
        self.llm = p | llm
        return

    def get_executor_name(self) -> str:
        return "通用聊天机器人"

    def get_ability(self) -> str:
        return "一个通用的聊天机器人，用于处理其他团队成员无法处理的问题。"

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        msgs = list(input["messages"])
        user_input = input["user_input"]
        ori_msgs = input.get("ori_messages", [])
        memory = self.load_memory(ori_msgs)

        # 因为上层的Supervisor会替换用户的输入，所以这里就重新替换回来
        msgs[-1].content = user_input
        result = self.llm.invoke({
            "messages": msgs,
            "memory": memory,
            "msgs_his": format_messages(ori_msgs),
        }, config)
        result.name = "General_Agent"
        return {
            "messages": [result],
            "is_finish": True
        }

    def load_memory(self, msgs: List[BaseMessage], config: Optional[RunnableConfig] = None) -> str:
        return ""
