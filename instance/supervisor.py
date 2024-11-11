from typing import List, Optional, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from pydantic import BaseModel, Field

from framework.core.worker import Worker
from instance.utils import format_members, format_messages

supervisor_sys_temp = """当前的上下文信息如下：
`````````
{msgs}
`````````

用户的意图分析：
```
{intention}
```

你是一个团队的领导，你的团队成员有：{members}。
"""


class NextWorker(BaseModel):
    """next worker to assign the job"""
    next_worker: str = Field(description="next worker to assign the task")
    task: str = Field(description="task to be done")
    is_finish: bool = Field(description="is finished?")
    reason: str = Field(description="the reason that choose this worker")


class Supervisor(Runnable):

    def __init__(self, llm: BaseChatModel, members: List[Worker]):
        # 定义团队成员选择的逻辑
        self.members_str = format_members(members)
        p = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(supervisor_sys_temp),
            MessagesPlaceholder("messages"),
            SystemMessagePromptTemplate.from_template("""当前的对话是否符合用户的意图？如果没有符合，请你继续生成下一个具体的任务并分配给合适的成员。
注意：你生成的任务描述必须包含具体的信息，目的为了让团队成员能够完成该任务。
                            """)
        ])
        self.choose_next_worker = p | llm.with_structured_output(NextWorker)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        user_input = input["user_input"]
        msgs = input.get("messages", [])
        is_finish = input.get("is_finish", False)
        if input["cur_step"] >= input["max_step"]:
            return {
                "next_worker": "Task Completed",
                "messages": [AIMessage(content="任务轮数超出限制，任务结束。")]
            }
        if is_finish:
            return {
                "next_worker": "Task Completed",
            }
        resp = self.choose_next_worker.invoke({
            "members": self.members_str,
            "msgs": format_messages(input.get("ori_messages", [])),
            "intention": input.get("intention", ""),
            "messages": msgs,
            "user_input": user_input,
        }, config)
        if resp.is_finish:
            return {
                "next_worker": "Task Completed",
            }
        return {
            "next_worker": resp.next_worker,
            "messages": [HumanMessage(content=resp.task, name="User")],
            "cur_step": input["cur_step"] + 1,
        }
