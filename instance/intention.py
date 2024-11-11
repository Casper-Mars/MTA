from typing import Optional, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from framework.utils import format_messages

intention_sys_temp = """当前的上下文信息如下：
`````````
{msgs_his}
`````````

用户最新的消息是：
```
{user_input}
```

请你根据上下文信息，总结用户的意图。
"""


class IntentionSummarizer(Runnable):

    def __init__(self, llm: BaseChatModel):
        p = PromptTemplate.from_template(intention_sys_temp)
        self.summarize_intention = p | llm

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        msgs = input.get("ori_messages", [])
        resp = self.summarize_intention.invoke({
            "msgs_his": format_messages(msgs),
            "user_input": input.get("user_input")
        }, config)

        return {
            "intention": resp.content
        }
