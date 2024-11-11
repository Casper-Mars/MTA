from typing import Optional, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from framework.utils import format_messages


class FinalSummary(Runnable):

    def __init__(self, llm: BaseChatModel):
        p = PromptTemplate.from_template("""你是一名高级助手。
如下是用户和AI的对话历史：
```
{ori_msgs}
```

如下是你的团队成员之间的对话内容：
```
{msgs}
```

用户说的话：
```
{user_input}
```

现在，请你结合你的团队成员之间的对话内容，用户和AI的对话历史作为补充信息，总结出最终回复用户说的话的回答。
        """)
        self.llm = p | llm

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        user_input = input.get("user_input", "")
        ori_msgs = format_messages(input.get("ori_messages", []))
        msgs = format_messages(input.get("messages", []))
        resp = self.llm.invoke({
            "user_input": user_input,
            "ori_msgs": ori_msgs,
            "msgs": msgs,
        }, config)
        return {
            "final": resp
        }
