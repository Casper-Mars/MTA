from typing import Optional, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from framework.core.worker import Worker

document_format = """Here is the template of usecase document:
# 用例

## 1、<用例名称>

<用例的功能或目的简要描述>

### 1.1、用例描述

<用例执行流程详细描述，包括用户输入、系统输出和中间步骤>

### 1.2、前置条件

<执行用例之前必须满足的条件>

### 1.3、基本流程

<用例的正常执行流程>

### 1.4、其他边界流程

<用例的其他可能执行流程>

### 1.5、异常流程

<用例的异常执行流程>

### 1.6、用例图

```mermaid
<content>
```

For example:
# 用例

## 1、用户登录

### 1.1、用例描述

1. 用户输入用户名和密码。
2. 系统验证用户名和密码是否正确。
3. 如果验证成功，系统将用户登录到系统。
4. 如果验证失败，系统会提示用户登录失败。

### 1.2、前置条件

* 用户未登录系统。

### 1.3、基本流程

1. 用户访问登录页面。
2. 用户输入用户名和密码。
3. 系统验证用户名和密码是否正确。
4. 如果验证成功，系统将用户登录到系统。
5. 如果验证失败，系统会提示用户登录失败。

### 1.4、其他边界流程

* 用户输入的用户名或密码为空。
* 用户输入的用户名或密码不正确。

### 1.5、异常流程

* 系统在验证用户输入信息时遇到错误。

### 1.6、用例图

```mermaid
sequenceDiagram
    participant 运营人员
    participant 系统

    运营人员->>系统: 选择公会并添加大神号
    系统->>系统: 记录托管配置
    系统->>运营人员: 确认托管配置
```

## 2、用户注册

### 2.1、用例描述
...
"""

sys_temp = """你是一个经验丰富的系统架构师，你特别擅长用例分析。
你的日常工作是生成高质量的用例，以及维护用例文档。

你需要遵守如下规则：
- 把需求拆分成原子的、简单的用例。
- 用例的描述必须清晰明了。
- 用例的流程描述必须包括用户、客户端和服务端的步骤。
"""


@tool
def get_document_format() -> str:
    """
    get the usecase document format. Usually, it should get the format before generating the document.
    :return:
    """
    return document_format


@tool
def save_to_file(content: str) -> str:
    """
    save the usecase document to file
    :param content: usecase document content
    :return:
    """
    with open("usecase.md", "w") as f:
        f.write(content)
    return "The usecase document has been saved to usecase.md"


class Usecase(Worker):

    def __init__(self, llm: BaseChatModel):
        tools = [get_document_format]
        self.llm = create_react_agent(model=llm, tools=tools, messages_modifier=SystemMessage(content=sys_temp))

    def get_executor_name(self) -> str:
        return "用例设计专家"

    def get_ability(self) -> str:
        return "擅长处理用例相关的任务，生成高质量的用例文档"

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        msgs = list(input['messages'])

        resp = self.llm.invoke({
            "messages": msgs,
        }, config)

        return {
            "messages": [AIMessage(content=resp['messages'][-1].content, name="Usecase_Agent")]
        }
