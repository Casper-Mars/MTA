from typing import Optional, Any

import PyPDF2
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_core.tools import tool
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from framework.core.worker import Worker


@tool
def load_from_file(file_path: str) -> str:
    """
    Load content from file.
    :param file_path:
    :return: file content
    """
    try:
        # 如果是 PDF 文件，使用 PyPDF2 库读取
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages])
        else:
            with open(file_path, "r") as f:
                return f.read()
    except FileNotFoundError:
        # 判断文件是否存在
        return f"文件不存在: {file_path}"


class FileLoader(Worker):

    def __init__(self, llm: BaseChatModel):
        tools = [load_from_file]
        self.tool_executor = ToolExecutor(tools)
        self.llm = llm.bind_tools(tools)

    def get_executor_name(self) -> str:
        return "File Loader"

    def get_ability(self) -> str:
        return "加载文件内容。每次只能加载一个文件。"

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        msgs = input['messages']
        resp = self.llm.invoke(msgs[-1].content, config)
        if len(resp.tool_calls) == 0:
            return {
                "messages": [resp]
            }
        tool_call = resp.tool_calls[0]
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        # We call the tool_executor and get back a response
        response = self.tool_executor.invoke(action)

        file_path = tool_call["args"]['file_path']

        return {
            "messages": [AIMessage(name="File_Loader",
                                   content=f"Here is the content from file({file_path}):\n\n{response}")],
        }
