import operator
from typing import List, TypedDict, Annotated, Optional, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from framework.core.worker import Worker
from framework.utils import format_messages


class State(TypedDict):
    ori_messages: Annotated[List[BaseMessage], operator.add]
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str
    next_worker: str
    cur_step: int
    max_step: int
    is_finish: bool
    memory: str
    intention: str


def enter_graph(state: State):
    msgs = list(state["messages"])
    user_input = msgs[-1].content
    max_step = state.get("max_step", 10)
    new_state = {**state, "user_input": user_input, "messages": [], "max_step": max_step, "cur_step": 0,
                 "ori_messages": msgs}
    return new_state


class Core(Runnable):

    def __init__(self,
                 supervisor: Runnable,
                 executors: List[Worker],
                 intention_summarizer: Runnable = None,
                 final_summarizer: Runnable = None
                 ):
        agent = StateGraph(State)

        # 建立关系
        agent.add_node("Supervisor", supervisor)
        agent.set_entry_point("Supervisor")
        if intention_summarizer:
            agent.add_node("IntentionSummarizer", intention_summarizer)
            agent.add_edge("IntentionSummarizer", "Supervisor")
            agent.set_entry_point("IntentionSummarizer")

        node_map = {
            "Task Completed": END
        }

        if final_summarizer:
            agent.add_node("FinalSummary", final_summarizer)
            node_map["Task Completed"] = "FinalSummary"
            node_map["FinalSummary"] = END

        for executor in executors:
            agent.add_node(executor.get_executor_name(), executor)
            agent.add_edge(executor.get_executor_name(), "Supervisor")
            node_map[executor.get_executor_name()] = executor.get_executor_name()
        agent.add_conditional_edges(
            "Supervisor",
            lambda state: state['next_worker'],
            node_map
        )

        self.invoker = enter_graph | agent.compile()

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        return self.invoker.invoke(input, config, **kwargs)
