from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from framework.core.main import Core
from instance.final_summarizer import FinalSummary
from instance.intention import IntentionSummarizer
from instance.supervisor import Supervisor
from instance.workers.file_loader import FileLoader
from instance.workers.general_agent import GeneralAgent
from instance.workers.usecase import Usecase


def build(llm: BaseChatModel) -> Runnable:
    executors = [
        GeneralAgent(llm),
        FileLoader(llm),
        Usecase(llm),
    ]
    supervisor = Supervisor(llm, executors)
    intention_summarizer = IntentionSummarizer(llm)
    final_summary = FinalSummary(llm)
    core = Core(supervisor, executors, intention_summarizer=intention_summarizer, final_summarizer=final_summary)
    return core
