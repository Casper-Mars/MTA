from abc import ABC, abstractmethod

from langchain_core.runnables import Runnable


class Worker(Runnable, ABC):

    @abstractmethod
    def get_executor_name(self) -> str:
        pass

    @abstractmethod
    def get_ability(self) -> str:
        pass
