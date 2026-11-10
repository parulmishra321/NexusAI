from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract Base Class for all Agents.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, query: str) -> str:
        """
        Execute the agent's logic.
        :param query: The user's input.
        :return: The agent's response as a string.
        """
        pass
