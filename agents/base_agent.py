from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
