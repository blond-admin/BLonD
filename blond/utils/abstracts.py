from abc import ABC, abstractmethod


class TrackableBaseClass(ABC):

    @abstractmethod
    def track(self) -> None:
        pass
