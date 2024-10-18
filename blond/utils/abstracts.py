from abc import ABC, abstractmethod


class TrackableBaseClass(ABC):

    @abstractmethod
    def track(self) -> None:
        pass


class CpuGpuTransferable(ABC):
    @abstractmethod
    def to_cpu(self, recursive=True) -> None:
        pass

    @abstractmethod
    def to_gpu(self, recursive=True) -> None:
        pass


class CpuGpuTrackable(TrackableBaseClass, CpuGpuTransferable):
    pass


