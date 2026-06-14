from abc import ABC, abstractmethod

class IFase0WarmUp(ABC):
    @abstractmethod
    def calentar(self):
        """
        Fase 0: Ejecución en vacío para encender motores (CUDA/Numba).
        Para CPU secuencial, solo debe tener un 'pass'.
        """
        pass

class IFase1TransferenciaIn(ABC):
    @abstractmethod
    def host_to_device(self, lote_host):
        """
        Fase 1: Mueve el lote de la memoria RAM principal (Host) al acelerador (Device).
        Debe retornar el puntero/tensor en el dispositivo.
        """
        pass

class IFase2Computo(ABC):
    @abstractmethod
    def procesarComputo1(self, lote_device):
        """
        Fase 2: Ejecuta el núcleo matemático (kernel) sobre los datos.
        Debe retornar el lote procesado (aún en el dispositivo y en float32).
        """
        pass

class IFase3Computo(ABC):
    @abstractmethod
    def procesarComputo2(self, lote_device):
        """
        Fase 3: Ejecuta el núcleo matemático (kernel) sobre los datos.
        Debe retornar el lote procesado (aún en el dispositivo y en float32).
        """
        pass

class IFase4TransferenciaOut(ABC):
    @abstractmethod
    def device_to_host(self, lote_procesado_device):
        """
        Fase 3: Trae los datos calculados de vuelta a la memoria RAM.
        Debe retornar un array NumPy estándar (float32).
        """
        pass

class IFase5Auxiliar(ABC):
    @abstractmethod
    def auxiliar(self, lote_procesado_device):
        """
        Fase 4: Auxiliar si es requerido
        """
        pass