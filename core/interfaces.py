from abc import ABC, abstractmethod

class IFaceRecognizer(ABC):
    @abstractmethod
    def get_faces(self, frame):
        pass

class IStorage(ABC):
    @abstractmethod
    def save_face(self, face_data):
        pass
    
    @abstractmethod
    def load_recent_faces(self, time_window):
        pass