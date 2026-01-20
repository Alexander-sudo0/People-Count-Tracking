import insightface
from .interfaces import IFaceRecognizer

class InsightFaceRecognizer(IFaceRecognizer):
    def __init__(self, use_gpu=False):
        # Try GPU first, fallback to CPU if CUDA is not available
        if use_gpu:
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"[INFO] Attempting to use GPU (CUDA) with CPU fallback")
        else:
            self.providers = ['CPUExecutionProvider']
            print(f"[INFO] Using CPU execution")
        
        try:
            # Load Buffalo_S (Lightweight & Fast)
            self.app = insightface.app.FaceAnalysis(name='buffalo_s', providers=self.providers)
            self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
            
            # Check which provider is actually being used
            if hasattr(self.app.models, 'values'):
                for model in self.app.models.values():
                    if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
                        actual_providers = model.session.get_providers()
                        print(f"[INFO] Active providers: {actual_providers}")
                        break
        except Exception as e:
            print(f"[WARNING] GPU initialization failed: {e}")
            print(f"[INFO] Falling back to CPU execution")
            self.providers = ['CPUExecutionProvider']
            self.app = insightface.app.FaceAnalysis(name='buffalo_s', providers=self.providers)
            self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def get_faces(self, frame):
        """
        Returns a list of face objects. 
        Each object contains .embedding (512-d vector) and .bbox
        """
        return self.app.get(frame)