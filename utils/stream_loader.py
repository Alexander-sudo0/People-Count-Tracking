import cv2
import threading
import time

class RTSPStreamLoader:
    def __init__(self, src=0, process_fps=5):
        self.src = src
        self.process_fps = process_fps
        self.interval = 1.0 / process_fps
        self.current_frame = None
        self.last_access_time = 0
        self.status = False
        self.capture = None
        self.error_count = 0
        self.max_errors = 50
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Initialize capture with codec preferences
        self._init_capture()
        
        # Start background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _init_capture(self):
        """Initialize video capture with error handling and codec optimization"""
        try:
            self.capture = cv2.VideoCapture(self.src)
            
            # Set codec preferences for RTSP/network streams
            # Use hardware acceleration if available
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            # Try H.264 codec first (more stable than HEVC)
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            
            # For RTSP, disable certain decoders to avoid HEVC issues
            if isinstance(self.src, str) and ('rtsp://' in self.src.lower() or 'http://' in self.src.lower()):
                # Force software decoding for better compatibility
                self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            # Read initial frame to verify connection
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame
                self.status = True
                self.error_count = 0
                self.reconnect_attempts = 0
                print(f"[RTSPStreamLoader] Connected to {self.src}")
                return True
            else:
                print(f"[RTSPStreamLoader] Failed to read initial frame from {self.src}")
                self.status = False
                return False
        except Exception as e:
            print(f"[RTSPStreamLoader] Error initializing capture: {e}")
            self.status = False
            return False

    def update(self):
        """Background thread for continuous frame reading with error recovery"""
        while True:
            try:
                if self.capture is None or not self.capture.isOpened():
                    self.status = False
                    time.sleep(1)
                    continue
                
                ret, frame = self.capture.read()
                
                if ret and frame is not None:
                    self.current_frame = frame
                    self.status = True
                    self.error_count = 0  # Reset error count on success
                else:
                    self.error_count += 1
                    
                    # If too many errors, attempt reconnection
                    if self.error_count > self.max_errors:
                        print(f"[RTSPStreamLoader] Too many read errors ({self.error_count}), attempting reconnection...")
                        self._reconnect()
                        self.error_count = 0
                    
                    self.status = False
                    time.sleep(0.05)  # Small delay before retry
            except Exception as e:
                print(f"[RTSPStreamLoader] Exception in update loop: {e}")
                self.error_count += 1
                self.status = False
                if self.error_count > self.max_errors:
                    self._reconnect()
                    self.error_count = 0
                time.sleep(0.1)

    def _reconnect(self):
        """Attempt to reconnect to the stream"""
        self.reconnect_attempts += 1
        if self.reconnect_attempts > self.max_reconnect_attempts:
            print(f"[RTSPStreamLoader] Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return
        
        print(f"[RTSPStreamLoader] Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        try:
            if self.capture is not None:
                self.capture.release()
            
            time.sleep(1)  # Wait before reconnecting
            self._init_capture()
        except Exception as e:
            print(f"[RTSPStreamLoader] Reconnection failed: {e}")

    def get_frame(self):
        """
        Returns frame only if enough time has passed (throttling to process_fps)
        """
        now = time.time()
        if (now - self.last_access_time) > self.interval:
            self.last_access_time = now
            return self.current_frame
        return None
    
    def stop(self):
        """Clean shutdown"""
        if self.capture is not None:
            self.capture.release()
        self.status = False