import argparse
import cv2
from config import Config
from core.recognizer import InsightFaceRecognizer
from core.storage import JSONStorage
from core.counter import UniquePeopleCounter
from utils.stream_loader import RTSPStreamLoader

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="RTSP People Counter")
    parser.add_argument("--source", type=str, default="0", help="RTSP URL or Camera Index")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    args = parser.parse_args()

    # Handle numeric camera index vs RTSP string
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"Starting Application...")
    print(f"Source: {source}")
    print(f"Mode: {'GPU' if args.gpu else 'CPU'}")

    # 2. Setup Dependencies (Dependency Injection)
    Config.setup_dirs()
    storage = JSONStorage()
    recognizer = InsightFaceRecognizer(use_gpu=args.gpu)
    counter = UniquePeopleCounter(recognizer, storage)
    
    # 3. Start Stream
    stream = RTSPStreamLoader(src=source, process_fps=5)
    
    try:
        while True:
            frame = stream.get_frame()
            
            if frame is not None:
                # Run Logic
                results, unique_count = counter.process_frame(frame)
                
                # Visualize
                cv2.putText(frame, f"Unique (1hr): {unique_count}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                for res in results:
                    x1, y1, x2, y2 = res['bbox'].astype(int)
                    color = (0, 255, 0) if res['is_new'] else (255, 0, 0) # Green=New, Blue=Return
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, str(res['id']), (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imshow("People Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()