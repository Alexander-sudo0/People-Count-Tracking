people_counter_app/
├── main.py                 # Entry point (CLI argument parsing)
├── requirements.txt        # Dependencies
├── config.py               # Configuration constants
├── core/
│   ├── __init__.py
│   ├── interfaces.py       # Abstract Base Classes (DIP)
│   ├── recognizer.py       # InsightFace wrapper (SRP)
│   ├── counter.py          # Counting logic & deduplication (SRP)
│   └── storage.py          # Database/File handling for "probing"
└── utils/
    ├── __init__.py
    └── stream_loader.py    # Threaded RTSP handling