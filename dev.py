#!/usr/bin/env python3
"""Development server with auto-reload on file changes."""
import sys
import os
import time
import subprocess
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ReloadHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.start_app()
    
    def start_app(self):
        with self.lock:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                time.sleep(1)
            
            print("\n🔄 Restarting app...")
            self.process = subprocess.Popen(
                [sys.executable, "run.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )
            # Wait for startup
            for _ in range(10):
                time.sleep(1)
                if self.process.poll() is not None:
                    output = self.process.stdout.read()
                    print("✗ Failed to start:")
                    print(output if output else "No output")
                    return
                # Check if port is listening
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', 7860))
                sock.close()
                if result == 0:
                    print(f"✓ App started (PID: {self.process.pid})")
                    return
    
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f"\n📝 File changed: {os.path.basename(event.src_path)}")
            self.start_app()

if __name__ == "__main__":
    print("🚀 Starting development server with auto-reload...")
    print("Watching for .py file changes in current directory...\n")
    
    event_handler = ReloadHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹ Shutting down...")
        if event_handler.process:
            event_handler.process.terminate()
        observer.stop()
    observer.join()
