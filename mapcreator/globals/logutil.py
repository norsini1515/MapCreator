import re
import sys
import os
from pathlib import Path
from datetime import datetime

from mapcreator.globals import directories

################################################################################################
ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
TIMESTAMP_PREFIX = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")
RESET = '\x1b[0m'
BOLD = "\033[1m"

def _enable_windows_ansi():
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            kernel32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        # best effort; if it fails you still get plain text
        pass

_enable_windows_ansi()


class Logger:
    _instance = None  # Keep a reference to active logger

    def __init__(self, logfile_path=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logfile_path = logfile_path or directories.LOGS_DIR / f"process_{timestamp}.log"
        self.logfile_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._prev_stdout = sys.stdout
        self._prev_stderr = sys.stderr

        self.logfile = open(self.logfile_path, "a", encoding="utf-8", buffering=1)  # line-buffered
        sys.stdout = self
        sys.stderr = self
    # context manager
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()

    def write(self, message):
        from datetime import datetime
        # 1) coerce to str
        if message is None:
            return
        if isinstance(message, (bytes, bytearray)):
            message = message.decode("utf-8", errors="replace")
        elif not isinstance(message, str):
            message = str(message)

        # 2) write per-line (keep newlines), timestamp non-empty lines
        for part in message.splitlines(keepends=True):
            if part.strip() and not TIMESTAMP_PREFIX.match(part):
                part = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {part}"

            # 3) mirror to console
            try:
                self._prev_stdout.write(part)
            except Exception:
                # never crash the excepthook; best-effort fallback
                pass

            # 4) write to file (strip ANSI)
            try:
                self.logfile.write(ANSI_ESCAPE.sub("", part))
            except Exception as e:
                print(f"Failed to write to log file: {self.logfile_path} {e}")
                pass

    def flush(self):
        # Needed for Python's `print()` buffering behavior
        self._prev_stdout.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()
        sys.stdout = self._prev_stdout
        sys.stderr = self._prev_stderr
        Logger._instance = None
    
    @classmethod
    def setup(cls, logfile_path=None):
        if isinstance(sys.stdout, cls):   # already active
            return cls._instance
        if cls._instance is None:
            cls._instance = cls(logfile_path)
        return cls._instance

    @classmethod
    def teardown(cls):
        if cls._instance:
            cls._instance.close()

def rgb_prefix(r: int, g: int, b: int) -> str:
    """Start an RGB color (leave it open)."""
    return f"\033[38;2;{r};{g};{b}m"

def color_text(text: str, r: int, g: int, b: int) -> str:
    """Color a specific chunk of text, then reset."""
    return f"{rgb_prefix(r, g, b)}{text}{RESET}"

def _log(level, color_code, msg):
    """
    color_code can be:
      - a colorama constant (e.g., Fore.CYAN)
      - an (r, g, b) tuple for 24-bit color
    """
    if isinstance(color_code, tuple):
        start = rgb_prefix(*color_code)
        prefix = f"{start}{BOLD}[{level}]{RESET} "
    else:
        prefix = f"{color_code}{BOLD}[{level}]{RESET} "
    print(prefix + msg)

def process_step(msg):  _log("PROCESS", (171, 52, 235), msg)
def info(msg): _log("INFO", (0, 255, 255), msg)
def warn(msg): _log("WARNING", (255, 255, 0), msg)
def error(msg): _log("ERROR", (255, 0, 0), msg)
def success(msg): _log("SUCCESS", (0, 255, 0), msg)
def setting_config(msg): _log("SETTING", (250, 197, 97), msg)

if __name__ == "__main__":
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d %H%M")

    log_path = directories.LOGS_DIR / f"process_{timestamp}.log"
    # Activate logging
    logger = Logger(log_path)

    # Example prints (go to both console and log file)
    print("This goes to both console and the log file.")
    print("Useful for debugging and record-keeping.")
    info("This is an info message.")
    warn("This is a warning message.")
    error("This is an error message.")
    success("This is a success message.")
    process_step("This is a process step message.")

    logger.teardown()
