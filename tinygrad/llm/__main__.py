import sys
from tinygrad.llm.cli import main

if __name__ == "__main__":
  try: main()
  except KeyboardInterrupt: sys.exit(1)
