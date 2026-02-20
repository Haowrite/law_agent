

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

AGENT_EXECUTOR = ThreadPoolExecutor(max_workers= 16)

PROCESS_POOL = ProcessPoolExecutor(max_workers=2)