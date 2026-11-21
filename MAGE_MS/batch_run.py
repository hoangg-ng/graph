"""Run several train.py jobs inside ONE process (one thread each), so a GPU shows a single task (2026-11-18).
Each job's output goes to its own log through a thread-local stdout. Kernels from all threads share the GPU.
Note: threads share torch / numpy global RNG, so runs are not bit-reproducible against separate-process runs.
Usage: python batch_run.py jobs.json      (jobs.json = [[log_path, [train.py args ...]], ...])
"""
import json
import sys
import threading
import traceback

import train


class ThreadStdout:
    def __init__(self, default):
        self.default, self.local = default, threading.local()

    def write(self, s):
        return getattr(self.local, 'f', self.default).write(s)

    def flush(self):
        getattr(self.local, 'f', self.default).flush()


out = ThreadStdout(sys.stdout)
sys.stdout = sys.stderr = out


def work(log, argv):
    with open(log, 'w') as f:
        out.local.f = f
        try:
            train.main(argv)
        except Exception:
            traceback.print_exc(file=f)
        f.flush()


jobs = json.load(open(sys.argv[1]))
ts = [threading.Thread(target=work, args=(log, argv)) for log, argv in jobs]
for t in ts:
    t.start()
for t in ts:
    t.join()
out.default.write('BATCH DONE\n')
