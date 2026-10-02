"""A reusable test/programmatic client: one subprocess for all requests."""
import json
import subprocess


class JsonlClient:
    def __init__(self, command, stderr=None):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, encoding='utf-8', bufsize=1)
        self.ready = self._read()
        if self.ready.get('status') != 'READY':
            self.close()
            raise RuntimeError(self.ready)
    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Bridge ended without a JSONL response')
        return json.loads(line)
    def request(self, tool, **arguments):
        self.process.stdin.write(json.dumps({'tool': tool, 'arguments': arguments})+'\n')
        self.process.stdin.flush()
        return self._read()
    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.wait(timeout=30)
        if self.process.stdout and not self.process.stdout.closed:
            self.process.stdout.close()
