class _FakeCollection:
    def count(self): return 1
    def add(self, *a, **k): pass
    def query(self, *a, **k): return {"documents": [[]]}
class PersistentClient:
    def __init__(self, *a, **k): pass
    def get_or_create_collection(self, *a, **k): return _FakeCollection()
