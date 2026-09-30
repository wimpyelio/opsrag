from src import retrieval


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def test_bm25_builds_multi_match_query():
    client = FakeClient({"hits": {"hits": [{"_score": 2.0, "_source": {"chunk_id": "c1", "text": "hello"}}]}})
    results = retrieval.search_bm25("hello", top_k=5, client=client)
    assert results[0]["chunk_id"] == "c1"
    query = client.calls[0]["query"]["multi_match"]
    assert query["query"] == "hello"
    assert "text^3" in query["fields"]


def test_vector_builds_knn_query(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_query", lambda _: [0.1, 0.2, 0.3])
    client = FakeClient({"hits": {"hits": []}})
    retrieval.search_vector("hello", top_k=5, client=client)
    knn = client.calls[0]["knn"]
    assert knn["field"] == "embedding"
    assert knn["k"] == 5
    assert knn["query_vector"] == [0.1, 0.2, 0.3]


def test_hybrid_uses_rrf_with_standard_and_knn(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_query", lambda _: [0.1, 0.2, 0.3])
    client = FakeClient({"hits": {"hits": []}})
    retrieval.search_hybrid("hello", top_k=5, client=client)
    rrf = client.calls[0]["retriever"]["rrf"]
    retrievers = rrf["retrievers"]
    assert "standard" in retrievers[0]
    assert "knn" in retrievers[1]
    assert rrf["rank_constant"] == 60


class RaisingClient(FakeClient):
    def __init__(self, payload):
        super().__init__(payload)
        self.closed = False

    def search(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("synthetic failure")

    def close(self):
        self.closed = True


def test_owned_client_is_closed_when_search_fails(monkeypatch):
    client = RaisingClient({})
    monkeypatch.setattr(retrieval, "get_client", lambda: client)
    try:
        retrieval.search_bm25("hello")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected search failure")

    assert client.closed is True
