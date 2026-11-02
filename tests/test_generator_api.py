import httpx
import pytest

from hybridrag.generator import call_llm


@pytest.mark.parametrize("provider", ["gemini", "openai", "groq", "ollama"])
def test_provider_request_and_response(provider, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    client_class = httpx.Client

    def respond(request):
        import json

        payload = json.loads(request.content)
        if provider == "gemini":
            assert request.url.path.endswith("/models/test-model:generateContent")
            assert request.headers["x-goog-api-key"] == "test-key"
            assert "systemInstruction" in payload
            assert "CONTEXT" in payload["contents"][0]["parts"][0]["text"]
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "Đáp án [S1]"}]}}]})
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == "Bearer test-key"
        assert payload["model"] == "test-model"
        assert payload["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"choices": [{"message": {"content": "Đáp án [S1]"}}]})

    monkeypatch.setattr("hybridrag.generator.httpx.Client", lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs))
    assert call_llm("question", "context") == "Đáp án [S1]"


def test_missing_credentials_fail_before_network(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        call_llm("question", "context")
