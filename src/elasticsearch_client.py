"""Thin Elasticsearch client factory and connection helpers."""

from __future__ import annotations


def get_client():
    from elasticsearch import Elasticsearch

    from .config import settings

    kwargs: dict[str, object] = {"hosts": [settings.elasticsearch_url]}

    if settings.elasticsearch_api_key:
        kwargs["api_key"] = settings.elasticsearch_api_key
    elif settings.elasticsearch_username and settings.elasticsearch_password:
        kwargs["basic_auth"] = (
            settings.elasticsearch_username,
            settings.elasticsearch_password,
        )

    if settings.elasticsearch_url.lower().startswith("https://"):
        kwargs["verify_certs"] = settings.elasticsearch_verify_certs
        if settings.elasticsearch_ca_cert:
            kwargs["ca_certs"] = settings.elasticsearch_ca_cert

    return Elasticsearch(**kwargs)


def check_connection() -> tuple[bool, str]:
    """Return a friendly health result without hiding client errors."""
    try:
        client = get_client()
        try:
            info = client.info()
            version = info.get("version", {}).get("number", "unknown")
            return True, f"Connected to Elasticsearch {version}."
        finally:
            client.close()
    except Exception as exc:  # client exceptions differ across major versions
        return False, (
            "Elasticsearch is unreachable. Start the local service and verify "
            f"ELASTICSEARCH_URL. Error type: {type(exc).__name__}."
        )
