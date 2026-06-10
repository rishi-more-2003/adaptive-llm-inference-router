from src.cache.prefix_cache import PrefixCacheResult
from src.router.features import RequestFeatureExtractor, classify_request, estimate_kv_cache_mb


def test_prompt_token_estimate_is_positive():
    extractor = RequestFeatureExtractor()
    features = extractor.extract("hello world", max_new_tokens=16)

    assert features.estimated_prompt_tokens > 0
    assert features.prompt_chars == len("hello world")


def test_code_prompt_classified_as_code():
    request_type = classify_request("Write code for a binary search function.", 12, 4096)
    assert request_type == "code"


def test_long_context_classified_as_long_context():
    request_type = classify_request("many tokens", 5000, 4096)
    assert request_type == "long_context"


def test_rag_style_prompt_classified_as_rag():
    request_type = classify_request("Context: document text\nQuestion: summarize it", 20, 4096)
    assert request_type == "rag"


def test_prefix_cache_result_sets_feature_flags():
    extractor = RequestFeatureExtractor()
    prefix_result = PrefixCacheResult(
        hit=True,
        prefix_hash="abc",
        estimated_tokens_saved=10,
    )

    features = extractor.extract(
        "shared prefix prompt",
        max_new_tokens=32,
        latency_target_ms=1000,
        prefix_cache_result=prefix_result,
    )

    assert features.prefix_cache_hit is True
    assert features.has_shared_prefix is True


def test_kv_cache_estimate_increases_with_tokens():
    small = estimate_kv_cache_mb(128, 64)
    large = estimate_kv_cache_mb(2048, 256)

    assert large > small
