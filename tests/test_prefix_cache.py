from src.cache.prefix_cache import PrefixCacheSimulator


def test_first_request_is_miss():
    cache = PrefixCacheSimulator(prefix_chars=32)
    result = cache.check_and_update("hello world")

    assert result.hit is False
    assert cache.metrics()["prefix_cache_misses"] == 1


def test_repeated_prefix_is_hit():
    cache = PrefixCacheSimulator(prefix_chars=13)
    cache.check_and_update("shared prefix question one")
    result = cache.check_and_update("shared prefix question two")

    assert result.hit is True
    assert result.estimated_tokens_saved > 0


def test_whitespace_normalization_works():
    cache = PrefixCacheSimulator(prefix_chars=64)
    cache.check_and_update("same   prefix\nwith\tspacing")
    result = cache.check_and_update("same prefix with spacing")

    assert result.hit is True


def test_max_cache_size_is_respected():
    cache = PrefixCacheSimulator(prefix_chars=64, max_entries=2)
    cache.check_and_update("first prompt")
    cache.check_and_update("second prompt")
    cache.check_and_update("third prompt")

    assert cache.metrics()["entries"] == 2
    assert cache.check_and_update("first prompt").hit is False
