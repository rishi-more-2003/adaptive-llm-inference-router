from src.router.features import RequestFeatures
from src.router.policy import AdaptivePolicy, InferencePath, RouterConfig


def features(**overrides):
    values = {
        "prompt_chars": 120,
        "estimated_prompt_tokens": 64,
        "max_new_tokens": 128,
        "batch_size": 1,
        "latency_target_ms": None,
        "has_shared_prefix": False,
        "prefix_cache_hit": False,
        "estimated_kv_cache_mb": 16.0,
        "request_type": "chat",
        "cuda_available": False,
        "gpu_memory_allocated_mb": None,
        "gpu_memory_reserved_mb": None,
    }
    values.update(overrides)
    return RequestFeatures(**values)


def test_short_batch1_request_selects_int4_gemv():
    decision = AdaptivePolicy(RouterConfig()).decide(features())
    assert decision.path == InferencePath.INT4_GEMV


def test_repeated_prefix_selects_prefix_cache():
    decision = AdaptivePolicy(RouterConfig()).decide(features(prefix_cache_hit=True))
    assert decision.path == InferencePath.PREFIX_CACHE


def test_long_prompt_selects_long_context():
    decision = AdaptivePolicy(RouterConfig()).decide(features(estimated_prompt_tokens=5000))
    assert decision.path == InferencePath.LONG_CONTEXT


def test_tight_latency_selects_speculative_only_when_enabled():
    base_features = features(
        estimated_prompt_tokens=900,
        max_new_tokens=512,
        latency_target_ms=500,
    )
    disabled = AdaptivePolicy(RouterConfig(speculative_enabled=False, int4_gemv_enabled=False)).decide(
        base_features
    )
    enabled = AdaptivePolicy(RouterConfig(speculative_enabled=True, int4_gemv_enabled=False)).decide(
        base_features
    )

    assert disabled.path == InferencePath.BASELINE
    assert enabled.path == InferencePath.SPECULATIVE


def test_fallback_selects_baseline():
    decision = AdaptivePolicy(RouterConfig(int4_gemv_enabled=False, vllm_enabled=False)).decide(
        features(estimated_prompt_tokens=900, max_new_tokens=512)
    )
    assert decision.path == InferencePath.BASELINE


def test_code_prompt_selects_baseline_not_int4():
    decision = AdaptivePolicy(RouterConfig()).decide(features(request_type="code"))
    assert decision.path == InferencePath.BASELINE


def test_batch_two_short_request_does_not_select_int4():
    decision = AdaptivePolicy(RouterConfig()).decide(features(batch_size=2))
    assert decision.path == InferencePath.BASELINE


def test_latency_prediction_is_attached_to_decision_features():
    decision = AdaptivePolicy(RouterConfig()).decide(features(max_new_tokens=16, latency_target_ms=1000))

    prediction = decision.features["latency_prediction"]
    assert prediction["estimated_latency_ms"] > 0
    assert prediction["latency_target_ms"] == 1000
    assert prediction["latency_target_feasible"] is True


def test_admission_control_can_recommend_rejection():
    decision = AdaptivePolicy(RouterConfig(reject_impossible_latency_targets=True)).decide(
        features(max_new_tokens=512, latency_target_ms=1)
    )

    assert decision.features["latency_prediction"]["latency_target_feasible"] is False
    assert decision.features["admission_control"]["enabled"] is True
    assert decision.features["admission_control"]["should_reject"] is True
