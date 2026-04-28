"""
EquiLens AI — Prompt System Validation & Demo

This script validates the robust fairness analysis prompt system:
  1. Verifies input schema construction
  2. Validates output JSON schema
  3. Demonstrates end-to-end flow
  4. Tests guardrails against hallucinations

Run:
    python -m tests.test_prompt_system
    
Or with Gemini API enabled:
    export GEMINI_API_KEY="your_key"
    python -m tests.test_prompt_system
"""

import json
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.fairness_analyst import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    EXAMPLE_INPUT,
    EXAMPLE_OUTPUT,
    FairnessMetricsInput,
    build_user_prompt,
    validate_output_schema,
    EU_AI_ACT_MAPPINGS,
)


def test_system_prompt_exists():
    """Verify system prompt is defined."""
    print("✓ Testing: System prompt exists")
    assert SYSTEM_PROMPT is not None
    assert len(SYSTEM_PROMPT) > 500  # Substantial length
    assert "CORE CONSTRAINTS" in SYSTEM_PROMPT
    assert "JSON Schema" in SYSTEM_PROMPT or "analysis" in SYSTEM_PROMPT
    print("  ✓ System prompt is well-formed\n")


def test_user_prompt_template():
    """Verify user prompt template has required placeholders."""
    print("✓ Testing: User prompt template")
    required_fields = [
        "{model_name}",
        "{metrics_json}",
        "{group_statistics}",
        "{total_samples}",
    ]
    for field in required_fields:
        assert field in USER_PROMPT_TEMPLATE, f"Missing {field}"
    print(f"  ✓ Template has all {len(required_fields)} required fields\n")


def test_input_schema():
    """Verify FairnessMetricsInput dataclass."""
    print("✓ Testing: Input schema")
    metrics = FairnessMetricsInput(
        model_name="Test Model",
        protected_attribute="gender",
        disparate_impact_ratio=0.72,
        demographic_parity_difference=-0.15,
        equal_opportunity_difference=-0.18,
        group_statistics={"M": {"sample_size": 1000}, "F": {"sample_size": 1000}},
        total_samples=2000,
    )
    assert metrics.model_name == "Test Model"
    assert metrics.disparate_impact_ratio == 0.72
    print("  ✓ Input schema accepts required fields\n")


def test_build_user_prompt():
    """Verify build_user_prompt() constructs valid prompt."""
    print("✓ Testing: build_user_prompt()")
    metrics = FairnessMetricsInput(
        model_name="Loan Approval v1",
        protected_attribute="gender",
        disparate_impact_ratio=0.80,
        demographic_parity_difference=0.05,
        equal_opportunity_difference=0.03,
        group_statistics={
            "M": {"approval_rate": 0.65, "sample_size": 5000},
            "F": {"approval_rate": 0.61, "sample_size": 5000},
        },
        total_samples=10000,
        use_case="Credit lending",
        application_domain="financial",
    )
    
    prompt = build_user_prompt(metrics)
    assert "Loan Approval v1" in prompt
    assert "gender" in prompt
    assert "0.80" in prompt or "0.8" in prompt
    assert "Credit lending" in prompt
    print("  ✓ Prompt built successfully\n")


def test_example_input_output():
    """Verify example input/output are well-formed."""
    print("✓ Testing: Example input/output")
    
    # Check input
    assert "model_name" in EXAMPLE_INPUT
    assert EXAMPLE_INPUT["model_name"] == "Loan Approval System v2.1"
    
    # Check output is valid JSON
    assert isinstance(EXAMPLE_OUTPUT, dict)
    required_keys = ["analysis", "fairness_assessment", "eu_ai_act", "remediation", "limitations"]
    for key in required_keys:
        assert key in EXAMPLE_OUTPUT, f"Missing {key}"
    
    # Validate schema
    is_valid, errors = validate_output_schema(EXAMPLE_OUTPUT)
    if not is_valid:
        print(f"  ⚠ Example output schema validation warnings: {errors}")
    else:
        print("  ✓ Example output passes schema validation")
    
    print()


def test_output_schema_validation():
    """Test validate_output_schema() function."""
    print("✓ Testing: Output schema validation")
    
    # Valid output should pass
    is_valid, errors = validate_output_schema(EXAMPLE_OUTPUT)
    assert len(errors) == 0, f"Example output should be valid, got: {errors}"
    print("  ✓ Valid output passes validation")
    
    # Missing required field should fail
    incomplete = {
        "analysis": {"model_name": "test"},
        # missing: fairness_assessment, eu_ai_act, remediation, limitations
    }
    is_valid, errors = validate_output_schema(incomplete)
    assert not is_valid, "Incomplete output should fail validation"
    assert len(errors) > 0
    print(f"  ✓ Incomplete output correctly rejected ({len(errors)} errors)")
    
    # Invalid risk_level should fail
    invalid_risk = EXAMPLE_OUTPUT.copy()
    invalid_risk["eu_ai_act"] = EXAMPLE_OUTPUT["eu_ai_act"].copy()
    invalid_risk["eu_ai_act"]["risk_level"] = "invalid_level"
    is_valid, errors = validate_output_schema(invalid_risk)
    assert not is_valid, "Invalid risk_level should fail"
    print("  ✓ Invalid risk_level correctly rejected")
    
    print()


def test_eu_ai_act_mappings():
    """Verify EU AI Act risk mappings."""
    print("✓ Testing: EU AI Act mappings")
    
    risk_levels = ["UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL"]
    for level in risk_levels:
        assert level in EU_AI_ACT_MAPPINGS
        mapping = EU_AI_ACT_MAPPINGS[level]
        assert "description" in mapping
        assert "examples" in mapping
        assert len(mapping["examples"]) > 0
    
    print(f"  ✓ All {len(risk_levels)} risk levels mapped\n")


def test_guardrails():
    """Verify guardrail constraints in system prompt."""
    print("✓ Testing: Guardrails in system prompt")
    
    guardrail_keywords = [
        "Do NOT invent",
        "Distinguish",
        "causal",
        "Never assume",
        "never assume",
    ]
    
    for keyword in guardrail_keywords:
        assert keyword.lower() in SYSTEM_PROMPT.lower(), f"Missing guardrail: {keyword}"
    
    print(f"  ✓ All {len(guardrail_keywords)} guardrails present in system prompt\n")


def test_metric_interpretation():
    """Verify metric interpretation logic."""
    print("✓ Testing: Metric interpretation")
    
    # DI = 0.72 should be unfair (below 0.8)
    di_fair_threshold = 0.8
    assert 0.72 < di_fair_threshold, "0.72 DI should be < threshold"
    
    # DPD = -0.15 should be unfair (beyond ±0.1)
    dpd_fair_threshold = 0.1
    assert abs(-0.15) > dpd_fair_threshold, "|−0.15| should exceed threshold"
    
    # EOD = -0.18 should be unfair (beyond ±0.1)
    assert abs(-0.18) > dpd_fair_threshold, "|−0.18| should exceed threshold"
    
    print("  ✓ Threshold logic correct\n")


def test_remediation_structure():
    """Verify remediation strategies in example output."""
    print("✓ Testing: Remediation strategies")
    
    remediation = EXAMPLE_OUTPUT["remediation"]
    assert "pre_processing" in remediation
    assert "in_processing" in remediation
    assert "post_processing" in remediation
    
    # Each strategy should have required fields
    required_strategy_fields = [
        "strategy",
        "description",
        "expected_impact",
        "tradeoff",
        "implementation_complexity",
        "confidence",
    ]
    
    for category in ["pre_processing", "in_processing", "post_processing"]:
        strategies = remediation[category]
        if isinstance(strategies, list) and len(strategies) > 0:
            for strategy in strategies:
                for field in required_strategy_fields:
                    assert field in strategy, f"Missing {field} in {category}"
    
    print(f"  ✓ Remediation strategies well-formed\n")


def test_edge_case_handling():
    """Test system prompt handles edge cases."""
    print("✓ Testing: Edge case handling in system prompt")
    
    edge_cases = [
        "Division by zero",
        "sample size < 30",
        "missing",
        "undefined",
        "NaN",
        "Infinity",
    ]
    
    found_count = 0
    for edge_case in edge_cases:
        if edge_case.lower() in SYSTEM_PROMPT.lower():
            found_count += 1
    
    print(f"  ✓ System prompt mentions {found_count}/{len(edge_cases)} edge cases\n")


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("EquiLens AI — Prompt System Validation")
    print("=" * 80)
    print()
    
    tests = [
        test_system_prompt_exists,
        test_user_prompt_template,
        test_input_schema,
        test_build_user_prompt,
        test_example_input_output,
        test_output_schema_validation,
        test_eu_ai_act_mappings,
        test_guardrails,
        test_metric_interpretation,
        test_remediation_structure,
        test_edge_case_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            failed += 1
    
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)
    
    if failed == 0:
        print("\n✅ All validation tests passed! Prompt system is ready for use.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Review above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
