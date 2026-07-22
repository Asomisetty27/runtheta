"""Active probe grading (pure analysis — no GPU, no torch).

Pins the honesty rules from the capability audit:
- spec-fraction against vendor datasheet peaks; unknown SKU = absolutes only;
- conservative floor flags, WITHHELD when the probe thermally throttled
  (a throttled probe measures the cooling, not the silicon);
- cohort_z stays null with an explicit reason until a corpus exists;
- the certificate carries the block (or an explicit not-run note) and the
  integrity seal still verifies.
"""
from theta.agent.activeprobe import FLOOR_FRAC, grade_probe, lookup_spec
from theta.agent.certificate import build_certificate, verify_certificate

H100_GOOD = dict(gemm_tflops=820.0, mem_bw_gbs=2900.0, duration_s=60.0,
                 max_temp_c=64, mean_power_w=690.0, min_sm_mhz=1750,
                 thermal_throttle_s=0)


class TestSpecLookup:
    def test_longest_match_wins(self):
        assert lookup_spec("NVIDIA GeForce RTX 3080 Ti")["sku"] == "RTX 3080 Ti"

    def test_unknown_sku_is_none(self):
        assert lookup_spec("Radeon RX 550") is None


class TestGrading:
    def test_healthy_h100_fractions_and_no_flags(self):
        g = grade_probe(H100_GOOD, "NVIDIA H100 80GB HBM3")
        assert g["sku_match"] == "H100"
        assert abs(g["spec_fraction"]["gemm"] - 0.829) < 1e-3   # 820/989
        assert abs(g["spec_fraction"]["mem_bw"] - 0.866) < 1e-3  # 2900/3350
        assert g["flags"] == []
        assert g["validation"] == "measured-active"

    def test_below_floor_flags(self):
        weak = {**H100_GOOD, "gemm_tflops": 400.0, "mem_bw_gbs": 1500.0}
        g = grade_probe(weak, "NVIDIA H100 80GB HBM3")
        assert "gemm_below_floor" in g["flags"]        # 0.40 < 0.60
        assert "mem_bw_below_floor" in g["flags"]      # 0.45 < 0.60

    def test_throttled_probe_withholds_verdict(self):
        # slow AND throttled: the flag says throttled, not below-floor —
        # the probe measured the cooling, not the silicon
        hot = {**H100_GOOD, "gemm_tflops": 400.0, "thermal_throttle_s": 22}
        g = grade_probe(hot, "NVIDIA H100 80GB HBM3")
        assert g["flags"] == ["thermally_throttled_during_probe"]

    def test_unknown_sku_reports_absolutes_only(self):
        g = grade_probe(H100_GOOD, "Mystery Accelerator 9000")
        assert g["sku_match"] is None
        assert g["spec_fraction"]["gemm"] is None
        assert "unknown SKU" in g["spec_fraction"]["basis"]
        assert g["flags"] == []                        # no spec, no floor verdict
        assert g["measured"]["gemm_tflops"] == 820.0   # numbers still recorded

    def test_cohort_z_null_until_corpus(self):
        g = grade_probe(H100_GOOD, "NVIDIA H100 80GB HBM3")
        assert g["cohort_z"] is None
        assert "pending corpus" in g["cohort_z_reason"]

    def test_floor_is_conservative(self):
        assert FLOOR_FRAC <= 0.65   # never flag harness overhead as a bad card


class TestCertificateIntegration:
    IDENT = {"gpu_index": 0, "name": "H100", "uuid": "GPU-x"}

    def test_probe_block_rides_the_certificate(self):
        block = grade_probe(H100_GOOD, "NVIDIA H100 80GB HBM3")
        c = build_certificate(now=1e9, agent_version="0.1.13",
                              identity=self.IDENT, active_probe=block)
        assert c["active_probe"]["validation"] == "measured-active"
        assert verify_certificate(c)

    def test_passive_certificate_says_not_run(self):
        c = build_certificate(now=1e9, agent_version="0.1.13",
                              identity=self.IDENT)
        assert c["active_probe"]["run"] is False
        assert "certify --active" in c["active_probe"]["note"]
        assert verify_certificate(c)
