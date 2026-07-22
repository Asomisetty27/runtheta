"""Fleet economics accounting (turn detections into dollars).

Pins the three value numbers and the honesty guards:
- realized loss counts ONLY thermal throttle (not power-cap — that's intentional);
- clock deficit weights the lost time (throttled-but-near-max costs little);
- recurrence exposure only for watched units;
- capacity-at-cliff = degraded AND low thermal margin.
"""
from theta.agent.economics import FleetEconomics, PriceConfig

POWER_CAP_BIT = 0x04
HW_THERMAL_BIT = 0x40


def _run(ticks, prices=None):
    fe = FleetEconomics(prices)
    for t in ticks:
        fe.observe(**t)
    return fe


def _tick(gpu, ts, *, throttle=0, sm=1400, sm_max=1400, power=300.0,
          margin=None, degraded=False, watch=False, peers=0):
    return dict(gpu_index=gpu, ts=ts, power_w=power, throttle_reasons=throttle,
                sm_clock_mhz=sm, sm_clock_max_mhz=sm_max, thermal_margin_c=margin,
                degraded_cooling=degraded, on_recurrence_watch=watch,
                sync_peers=peers)


class TestRealizedLoss:
    def test_thermal_throttle_at_deficit_costs_money(self):
        # 3600 s throttled at 1050/1400 = 25% deficit → 0.25 lost GPU-hours
        fe = _run([_tick(0, 0), _tick(0, 3600, throttle=HW_THERMAL_BIT, sm=1050)],
                  PriceConfig(gpu_hour_usd=2.0))
        r = fe.gpu_report(0)
        assert abs(r["lost_gpu_hours"] - 0.25) < 1e-6
        assert abs(r["realized_loss_usd"] - 0.50) < 1e-6

    def test_power_cap_throttle_is_not_charged_to_cooling(self):
        # power-cap throttle at a big clock deficit must cost $0 (intentional, not a fault)
        fe = _run([_tick(0, 0), _tick(0, 3600, throttle=POWER_CAP_BIT, sm=700)])
        assert fe.gpu_report(0)["realized_loss_usd"] == 0.0

    def test_healthy_full_clock_costs_nothing(self):
        fe = _run([_tick(0, 0), _tick(0, 3600, sm=1400, sm_max=1400)])
        assert fe.gpu_report(0)["realized_loss_usd"] == 0.0

    def test_large_gap_between_ticks_ignored(self):
        # a >1h gap (agent restart) must not accrue phantom time
        fe = _run([_tick(0, 0), _tick(0, 10000, throttle=HW_THERMAL_BIT, sm=700)])
        assert fe.gpu_report(0)["observed_hours"] == 0.0


class TestHeadroomAndRecurrence:
    def test_min_margin_tracked(self):
        fe = _run([_tick(0, 0, margin=12.0), _tick(0, 60, margin=3.0), _tick(0, 120, margin=8.0)])
        assert fe.gpu_report(0)["min_thermal_margin_c"] == 3.0

    def test_recurrence_exposure_only_when_watched(self):
        prices = PriceConfig(gpu_hour_usd=2.0, incident_downtime_hours=4.0)
        fe = _run([_tick(0, 0, watch=True), _tick(0, 60, watch=True)], prices)
        # 0.56 * 4h * $2 = $4.48
        assert abs(fe.gpu_report(0)["recurrence_exposure_usd"] - 4.48) < 1e-6
        fe2 = _run([_tick(1, 0), _tick(1, 60)], prices)
        assert fe2.gpu_report(1)["recurrence_exposure_usd"] == 0.0


class TestStragglerExposure:
    """The synchronous-barrier tax: a throttled straggler charges its peers.

    Pins the honesty guards: upper-bound only accrues on OBSERVED thermal
    throttle (F25: never projected from R_theta), and is zero without job info.
    """

    def test_throttled_straggler_charges_its_peers(self):
        # 3600 s thermally throttled at 25% deficit in an 8-GPU job (7 peers):
        # own loss 0.25 GPU-h ($0.50); barrier tax 0.25*7 = 1.75 GPU-h ($3.50)
        fe = _run([_tick(0, 0, peers=7),
                   _tick(0, 3600, throttle=HW_THERMAL_BIT, sm=1050, peers=7)],
                  PriceConfig(gpu_hour_usd=2.0))
        r = fe.gpu_report(0)
        assert abs(r["realized_loss_usd"] - 0.50) < 1e-6
        assert abs(r["straggler_exposure_usd"] - 3.50) < 1e-6

    def test_no_job_info_means_zero_straggler_exposure(self):
        # same throttle, no sync_peers -> straggler line stays 0 (old behavior)
        fe = _run([_tick(0, 0), _tick(0, 3600, throttle=HW_THERMAL_BIT, sm=1050)])
        assert fe.gpu_report(0)["straggler_exposure_usd"] == 0.0

    def test_power_cap_throttle_never_charges_peers(self):
        # power-cap throttle is intentional, not a cooling fault — $0 even with peers
        fe = _run([_tick(0, 0, peers=7),
                   _tick(0, 3600, throttle=POWER_CAP_BIT, sm=700, peers=7)])
        assert fe.gpu_report(0)["straggler_exposure_usd"] == 0.0

    def test_healthy_gpu_in_a_job_costs_nothing(self):
        # unthrottled at full clock with peers -> no straggler exposure
        fe = _run([_tick(0, 0, peers=7), _tick(0, 3600, peers=7)])
        assert fe.gpu_report(0)["straggler_exposure_usd"] == 0.0


class TestFleetRollup:
    def test_capacity_at_cliff_needs_degraded_and_low_margin(self):
        fe = _run([
            _tick(0, 0, degraded=True, margin=2.0), _tick(0, 60, degraded=True, margin=2.0),  # at cliff
            _tick(1, 0, degraded=True, margin=20.0), _tick(1, 60, degraded=True, margin=20.0),  # degraded, safe
            _tick(2, 0, degraded=False, margin=1.0), _tick(2, 60, degraded=False, margin=1.0),  # low margin but healthy
        ])
        rep = fe.fleet_report()
        assert rep["degraded_cooling_units"] == 2
        assert rep["capacity_at_cliff_units"] == 1   # only gpu0

    def test_total_exposure_sums_realized_and_recurrence(self):
        prices = PriceConfig(gpu_hour_usd=2.0, incident_downtime_hours=4.0)
        fe = _run([
            _tick(0, 0), _tick(0, 3600, throttle=HW_THERMAL_BIT, sm=700, sm_max=1400),  # 0.5 lost h → $1
            _tick(1, 0, watch=True), _tick(1, 60, watch=True),                          # $4.48
        ], prices)
        rep = fe.fleet_report()
        assert abs(rep["realized_loss_usd"] - 1.00) < 1e-6
        assert abs(rep["recurrence_exposure_usd"] - 4.48) < 1e-6
        assert abs(rep["total_exposure_usd"] - 5.48) < 1e-6
        assert rep["ranked_units"][0]["gpu_index"] == 1   # recurrence unit ranks above the $1 one

    def test_total_exposure_includes_straggler_and_reranks(self):
        prices = PriceConfig(gpu_hour_usd=2.0, incident_downtime_hours=4.0)
        fe = _run([
            # straggler in an 8-GPU job: $1 own + $7 barrier tax = $8 in play
            _tick(0, 0, peers=7),
            _tick(0, 3600, throttle=HW_THERMAL_BIT, sm=700, sm_max=1400, peers=7),
            _tick(1, 0, watch=True), _tick(1, 60, watch=True),   # $4.48 recurrence
        ], prices)
        rep = fe.fleet_report()
        assert abs(rep["straggler_exposure_usd"] - 7.00) < 1e-6
        assert abs(rep["total_exposure_usd"] - (1.00 + 4.48 + 7.00)) < 1e-6
        # the straggler now outranks the recurrence unit: it costs the JOB more
        assert rep["ranked_units"][0]["gpu_index"] == 0
