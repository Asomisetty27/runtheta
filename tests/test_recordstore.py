"""Record store: the hash-chained, anchorable device history (Phase 1).

Pins the trust properties the system design rests on:
- append-only chain: any retroactive edit, deletion, or reorder is detected;
- digests are privacy-safe (no raw telemetry) and signed when crypto exists;
- signature verification rejects tampered digests;
- the anchor spool is offline-first and never loses a digest on transport
  failure.
"""
import json

import pytest

from theta.agent.recordstore import (
    _CRYPTO,
    AnchorSpool,
    RecordStore,
)


@pytest.fixture
def store(tmp_path):
    return RecordStore(tmp_path / "record")


def _seed(store, n=5, t0=1_800_000_000.0):
    for i in range(n):
        store.append("snapshot", {"observed_s": 3600.0, "i": i}, ts=t0 + i * 3600)


class TestChain:
    def test_append_links_and_verifies(self, store):
        _seed(store)
        assert store.verify_chain()

    def test_value_edit_detected(self, store):
        _seed(store)
        seg = next(store.root.glob("records-*.jsonl"))
        lines = seg.read_text().splitlines()
        e = json.loads(lines[2])
        e["data"]["observed_s"] = 999999.0          # forge more coverage
        lines[2] = json.dumps(e, sort_keys=True)
        seg.write_text("\n".join(lines) + "\n")
        assert not store.verify_chain()

    def test_deletion_detected(self, store):
        _seed(store)
        seg = next(store.root.glob("records-*.jsonl"))
        lines = seg.read_text().splitlines()
        seg.write_text("\n".join(lines[:2] + lines[3:]) + "\n")   # drop an entry
        assert not store.verify_chain()

    def test_truncation_detected(self, store):
        # dropping the TAIL (hide last month's incidents) breaks HEAD linkage
        _seed(store)
        seg = next(store.root.glob("records-*.jsonl"))
        lines = seg.read_text().splitlines()
        seg.write_text("\n".join(lines[:-2]) + "\n")
        assert not store.verify_chain()

    def test_summary_spans_and_observed_hours(self, store):
        _seed(store, n=5)
        s = store.summary()
        assert s["n_entries"] == 5
        assert s["observed_hours"] == 5.0
        assert s["span_end"] - s["span_start"] == 4 * 3600
        assert s["kinds"] == {"snapshot": 5}


class TestDigest:
    def test_digest_is_privacy_safe(self, store):
        _seed(store)
        store.append("incident", {"gpu": 3, "category": "fallen_off_bus"})
        d = store.build_digest(now=1_800_100_000.0)
        blob = json.dumps(d)
        assert "fallen_off_bus" not in blob          # no event payloads
        assert "observed_s" not in blob              # no raw fields
        assert d["kinds"]["incident"] == 1           # counts only
        assert d["chain_head"] == store.head()

    @pytest.mark.skipif(not _CRYPTO, reason="cryptography not installed")
    def test_signature_verifies_and_rejects_tamper(self, store):
        _seed(store)
        d = store.build_digest(now=1_800_100_000.0)
        assert d["signed"] and RecordStore.verify_digest(d)
        d2 = dict(d)
        d2["observed_hours"] = 8760.0                # forge a year of coverage
        assert not RecordStore.verify_digest(d2)

    @pytest.mark.skipif(not _CRYPTO, reason="cryptography not installed")
    def test_keypair_is_stable_across_instances(self, tmp_path):
        a = RecordStore(tmp_path / "r")
        k1 = a.public_key_hex()
        b = RecordStore(tmp_path / "r")               # reopen same dir
        assert b.public_key_hex() == k1


class TestAnchorSpool:
    def test_spool_only_mode_reports_pending(self, store):
        _seed(store)
        spool = AnchorSpool(store, post_fn=None)
        spool.enqueue()
        r = spool.flush()
        assert r["pending"] == 1 and "spool-only" in r["note"]

    def test_flush_moves_sent_and_keeps_failed(self, store):
        _seed(store)
        calls = []

        def flaky(digest):
            calls.append(digest)
            return len(calls) != 1                   # first post fails
        spool = AnchorSpool(store, post_fn=flaky)
        spool.enqueue(store.build_digest(now=1.0))
        spool.enqueue(store.build_digest(now=2.0))
        r = spool.flush()
        assert r["sent"] == 1 and r["failed"] == 1 and r["pending"] == 1
        r2 = spool.flush()                           # retry succeeds
        assert r2["sent"] == 1 and r2["pending"] == 0

    def test_transport_exception_keeps_digest_queued(self, store):
        _seed(store)

        def boom(digest):
            raise ConnectionError("network down")
        spool = AnchorSpool(store, post_fn=boom)
        spool.enqueue()
        r = spool.flush()                            # must not raise
        assert r["failed"] == 1 and r["pending"] == 1
