"""
The device record store — Phase 1 of the health-record system (agent side).

The system design's load-bearing fact: a certificate generated on the
seller's machine can seal itself, but it cannot prove its INPUTS were honest,
because the certifying machine belongs to the party with the incentive to
lie. The fix is structural, not cryptographic cleverness: an append-only,
hash-chained local record whose small signed digests are ANCHORED off-host
over time. You can forge today's file; you cannot forge last March's anchor.

This module is the local half:

  * RecordStore — append-only JSONL segments (one per month). Every entry
    commits to the previous entry's hash; the chain head is the record's
    fingerprint at a point in time. verify_chain() re-walks everything, so
    any retroactive edit is detectable.
  * Digests — a compact, privacy-safe summary (span, entry counts, kind
    histogram, chain head, observed-hours). NO raw telemetry leaves the host
    in a digest; that is the default-ON privacy posture from the design.
  * Signing — ed25519 with an install-time keypair when the `cryptography`
    library is present (same optional-dependency posture as secrets.py:
    degrade loudly, never silently). Unsigned digests still chain — they are
    marked signed=false and the countersign service grades them lower.
  * AnchorSpool — digests queue on disk; a flush posts them to the anchor
    endpoint when configured/reachable and keeps them queued otherwise.
    Offline operation loses nothing; it just anchors late (and the anchor
    timestamps say so — that honesty is part of the trust model).

Entry conventions (write side lands with the daemon wiring):
  kind="snapshot"  data={"observed_s": <covered seconds>, ...aggregates}
  kind="incident"  data={"gpu": i, "category": ..., "node_synchronous": bool}
  kind="alert"     data={...}
  kind="probe"     data=<activeprobe grade block>
  kind="flash"     data={"vbios": old->new}   (the flash-washing detector)

Pure file I/O + hashlib; no NVML. Fully testable without a GPU.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

GENESIS = "genesis"
DIGEST_SCHEMA = "theta-anchor-digest/v1"

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )
    _CRYPTO = True
except ImportError:                                   # pragma: no cover
    _CRYPTO = False


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RecordStore:
    """Append-only, hash-chained device record. One instance per record dir."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._head_file = self.root / "HEAD"
        self._key_file = self.root / "agent_key.pem"
        self._pub_file = self.root / "agent_key.pub"

    # ── chain ────────────────────────────────────────────────────────────────
    def head(self) -> str:
        if self._head_file.exists():
            return self._head_file.read_text().strip()
        return GENESIS

    def _segment(self, ts: float) -> Path:
        ym = time.strftime("%Y%m", time.gmtime(ts))
        return self.root / f"records-{ym}.jsonl"

    def append(self, kind: str, data: dict, ts: Optional[float] = None) -> dict:
        ts = time.time() if ts is None else ts
        entry = {"ts": ts, "kind": kind, "data": data, "prev": self.head()}
        entry["hash"] = _sha256(_canonical(entry))
        with self._segment(ts).open("a") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        self._head_file.write_text(entry["hash"])
        return entry

    def entries(self):
        for seg in sorted(self.root.glob("records-*.jsonl")):
            with seg.open() as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)

    def verify_chain(self) -> bool:
        """Re-walk every entry: prev-links intact AND every hash recomputes.
        Any retroactive edit — value change, deletion, reorder — fails here."""
        prev = GENESIS
        last = GENESIS
        for e in self.entries():
            if e.get("prev") != prev:
                return False
            body = {k: v for k, v in e.items() if k != "hash"}
            if _sha256(_canonical(body)) != e.get("hash"):
                return False
            prev = e["hash"]
            last = e["hash"]
        return self.head() == last

    # ── summary / digest ─────────────────────────────────────────────────────
    def summary(self) -> dict:
        span_start = span_end = None
        n = 0
        kinds: dict[str, int] = {}
        observed_s = 0.0
        for e in self.entries():
            n += 1
            span_start = e["ts"] if span_start is None else min(span_start, e["ts"])
            span_end = e["ts"] if span_end is None else max(span_end, e["ts"])
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
            if e["kind"] == "snapshot":
                observed_s += float(e["data"].get("observed_s", 0.0))
        return {
            "span_start": span_start, "span_end": span_end,
            "n_entries": n, "kinds": kinds,
            "observed_hours": round(observed_s / 3600.0, 2),
            "chain_head": self.head(),
        }

    def build_digest(self, now: Optional[float] = None) -> dict:
        """The privacy-safe anchor payload: summary + chain head, signed when
        possible. No raw telemetry — span, counts, and hashes only."""
        d = {
            "schema": DIGEST_SCHEMA,
            "generated_at": time.time() if now is None else now,
            **self.summary(),
            "public_key": self.public_key_hex(),
        }
        sig = self._sign(_canonical(d))
        d["signed"] = sig is not None
        if sig is not None:
            d["signature"] = sig
        return d

    # ── keys / signing (optional-crypto posture, as secrets.py) ─────────────
    def _ensure_key(self):
        if not _CRYPTO:
            return None
        if self._key_file.exists():
            from cryptography.hazmat.primitives.serialization import (
                load_pem_private_key,
            )
            return load_pem_private_key(self._key_file.read_bytes(), password=None)
        key = Ed25519PrivateKey.generate()
        self._key_file.write_bytes(key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
        self._key_file.chmod(0o600)
        self._pub_file.write_bytes(key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw))
        return key

    def public_key_hex(self) -> Optional[str]:
        if not _CRYPTO:
            return None
        self._ensure_key()
        return self._pub_file.read_bytes().hex() if self._pub_file.exists() else None

    def _sign(self, payload: bytes) -> Optional[str]:
        if not _CRYPTO:
            log.warning("cryptography not installed - anchor digests are "
                        "UNSIGNED (chained only). pip install cryptography")
            return None
        key = self._ensure_key()
        return key.sign(payload).hex()

    @staticmethod
    def verify_digest(digest: dict) -> bool:
        """True iff the digest's signature verifies against its embedded key.
        Unsigned digests return False — the caller decides how to grade them."""
        if not _CRYPTO or not digest.get("signed"):
            return False
        body = {k: v for k, v in digest.items()
                if k not in ("signed", "signature")}
        try:
            pub = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(digest["public_key"]))
            pub.verify(bytes.fromhex(digest["signature"]), _canonical(body))
            return True
        except Exception:  # noqa: BLE001 — any failure = not verified
            return False


class AnchorSpool:
    """Disk-backed queue of digests awaiting anchor upload.

    Offline-first: enqueue always succeeds; flush() posts what it can and
    leaves the rest. The transport is injectable (post_fn) so tests and
    future backends (Supabase endpoint, self-hosted) share the logic."""

    def __init__(self, store: RecordStore,
                 post_fn: Optional[Callable[[dict], bool]] = None):
        self.store = store
        self.pending = store.root / "anchors" / "pending"
        self.sent = store.root / "anchors" / "sent"
        self.pending.mkdir(parents=True, exist_ok=True)
        self.sent.mkdir(parents=True, exist_ok=True)
        self._post = post_fn

    def enqueue(self, digest: Optional[dict] = None) -> Path:
        d = digest or self.store.build_digest()
        name = f"digest-{int(d['generated_at'])}-{d['chain_head'][:12]}.json"
        p = self.pending / name
        p.write_text(json.dumps(d, sort_keys=True))
        return p

    def flush(self) -> dict:
        """Attempt upload of every pending digest. Returns counts; never raises
        on transport failure (the digest just stays pending)."""
        sent = failed = 0
        if self._post is None:
            return {"sent": 0, "failed": 0,
                    "pending": len(list(self.pending.glob("*.json"))),
                    "note": "no anchor endpoint configured - spool-only mode"}
        for p in sorted(self.pending.glob("*.json")):
            try:
                ok = self._post(json.loads(p.read_text()))
            except Exception:  # noqa: BLE001 — transport errors keep it queued
                ok = False
            if ok:
                p.rename(self.sent / p.name)
                sent += 1
            else:
                failed += 1
        return {"sent": sent, "failed": failed,
                "pending": len(list(self.pending.glob("*.json")))}


def http_post_fn(endpoint: str, timeout: float = 10.0) -> Callable[[dict], bool]:
    """Default transport: POST the digest JSON; 2xx = anchored."""
    def _post(digest: dict) -> bool:
        import httpx
        r = httpx.post(endpoint, json=digest, timeout=timeout)
        return 200 <= r.status_code < 300
    return _post
