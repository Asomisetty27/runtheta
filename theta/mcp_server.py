"""
Theta MCP server — exposes live daemon state as Claude tools.

Implements the Model Context Protocol (stdio transport, JSON-RPC 2.0).
Reads from the Theta health API (port 9102 by default). Falls back to
a "daemon unreachable" response rather than erroring, so the tool still
works when Theta isn't running (returns last-known or error state).

Add to .mcp.json:
  {
    "mcpServers": {
      "theta": {
        "command": "python",
        "args": ["/Users/amogh/thermalos-agent/theta/mcp_server.py"],
        "env": {"THETA_PORT": "9102"}
      }
    }
  }

Tools exposed:
  theta_fleet_status   — R_θ, risk, state for all GPUs
  theta_gpu_details    — causal explanation, fault, maintenance for one GPU
  theta_gpu_risk       — quick risk score for a GPU (sub-second)
  theta_fleet_summary  — one-line narrative: "3 GPUs nominal, GPU 2 drifting"
  theta_gpu_prognosis  — forward-looking per-component health + RUL + confidence tier
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

THETA_PORT = int(os.environ.get("THETA_PORT", "9102"))
SERVER_NAME = "theta-mcp"
SERVER_VERSION = "0.1.0"

TOOLS = [
    {
        "name": "theta_fleet_status",
        "description": (
            "Get current thermal state for all GPUs from the running Theta daemon. "
            "Returns R_θ (thermal resistance C/W), risk score, state (idle/load/drifting/critical), "
            "and fault classification for each GPU."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "theta_gpu_details",
        "description": (
            "Get rich diagnostic detail for a single GPU: causal explanation of why it's in its "
            "current state, fault curve analysis, maintenance score, and CNN prediction. "
            "Use this when a GPU shows anomalous risk or state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "gpu_index": {
                    "type": "integer",
                    "description": "Zero-based GPU index (0 = first GPU)",
                }
            },
            "required": ["gpu_index"],
        },
    },
    {
        "name": "theta_gpu_risk",
        "description": (
            "Get the risk score (0.0–1.0) for a specific GPU. "
            "0.0 = nominal, >0.5 = watch, >0.8 = act immediately."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "gpu_index": {
                    "type": "integer",
                    "description": "Zero-based GPU index",
                }
            },
            "required": ["gpu_index"],
        },
    },
    {
        "name": "theta_fleet_summary",
        "description": (
            "Get a one-sentence narrative summary of fleet health. "
            "E.g. '7 GPUs nominal, GPU 3 drifting (R_θ=0.074, risk=0.81, fault=cooling_degradation)'. "
            "Use this for quick status checks before diving into details."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "theta_gpu_prognosis",
        "description": (
            "Get the FORWARD-LOOKING prognostic view for one GPU: which specific subsystem "
            "(die/TIM/HBM/power-delivery/fan/fabric/silicon) is degrading, its graded health "
            "(1.0 = nominal), confirmed micro-changes, remaining-useful-life (RUL) estimate, and "
            "a per-component confidence tier. Also cross-checks the RUL engine against the "
            "signature fault classifier (engine_agreement: agree / conflict / n/a). "
            "IMPORTANT honesty contract you MUST respect when reasoning over this: an RUL whose "
            "rul_confidence is 'UNCALIBRATED' is a physically-plausible estimate with an ASSUMED "
            "threshold, NOT a validated lead-time — never present it to an operator as a firm "
            "time-to-failure. Only report RUL as actionable when in_alarm is true (a confirmed "
            "micro-change) AND the tier is 'CALIBRATED'. Use theta_gpu_details for the reactive "
            "causal narrative; use this for 'what is quietly drifting and how long do we have'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "gpu_index": {
                    "type": "integer",
                    "description": "Zero-based GPU index (0 = first GPU)",
                }
            },
            "required": ["gpu_index"],
        },
    },
    {
        "name": "theta_knowledge_lookup",
        "description": (
            "Retrieve Theta's grounding knowledge for a topic: what a signature MEANS, "
            "the validated finding that backs it (F7/F15 detection, F16 TIM magnitude, "
            "per-generation R_theta), the honesty tiers, and the concrete repair "
            "playbook for a component. Call this BEFORE explaining why a number matters "
            "or recommending a fix, so the explanation is grounded in validated findings "
            "rather than invented. Returns [] when the corpus has nothing on the query -- "
            "in that case say you lack grounding, do not fabricate it. Query with the "
            "component or concept, e.g. 'TIM degradation repair' or 'what is UNCALIBRATED RUL'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic/component/concept to ground, e.g. 'fan bearing wear' or 'R_theta per generation'.",
                },
                "k": {
                    "type": "integer",
                    "description": "Max entries to return (default 3).",
                },
            },
            "required": ["query"],
        },
    },
]


def _api(path: str) -> dict:
    url = f"http://localhost:{THETA_PORT}{path}"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return json.loads(r.read().decode())
    except urllib.error.URLError as e:
        return {"error": str(e), "daemon_status": "unreachable", "port": THETA_PORT}
    except Exception as e:
        return {"error": str(e)}


def _fleet_summary(status: dict) -> str:
    if "error" in status:
        return f"Theta daemon unreachable on port {THETA_PORT}. Start with: theta monitor"
    gpus = status.get("gpus", [])
    if not gpus:
        return "No GPUs found."
    nominal = [g for g in gpus if g.get("state") in ("clean_idle", "under_load", "idle", "load")]
    anomalous = [g for g in gpus if g.get("state") not in ("clean_idle", "under_load", "idle", "load", None)]
    parts = []
    if nominal:
        parts.append(f"{len(nominal)} GPU{'s' if len(nominal) > 1 else ''} nominal")
    for g in anomalous:
        idx = g.get("index", g.get("gpu_index", "?"))
        state = g.get("state", "unknown")
        rtheta = g.get("rtheta_cw", g.get("rtheta", None))
        risk = g.get("risk_score", None)
        fault = g.get("fault_class", None)
        desc = f"GPU {idx} {state}"
        if rtheta is not None:
            desc += f" (R_θ={rtheta:.3f}"
        if risk is not None:
            desc += f", risk={risk:.2f}"
        if fault:
            desc += f", fault={fault}"
        if rtheta is not None:
            desc += ")"
        parts.append(desc)
    return ", ".join(parts) if parts else "Fleet status unknown."


def _prognosis_from_details(details: dict, idx: int) -> dict:
    """Pull the prognostic report the daemon layered onto causal_explanation.
    Returns a clear, honest message when it isn't present (daemon not running,
    older daemon without the prognostic wiring, or GPU not yet warmed up)."""
    if "error" in details:
        return details
    causal = details.get("causal_explanation") or {}
    prog = causal.get("prognosis")
    if not prog:
        return {
            "gpu_index": idx,
            "prognosis": None,
            "note": (
                "No prognostic report available for this GPU yet. Either the daemon "
                "is not running the prognostic layer, or this GPU has not accumulated "
                "enough steady-state windows to grade its components."
            ),
        }
    return {"gpu_index": idx, **prog}


def call_tool(name: str, args: dict) -> dict:
    """Execute a Theta tool by name and return its raw result dict.

    Single source of truth for tool dispatch, shared by the MCP stdio server
    (_handle) and the LLM operator agent (theta.agent.operator). Every tool here
    is READ-ONLY -- it reads live daemon state and never mutates the fleet or
    fires an alert. That read-only-by-construction property is what lets an LLM
    reason over these tools without ever making a trust-critical action."""
    args = args or {}
    if name == "theta_fleet_status":
        return _api("/api/v1/agent/fleet/status")

    if name == "theta_gpu_details":
        idx = args.get("gpu_index", 0)
        return _api(f"/api/v1/agent/gpu/{idx}/details")

    if name == "theta_gpu_risk":
        idx = args.get("gpu_index", 0)
        health = _api(f"/api/v1/health/gpu/{idx}")
        if "error" in health:
            return health
        return {
            "gpu_index": idx,
            "risk_score": health.get("risk_score"),
            "state": health.get("state"),
            "rtheta_cw": health.get("rtheta_cw"),
        }

    if name == "theta_gpu_prognosis":
        idx = args.get("gpu_index", 0)
        details = _api(f"/api/v1/agent/gpu/{idx}/details")
        return _prognosis_from_details(details, idx)

    if name == "theta_fleet_summary":
        status = _api("/api/v1/agent/fleet/status")
        return {"summary": _fleet_summary(status)}

    if name == "theta_knowledge_lookup":
        # Agentic RAG (Ch 14): local curated corpus, not a daemon call. Read-only.
        from .agent.knowledge import lookup
        entries = lookup(args.get("query", ""), k=int(args.get("k", 3) or 3))
        return {"query": args.get("query", ""), "results": entries}

    return {"error": f"Unknown tool: {name}"}


def _handle(request: dict) -> dict:
    method = request.get("method", "")

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    if method == "notifications/initialized":
        return {}

    if method == "tools/list":
        return {"tools": TOOLS}

    if method == "tools/call":
        name = request.get("params", {}).get("name", "")
        args = request.get("params", {}).get("arguments", {})
        result = call_tool(name, args)
        return {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
        }

    # Unhandled method — return empty result (not an error)
    return {}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        result = _handle(req)

        # notifications don't get a response
        if req.get("method", "").startswith("notifications/"):
            continue

        resp = {
            "jsonrpc": "2.0",
            "id": req.get("id"),
            "result": result,
        }
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
