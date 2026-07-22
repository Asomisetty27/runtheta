# Theta Helm chart

Deploys the Theta GPU-reliability agent as a per-node **DaemonSet** — the same
pattern as the NVIDIA GPU Operator's DCGM exporter, which Theta is designed to
run alongside.

## Quick start

GPU cluster (nodes managed by the NVIDIA GPU Operator):

```bash
helm install theta deploy/helm/theta
```

Schedules onto nodes labeled `nvidia.com/gpu.present=true`, runs under the
`nvidia` RuntimeClass, and tolerates `nvidia.com/gpu` taints.

Local development (no GPU — agent starts in demo/synthetic-telemetry mode):

```bash
k3d cluster create theta-dev --agents 1
docker build -t somisett/agent:0.1.10 .
k3d image import somisett/agent:0.1.10 -c theta-dev
helm install theta deploy/helm/theta --set gpu.enabled=false --set image.tag=0.1.10
```

Verify:

```bash
kubectl get pods -l app.kubernetes.io/instance=theta
kubectl run curl-test --rm -i --restart=Never --image=curlimages/curl -- \
  -s http://theta-metrics:9101/metrics | grep theta_gpu_rtheta
```

Watch the whole fleet live (one card per GPU, labeled by node — auto
port-forwards to every agent pod and cleans up on exit):

```bash
pip install 'runtheta[ui]'
theta top --k8s
```

## Prometheus

Metrics are exposed on every pod (`:9101/metrics`) and discovered via:

- the annotated headless service `<release>-metrics`
  (`prometheus.io/scrape` annotations), or
- a `ServiceMonitor` (`--set metrics.serviceMonitor.enabled=true`,
  requires the Prometheus Operator CRDs).

## Key values

| Value | Default | Meaning |
|---|---|---|
| `gpu.enabled` | `true` | GPU-node scheduling (runtimeClass, selector, tolerations); `false` = demo mode anywhere |
| `agent.interval` | `5` | Seconds between telemetry polls |
| `agent.metricsPort` | `9101` | Prometheus port |
| `metrics.serviceMonitor.enabled` | `false` | Emit a Prometheus Operator ServiceMonitor |
| `image.tag` | chart `appVersion` | Agent image tag |

The container runs as UID 1000 (non-root), read-only capabilities dropped,
`automountServiceAccountToken: false` — the agent needs no Kubernetes API access.
