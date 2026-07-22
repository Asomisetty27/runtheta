// theta-node-labeler — a small Kubernetes controller that reconciles node
// labels from Theta GPU-health telemetry.
//
// The control loop (the core Kubernetes pattern):
//
//	observe:   list Theta agent pods (one per GPU node, via the DaemonSet)
//	measure:   scrape each pod's Prometheus endpoint for theta_gpu_schedulable
//	reconcile: patch the pod's Node with labels describing GPU health
//
// Downstream consumers can then act on the labels without knowing anything
// about Theta: schedulers can avoid `theta.dev/gpu-health=degraded` nodes via
// nodeAffinity, ops dashboards can group by label, and drain automation
// (e.g. Draino/NVSentinel-style tooling) can key off it.
//
// Labels written:
//
//	theta.dev/gpu-health:        healthy | degraded | unknown
//	theta.dev/gpus-unschedulable: number of GPUs the agent marked unfit (0..N)
//
// Design choices, deliberately simple and defensible:
//   - Poll loop (default 30s) rather than informers/watches: node-health
//     changes on thermal timescales (minutes), so a resync-style poll is
//     appropriate and far easier to reason about than event-driven caching.
//   - No CRDs: the agent's metrics are already the source of truth; this
//     controller only projects them into the node API.
//   - Labels only — never cordon/taint. Observing is safe to run everywhere;
//     acting is a policy decision left to operators (Theta's design principle:
//     diagnose autonomously, act only behind an explicit gate).
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

const (
	labelHealth        = "theta.dev/gpu-health"
	labelUnschedulable = "theta.dev/gpus-unschedulable"

	healthHealthy  = "healthy"
	healthDegraded = "degraded"
	healthUnknown  = "unknown"
)

func main() {
	var (
		interval    = flag.Duration("interval", 30*time.Second, "reconcile interval")
		namespace   = flag.String("namespace", "", "namespace of Theta agent pods (empty = all)")
		podSelector = flag.String("selector", "app.kubernetes.io/name=theta", "label selector for Theta agent pods")
		metricsPort = flag.Int("metrics-port", 9101, "Theta agent Prometheus port")
		kubeconfig  = flag.String("kubeconfig", "", "path to kubeconfig (empty = in-cluster)")
	)
	flag.Parse()

	client, err := buildClient(*kubeconfig)
	if err != nil {
		log.Fatalf("kubernetes client: %v", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	log.Printf("theta-node-labeler starting: interval=%s selector=%q port=%d",
		*interval, *podSelector, *metricsPort)

	// Reconcile immediately on startup, then on every tick.
	reconcile(ctx, client, *namespace, *podSelector, *metricsPort)
	ticker := time.NewTicker(*interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			log.Println("shutting down")
			return
		case <-ticker.C:
			reconcile(ctx, client, *namespace, *podSelector, *metricsPort)
		}
	}
}

// buildClient prefers in-cluster config (the normal deployment) and falls
// back to a kubeconfig path for local development (`go run . -kubeconfig ...`).
func buildClient(kubeconfigPath string) (*kubernetes.Clientset, error) {
	cfg, err := rest.InClusterConfig()
	if err != nil {
		if kubeconfigPath == "" {
			kubeconfigPath = os.Getenv("KUBECONFIG")
		}
		cfg, err = clientcmd.BuildConfigFromFlags("", kubeconfigPath)
		if err != nil {
			return nil, fmt.Errorf("neither in-cluster nor kubeconfig config available: %w", err)
		}
	}
	return kubernetes.NewForConfig(cfg)
}

// reconcile performs one observe→measure→patch pass. Errors on individual
// pods/nodes are logged and skipped — one bad node must never block the rest
// of the fleet (partial progress beats all-or-nothing in a control loop).
func reconcile(ctx context.Context, client *kubernetes.Clientset, namespace, selector string, port int) {
	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: selector,
		FieldSelector: "status.phase=Running",
	})
	if err != nil {
		log.Printf("list pods: %v", err)
		return
	}

	for _, pod := range pods.Items {
		node := pod.Spec.NodeName
		if node == "" || pod.Status.PodIP == "" {
			continue
		}

		health, unfit := healthUnknown, -1
		metrics, err := scrape(ctx, fmt.Sprintf("http://%s:%d/metrics", pod.Status.PodIP, port))
		if err != nil {
			log.Printf("node %s: scrape %s: %v", node, pod.Name, err)
		} else {
			unfit = countUnschedulable(metrics)
			if unfit == 0 {
				health = healthHealthy
			} else if unfit > 0 {
				health = healthDegraded
			}
		}

		if err := patchNodeLabels(ctx, client, node, health, unfit); err != nil {
			log.Printf("node %s: patch: %v", node, err)
			continue
		}
		log.Printf("node %s: %s=%s %s=%d", node, labelHealth, health, labelUnschedulable, unfit)
	}
}

// scrape GETs a Prometheus text-format endpoint with a short timeout.
func scrape(ctx context.Context, url string) (string, error) {
	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20)) // 1 MiB cap
	return string(body), err
}

// countUnschedulable parses Prometheus text format looking for
//
//	theta_gpu_schedulable{gpu_index="N",...} 0|1
//
// and returns how many GPUs report 0 (unfit for new work). A hand-rolled
// parse of one known series keeps the binary dependency-light; swap in
// prometheus/common/expfmt if the series set ever grows.
func countUnschedulable(metrics string) int {
	unfit := 0
	for _, line := range strings.Split(metrics, "\n") {
		if !strings.HasPrefix(line, "theta_gpu_schedulable") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		v, err := strconv.ParseFloat(fields[len(fields)-1], 64)
		if err == nil && v == 0 {
			unfit++
		}
	}
	return unfit
}

// patchNodeLabels applies the two labels with a strategic merge patch —
// minimal, idempotent, and safe against concurrent node updates (no
// read-modify-write race, unlike Get + Update).
func patchNodeLabels(ctx context.Context, client *kubernetes.Clientset, node, health string, unfit int) error {
	unfitStr := "unknown"
	if unfit >= 0 {
		unfitStr = strconv.Itoa(unfit)
	}
	patch := fmt.Sprintf(
		`{"metadata":{"labels":{%q:%q,%q:%q}}}`,
		labelHealth, health, labelUnschedulable, unfitStr,
	)
	_, err := client.CoreV1().Nodes().Patch(
		ctx, node, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{},
	)
	return err
}
