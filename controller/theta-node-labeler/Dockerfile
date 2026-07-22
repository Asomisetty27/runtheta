# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM golang:1.26-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY main.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /theta-node-labeler .

# ── Stage 2: runtime (distroless: no shell, no package manager) ──────────────
FROM gcr.io/distroless/static-debian12:nonroot
LABEL org.opencontainers.image.title="theta-node-labeler"
LABEL org.opencontainers.image.description="Reconciles Kubernetes node labels from Theta GPU-health telemetry"
LABEL org.opencontainers.image.source="https://github.com/Asomisetty27/theta"
COPY --from=builder /theta-node-labeler /theta-node-labeler
USER nonroot:nonroot
ENTRYPOINT ["/theta-node-labeler"]
