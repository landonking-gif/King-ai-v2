# Kubernetes Deployment Guide

This guide provides baseline Kubernetes manifests for Mother-Harness, plus homelab multi-node deployment steps that mirror the 4-node layout in `project documents/Todo.md`.

## Contents
- `docs/kubernetes/` contains core manifests for namespaces, secrets, storage, and services.
- Horizontal Pod Autoscalers (HPA) are defined for the orchestrator and agent workers.

## Prerequisites
- Kubernetes cluster (v1.25+ recommended).
- `kubectl` configured for the cluster.
- Storage class that can satisfy:
  - `ReadWriteOnce` for Redis and n8n.
  - `ReadWriteMany` for shared libraries (or provide a dedicated RWX volume).
- Metrics Server for HPA (`kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`).
- Built container images for orchestrator, docling, dashboard, and agent workers.

## Build and Push Images
The manifests reference the following images (replace with your registry):
- `mother-harness/orchestrator:latest`
- `mother-harness/docling:latest`
- `mother-harness/dashboard:latest`
- `mother-harness/agent-workers:latest`

Use your CI or local Docker builds to push them to a registry accessible by the cluster.

## Configure Secrets and Storage
1. Update `docs/kubernetes/secret.yaml` with real values for:
   - `REDIS_PASSWORD` and `REDIS_URL`
   - `N8N_API_KEY`, `N8N_PASSWORD`
   - `OLLAMA_CLOUD_API_KEY` (if used)
2. Adjust PVC sizes and storage class settings in `docs/kubernetes/storage.yaml` and `docs/kubernetes/redis-stack.yaml`.

## Apply Manifests
Apply the manifests in order:

```bash
kubectl apply -f docs/kubernetes/namespace.yaml
kubectl apply -f docs/kubernetes/secret.yaml
kubectl apply -f docs/kubernetes/configmap.yaml
kubectl apply -f docs/kubernetes/storage.yaml
kubectl apply -f docs/kubernetes/redis-stack.yaml
kubectl apply -f docs/kubernetes/orchestrator.yaml
kubectl apply -f docs/kubernetes/agent-workers.yaml
kubectl apply -f docs/kubernetes/docling.yaml
kubectl apply -f docs/kubernetes/n8n.yaml
kubectl apply -f docs/kubernetes/dashboard.yaml
```

## Horizontal Scaling
- `docs/kubernetes/orchestrator.yaml` defines an HPA to scale `orchestrator` between 2 and 8 replicas.
- `docs/kubernetes/agent-workers.yaml` defines an HPA to scale agent workers between 3 and 12 replicas.

Adjust CPU requests/limits or HPA thresholds to fit your hardware capacity.

## Multi-Node Homelab Deployment (4-Node Layout)
The `Todo.md` plan describes a 4-node homelab. Use node labels and (optional) affinity rules to pin workloads:

### 1. Label the Nodes
```bash
kubectl label node core1 mother-harness/node-role=core1
kubectl label node core2 mother-harness/node-role=core2
kubectl label node core3 mother-harness/node-role=core3
kubectl label node core4 mother-harness/node-role=core4
```

### 2. Place Core Services on Core1
Core1 hosts primary compute (orchestrator, Redis Stack, docling, dashboard, n8n). Add a `nodeSelector` or `nodeAffinity` to the following manifests:
- `docs/kubernetes/redis-stack.yaml`
- `docs/kubernetes/orchestrator.yaml`
- `docs/kubernetes/docling.yaml`
- `docs/kubernetes/n8n.yaml`
- `docs/kubernetes/dashboard.yaml`

Example snippet to add under `spec.template.spec`:

```yaml
nodeSelector:
  mother-harness/node-role: core1
```

### 3. Place Agent Workers on Core3
Core3 is intended for overflow compute and additional agent replicas. Add the same node selector to `docs/kubernetes/agent-workers.yaml`:

```yaml
nodeSelector:
  mother-harness/node-role: core3
```

### 4. Configure Ollama on Core2
Ollama runs outside this manifest set. Ensure it is reachable at the `OLLAMA_LOCAL_URL` defined in `docs/kubernetes/configmap.yaml`. If you deploy Ollama on Kubernetes, target core2 with a node selector or taints.

### 5. Attach Storage from Core4
Core4 is the storage node. Provide a `ReadWriteMany` volume (NFS/SMB/CSI) for `libraries-data` and ensure it is backed by the `/core4/libraries` share. Update `docs/kubernetes/storage.yaml` with the correct storage class or use a manually provisioned PV.

### 6. Validate Connectivity
- Orchestrator should reach Redis (`redis-stack:6379`) and n8n (`n8n:5678`).
- Docling should mount `/mnt/libraries` from the shared volume.
- Dashboard should resolve `orchestrator:8000` in-cluster.

## Notes
- These manifests are a starting point and assume in-cluster networking.
- Update image names, resource requests, and ingress as needed for your environment.
