# Podman GPU Setup (DGX Spark / PGX ThinkStation)

Voraussetzungen damit die Start-Skripte (`start.sh`, `start.mtp.sh` etc.) funktionieren.

## 1. NVIDIA Container Toolkit

```bash
# Repository hinzufuegen
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

## 2. CDI (Container Device Interface) generieren

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

Pruefen:
```bash
nvidia-ctk cdi list
# Erwartete Ausgabe:
#   nvidia.com/gpu=0
#   nvidia.com/gpu=all
```

## 3. Podman Policy

```bash
sudo mkdir -p /etc/containers
sudo tee /etc/containers/policy.json <<'EOF'
{
  "default": [{"type": "insecureAcceptAnything"}],
  "transports": {"docker-daemon": {"": [{"type":"insecureAcceptAnything"}]}}
}
EOF
```

Ohne diese Datei verweigert Podman das Starten von Containern (`policy.json: no`).

## 4. Registries (optional)

```bash
sudo tee /etc/containers/registries.conf <<'EOF'
[registries]
  [registries.search]
    registries = ['docker.io', 'quay.io', 'ghcr.io']
EOF
```

## 5. Pruefung

```bash
# GPU im Container sichtbar?
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi

# Erwartete Ausgabe: GPU-Tabelle mit GB10/RTX PRO 6000
```

## Versionen (getestet)

| Komponente | Version |
|---|---|
| Podman | 4.9.3 |
| nvidia-container-toolkit | 1.18.2 |
| OCI Runtime | crun |
| Driver | 590.48.01 |
| CUDA | 13.0 |

## Podman-Flags fuer GPU-Container

```bash
--device nvidia.com/gpu=all          # GPU via CDI
--security-opt=label=disable         # SELinux/AppArmor deaktivieren
--hooks-dir=/usr/share/containers/oci/hooks.d   # GPU Hooks (Legacy, optional mit CDI)
--network host --ipc=host            # Fuer Multi-Node / NCCL
```
