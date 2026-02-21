# Colab Guide: Run MangaAnimator with Flask + ngrok (Public URL)

This guide shows how to run the project in **Google Colab**, start the Flask app, and expose it securely using **ngrok**.

## 1) Start a Colab runtime

- In Colab: `Runtime -> Change runtime type`
- Prefer:
  - **GPU** (T4/L4/A100)
  - High-RAM if available

## 2) Clone repository

```bash
!git clone <YOUR_REPO_URL>
%cd MangaAnimator
```

## 3) Install dependencies + models

Run full setup (deps + auto model download from registry):

```bash
!DOWNLOAD_PROFILE=max_quality DOWNLOAD_REPOS=1 DOWNLOAD_STRICT=1 bash scripts/install_colab.sh
```

Optional behavior:

- Skip model downloads:

```bash
!DOWNLOAD_MODELS=0 bash scripts/install_colab.sh
```

- Use HF token for gated repos:

```bash
import os
os.environ["HF_TOKEN"] = "hf_xxx"
!DOWNLOAD_PROFILE=max_quality DOWNLOAD_REPOS=1 DOWNLOAD_STRICT=1 bash scripts/install_colab.sh
```

## 4) Configure Flask app host/port

The app should bind to `0.0.0.0` so ngrok can tunnel it.

If needed, set:

```python
import os
os.environ["FLASK_HOST"] = "0.0.0.0"
os.environ["FLASK_PORT"] = "5000"
```

## 5) Install ngrok in Colab

```bash
!pip install -q pyngrok
```

## 6) Set ngrok auth token

Create your token at https://dashboard.ngrok.com/get-started/your-authtoken.

```python
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
```

## 7) Start Flask app in background

Use a background process so notebook cells remain interactive:

```python
import os, subprocess, time

os.environ["FLASK_HOST"] = "0.0.0.0"
os.environ["FLASK_PORT"] = "5000"

proc = subprocess.Popen(
    ["python", "app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

time.sleep(5)
print("Flask process started")
```

## 8) Open ngrok tunnel to Flask port

```python
from pyngrok import ngrok

public_url = ngrok.connect(addr=5000, proto="http")
print("Public URL:", public_url)
```

Now use the printed URL in your browser.

## 9) Health checks and debugging

### Check local endpoint

```bash
!curl -I http://127.0.0.1:5000
```

### Stream Flask logs

```python
for _ in range(60):
    line = proc.stdout.readline()
    if not line:
        break
    print(line, end="")
```

### Common issues

1. **502/Bad Gateway from ngrok**
   - Flask not started or wrong port.
   - Ensure `app.run(host="0.0.0.0", port=5000)` behavior.

2. **Tunnel opens but app unreachable**
   - Port mismatch (ngrok connects to 5000, app uses other port).
   - Fix env vars and restart app/tunnel.

3. **Model loading too slow/OOM**
   - Use `DOWNLOAD_MODELS=0` for quick UI boot.
   - Reduce quality profile/config or move to stronger Colab GPU.

4. **ngrok token/auth errors**
   - Recheck token and call `ngrok.set_auth_token(...)` before `ngrok.connect(...)`.

## 10) Recommended Colab workflow

1. Install: `!DOWNLOAD_PROFILE=max_quality DOWNLOAD_REPOS=1 DOWNLOAD_STRICT=1 bash scripts/install_colab.sh`
2. Start Flask in background (`python app.py`)
3. Create ngrok tunnel (`ngrok.connect(5000)`)
4. Open public URL
5. Run jobs through UI/API

## 11) Shutdown cleanly

```python
from pyngrok import ngrok
ngrok.kill()

proc.terminate()
print("Stopped Flask and ngrok")
```

---

## Minimal one-cell bootstrap (optional)

```python
!pip install -q pyngrok
import os, subprocess, time
from pyngrok import ngrok

os.environ["HF_TOKEN"] = "hf_xxx"  # optional
os.environ["FLASK_HOST"] = "0.0.0.0"
os.environ["FLASK_PORT"] = "5000"

!DOWNLOAD_PROFILE=max_quality DOWNLOAD_REPOS=1 DOWNLOAD_STRICT=1 bash scripts/install_colab.sh

ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
proc = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(5)
url = ngrok.connect(5000, "http")
print("Open:", url)
```


## 12) Confirm GPU/VRAM usage logs

The pipeline now logs compute snapshots (device name + used/free VRAM) at startup and each stage.

```bash
!python -m src.orchestration.run_all --input /content/panel.png --workdir outputs/full_pipeline --config configs/default.yaml
```

Watch for log lines containing:
- `Compute detected: device=cuda ...`
- `Running stage ... used_vram=... free_vram=...`
- `GPU warmup result: {'warmup': True ...}`
