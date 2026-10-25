"""Start the BoneVision local web app.

    python serve.py                       # uses models/ssm_highres.npz
    python serve.py --demo                # builds/uses a SYNTHETIC demo model (UI testing only)
    python serve.py --model path/to/ssm.npz --device cuda --port 8000

Open http://127.0.0.1:8000. Uploads and results stay in this process's memory.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO_MODEL = ROOT / "models" / "ssm_demo_synthetic.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=None, help="Shape model .npz (default models/ssm_highres.npz)")
    ap.add_argument("--demo", action="store_true", help="Use the synthetic demo model, building it if missing")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--host", default="127.0.0.1", help="Keep 127.0.0.1 unless you need LAN access")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development)")
    args = ap.parse_args()
    if args.demo:
        if not DEMO_MODEL.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_model.py"), "--output", str(DEMO_MODEL)], check=True)
        args.model = DEMO_MODEL
    if args.model is not None:
        os.environ["BONEVISION_MODEL"] = str(args.model.resolve())
    os.environ["BONEVISION_DEVICE"] = args.device
    import uvicorn
    print(f"BoneVision → http://{args.host}:{args.port}")
    uvicorn.run("bonevision.server:app", host=args.host, port=args.port, reload=args.reload, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
