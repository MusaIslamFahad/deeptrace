#!/usr/bin/env python3
"""
scripts/test_deployment.py
Verifies your Render + Vercel deployment is working correctly.

Usage:
    python scripts/test_deployment.py \
        --api https://deeptrace-api.onrender.com \
        --frontend https://deeptrace.vercel.app \
        --key deeptrace-demo-key
"""

import sys
import time
import argparse
import urllib.request
import urllib.error
import json
import io

# Use only stdlib — no pip install needed
try:
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError
except ImportError:
    print("Python 3.6+ required")
    sys.exit(1)


def check(label: str, ok: bool, detail: str = ""):
    icon = "✅" if ok else "❌"
    msg = f"  {icon}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return ok


def get_json(url: str, headers: dict = None, timeout: int = 30) -> dict:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def test_api(api_base: str, api_key: str) -> bool:
    print(f"\n{'='*50}")
    print(f"  Testing API: {api_base}")
    print(f"{'='*50}")

    all_passed = True

    # 1. Root endpoint
    try:
        data = get_json(f"{api_base}/")
        all_passed &= check("Root endpoint /",
                            "service" in data,
                            data.get("service", ""))
    except Exception as e:
        all_passed &= check("Root endpoint /", False, str(e))

    # 2. Health check
    try:
        data = get_json(f"{api_base}/health")
        status = data.get("status", "unknown")
        all_passed &= check(f"Health check /health",
                            status in ("healthy", "degraded"),
                            f"status={status}")
        if status == "degraded":
            print("     ⚠️  Status is 'degraded' — Redis may not be connected yet")
    except Exception as e:
        all_passed &= check("Health check /health", False, str(e))

    # 3. Auth rejection (no key)
    try:
        req = Request(f"{api_base}/api/v1/predict", method="POST")
        req.add_header("Content-Type", "multipart/form-data; boundary=test")
        try:
            urlopen(req, timeout=10)
            all_passed &= check("Auth rejection (no key)", False, "Should have returned 401")
        except HTTPError as e:
            all_passed &= check("Auth rejection (no key)", e.code == 401, f"HTTP {e.code}")
    except Exception as e:
        # Connection error is expected if API is asleep
        check("Auth rejection (no key)", False, str(e))

    # 4. Predict with a tiny 1x1 test image
    try:
        import struct, zlib

        def make_tiny_png() -> bytes:
            """Create a minimal valid 1x1 white PNG in pure Python."""
            def chunk(name: bytes, data: bytes) -> bytes:
                c = struct.pack(">I", len(data)) + name + data
                return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

            png  = b"\x89PNG\r\n\x1a\n"
            png += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            raw  = b"\x00\xff\xff\xff"
            png += chunk(b"IDAT", zlib.compress(raw))
            png += chunk(b"IEND", b"")
            return png

        img_bytes = make_tiny_png()
        boundary = b"----DeepTraceBoundary"
        body = (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
            b"Content-Type: image/png\r\n\r\n"
            + img_bytes + b"\r\n"
            b"--" + boundary + b"--\r\n"
        )

        req = Request(
            f"{api_base}/api/v1/predict",
            data=body,
            method="POST",
            headers={
                "X-API-Key": api_key,
                "Content-Type": f"multipart/form-data; boundary={boundary.decode()}",
            }
        )
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        has_fields = all(
            k in data for k in
            ["image_id", "predicted_source", "confidence", "per_class_probs"]
        )
        prob_sum = sum(data.get("per_class_probs", {}).values())

        all_passed &= check(
            "POST /api/v1/predict (with test image)",
            has_fields and abs(prob_sum - 1.0) < 0.01,
            f"source={data.get('predicted_source')} conf={data.get('confidence', 0):.2f}"
        )

        if "demo" in data.get("model_version", ""):
            print("     ℹ️  Demo mode active — mock predictions (expected on free tier)")

    except Exception as e:
        all_passed &= check("POST /api/v1/predict", False, str(e))

    # 5. OpenAPI docs
    try:
        req = Request(f"{api_base}/docs")
        with urlopen(req, timeout=15) as resp:
            html = resp.read().decode()
        all_passed &= check("API docs /docs", "swagger" in html.lower() or "openapi" in html.lower())
    except Exception as e:
        all_passed &= check("API docs /docs", False, str(e))

    # 6. Prometheus metrics
    try:
        req = Request(f"{api_base}/metrics")
        with urlopen(req, timeout=10) as resp:
            text = resp.read().decode()
        all_passed &= check("Prometheus /metrics", len(text) > 0, f"{len(text)} bytes")
    except Exception as e:
        all_passed &= check("Prometheus /metrics", False, str(e))

    return all_passed


def test_frontend(frontend_url: str) -> bool:
    print(f"\n{'='*50}")
    print(f"  Testing Frontend: {frontend_url}")
    print(f"{'='*50}")

    all_passed = True

    # Homepage
    try:
        with urlopen(frontend_url, timeout=30) as resp:
            html = resp.read().decode()
        has_root = '<div id="root">' in html or "DeepTrace" in html
        all_passed &= check("Frontend homepage loads", has_root)
    except Exception as e:
        all_passed &= check("Frontend homepage loads", False, str(e))

    # SPA routing (should return 200 not 404)
    for path in ["/analytics", "/history"]:
        try:
            with urlopen(f"{frontend_url}{path}", timeout=15) as resp:
                all_passed &= check(f"SPA route {path}", resp.status == 200)
        except HTTPError as e:
            all_passed &= check(f"SPA route {path}", False, f"HTTP {e.code}")
        except Exception as e:
            all_passed &= check(f"SPA route {path}", False, str(e))

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test DeepTrace deployment")
    parser.add_argument("--api",      required=True,
                        help="Render API URL e.g. https://deeptrace-api.onrender.com")
    parser.add_argument("--frontend", default=None,
                        help="Vercel frontend URL e.g. https://deeptrace.vercel.app")
    parser.add_argument("--key",      default="deeptrace-demo-key",
                        help="API key")
    parser.add_argument("--wake",     action="store_true",
                        help="Wake up the Render service first (sends health ping)")
    args = parser.parse_args()

    print("\n🔍 DeepTrace Deployment Test")
    print(f"   API:      {args.api}")
    if args.frontend:
        print(f"   Frontend: {args.frontend}")
    print(f"   API key:  {args.key}")

    if args.wake:
        print(f"\n⏳ Waking up Render service (may take 30s)...")
        for i in range(6):
            try:
                get_json(f"{args.api}/health", timeout=10)
                print("   Service is awake!")
                break
            except Exception:
                print(f"   Waiting... ({(i+1)*5}s)")
                time.sleep(5)

    api_ok = test_api(args.api, args.key)
    frontend_ok = test_frontend(args.frontend) if args.frontend else True

    print(f"\n{'='*50}")
    if api_ok and frontend_ok:
        print("  🎉 All checks passed! DeepTrace is live.")
        print(f"\n  Share these links:")
        if args.frontend:
            print(f"    Dashboard : {args.frontend}")
        print(f"    API docs  : {args.api}/docs")
        print(f"    Health    : {args.api}/health")
    else:
        print("  ⚠️  Some checks failed — see above for details.")
        print("  Check the DEPLOYMENT_GUIDE.md troubleshooting section.")
    print(f"{'='*50}\n")

    sys.exit(0 if (api_ok and frontend_ok) else 1)


if __name__ == "__main__":
    main()
