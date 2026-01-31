#!/usr/bin/env python3
"""
RouteLLM API - Intelligent Model Routing Service
Built by SamTheArchitect

A simple HTTP API that routes requests to the optimal AI model.
Deploy on Render, Railway, or any Python host.
"""

import json
import re
import os
import hashlib
import hmac
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from datetime import datetime
from typing import Dict, Optional, Tuple, List

# ============== Model Router Logic ==============

MODELS = {
    "gpt-4o": {"provider": "openai", "input_cost": 0.005, "output_cost": 0.015, "quality": 0.95},
    "gpt-4o-mini": {"provider": "openai", "input_cost": 0.00015, "output_cost": 0.0006, "quality": 0.85},
    "claude-opus-4": {"provider": "anthropic", "input_cost": 0.015, "output_cost": 0.075, "quality": 0.98},
    "claude-sonnet-4": {"provider": "anthropic", "input_cost": 0.003, "output_cost": 0.015, "quality": 0.92},
    "claude-haiku-4": {"provider": "anthropic", "input_cost": 0.0008, "output_cost": 0.004, "quality": 0.82},
    "gemini-2.0-flash": {"provider": "google", "input_cost": 0.0001, "output_cost": 0.0004, "quality": 0.80},
    "gemini-1.5-pro": {"provider": "google", "input_cost": 0.00125, "output_cost": 0.005, "quality": 0.90},
}

COMPLEXITY_PATTERNS = {
    "simple": [r'\b(what is|define|translate|list \d+|yes or no)\b', r'^.{0,100}$'],
    "complex": [r'\b(analyze|debug|review code|architect|step by step)\b'],
    "expert": [r'\b(security audit|mathematical proof|formal verification)\b'],
}

def analyze_complexity(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for level in ["expert", "complex", "simple"]:
        for pattern in COMPLEXITY_PATTERNS.get(level, []):
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return level
    return "moderate"

def route_request(prompt: str, quality: str = "auto", max_budget: float = None) -> Dict:
    complexity = analyze_complexity(prompt)
    
    quality_thresholds = {"low": 0.7, "medium": 0.82, "high": 0.90}
    auto_thresholds = {"simple": 0.7, "moderate": 0.82, "complex": 0.88, "expert": 0.92}
    
    min_quality = auto_thresholds.get(complexity, 0.82) if quality == "auto" else quality_thresholds.get(quality, 0.82)
    
    # Find cheapest model that meets quality threshold
    candidates = []
    for name, model in MODELS.items():
        if model["quality"] >= min_quality:
            candidates.append((name, model["input_cost"], model["quality"]))
    
    if not candidates:
        candidates = [("gpt-4o-mini", 0.00015, 0.85)]
    
    # Sort by cost
    candidates.sort(key=lambda x: x[1])
    best = candidates[0]
    
    # Estimate cost
    input_tokens = len(prompt) // 4 + 50
    output_tokens = 500
    model_data = MODELS.get(best[0], MODELS["gpt-4o-mini"])
    estimated_cost = (input_tokens / 1000 * model_data["input_cost"]) + (output_tokens / 1000 * model_data["output_cost"])
    
    return {
        "model": best[0],
        "provider": model_data["provider"],
        "complexity": complexity,
        "quality_score": model_data["quality"],
        "estimated_cost_usd": round(estimated_cost, 6),
        "alternatives": [c[0] for c in candidates[1:3]]
    }

# ============== API Keys (In-Memory for Demo) ==============

API_KEYS = {}  # key -> {email, created, requests, plan}
FREE_LIMIT = 1000

def generate_api_key(email: str) -> str:
    """Generate a new API key"""
    raw = f"{email}:{datetime.utcnow().isoformat()}:{os.urandom(16).hex()}"
    key = f"rtllm_{hashlib.sha256(raw.encode()).hexdigest()[:32]}"
    API_KEYS[key] = {
        "email": email,
        "created": datetime.utcnow().isoformat(),
        "requests": 0,
        "plan": "free"
    }
    return key

def validate_api_key(key: str) -> Tuple[bool, Optional[Dict]]:
    """Validate API key and check limits"""
    if not key or not key.startswith("rtllm_"):
        return False, None
    
    if key not in API_KEYS:
        # For demo, accept any rtllm_ key
        API_KEYS[key] = {"email": "demo", "created": datetime.utcnow().isoformat(), "requests": 0, "plan": "free"}
    
    data = API_KEYS[key]
    if data["plan"] == "free" and data["requests"] >= FREE_LIMIT:
        return False, {"error": "Free limit exceeded", "limit": FREE_LIMIT}
    
    return True, data

# ============== HTTP Handler ==============

class APIHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _get_api_key(self) -> Optional[str]:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        return None
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == "/" or path == "/health":
            self._send_json({"status": "ok", "service": "RouteLLM", "version": "1.0.0"})
        
        elif path == "/v1/models":
            self._send_json({"models": list(MODELS.keys())})
        
        elif path == "/v1/stats":
            api_key = self._get_api_key()
            if not api_key:
                self._send_json({"error": "API key required"}, 401)
                return
            
            valid, data = validate_api_key(api_key)
            if not valid:
                self._send_json(data, 403)
                return
            
            self._send_json({
                "requests_used": data["requests"],
                "plan": data["plan"],
                "limit": FREE_LIMIT if data["plan"] == "free" else "unlimited"
            })
        
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        path = urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode() if content_length else "{}"
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return
        
        if path == "/v1/route":
            api_key = self._get_api_key()
            if not api_key:
                self._send_json({"error": "API key required. Get one at routellm.dev"}, 401)
                return
            
            valid, key_data = validate_api_key(api_key)
            if not valid:
                self._send_json(key_data, 403)
                return
            
            prompt = data.get("prompt", "")
            if not prompt:
                self._send_json({"error": "prompt is required"}, 400)
                return
            
            quality = data.get("quality", "auto")
            max_budget = data.get("max_budget")
            
            result = route_request(prompt, quality, max_budget)
            
            # Increment usage
            API_KEYS[api_key]["requests"] += 1
            result["requests_remaining"] = FREE_LIMIT - API_KEYS[api_key]["requests"] if key_data["plan"] == "free" else "unlimited"
            
            self._send_json(result)
        
        elif path == "/v1/signup":
            email = data.get("email", "")
            if not email or "@" not in email:
                self._send_json({"error": "Valid email required"}, 400)
                return
            
            api_key = generate_api_key(email)
            self._send_json({
                "api_key": api_key,
                "message": "Welcome to RouteLLM! You have 1000 free routes/month.",
                "docs": "https://routellm.dev/docs"
            })
        
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def log_message(self, format, *args):
        print(f"[{datetime.utcnow().isoformat()}] {args[0]}")

def run_server(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), APIHandler)
    print(f"🚀 RouteLLM API running on http://0.0.0.0:{port}")
    print(f"   POST /v1/route - Route a prompt to optimal model")
    print(f"   POST /v1/signup - Get an API key")
    print(f"   GET /v1/models - List available models")
    server.serve_forever()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    run_server(port)
