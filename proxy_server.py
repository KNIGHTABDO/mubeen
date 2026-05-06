from fastapi import FastAPI, Request, Response, HTTPException
import httpx
import uvicorn
import os
import re
from urllib.parse import urljoin, quote, unquote, urlparse

app = FastAPI()

# Simple token for security
PROXY_TOKEN = "mubeen_proxy_secure_7788"

def rewrite_m3u8(content: str, base_url: str, proxy_base: str, token: str) -> str:
    lines = content.splitlines()
    new_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("#"):
            if 'URI="' in line:
                def replace_uri(match):
                    uri = match.group(1)
                    full_uri = urljoin(base_url, uri)
                    proxied_uri = f"{proxy_base}?token={token}&url={quote(full_uri)}"
                    return f'URI="{proxied_uri}"'
                line = re.sub(r'URI="([^"]+)"', replace_uri, line)
            new_lines.append(line)
        else:
            full_url = urljoin(base_url, line)
            proxied_url = f"{proxy_base}?token={token}&url={quote(full_url)}"
            new_lines.append(proxied_url)
            
    return "\n".join(new_lines)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.api_route("/proxy", methods=["GET", "POST", "OPTIONS"])
async def proxy(request: Request):
    if request.method == "OPTIONS":
        return Response(status_code=200, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        })

    token = request.query_params.get("token") or request.headers.get("X-Proxy-Token")
    if token != PROXY_TOKEN:
        return Response(content="Unauthorized", status_code=403)

    url = request.query_params.get("url")
    if not url:
        return Response(content="Missing url parameter", status_code=400)
    
    url = unquote(url)
    parsed_url = urlparse(url)

    # Filter headers - keep essential ones
    excluded_headers = ["host", "x-proxy-token", "connection", "origin", "referer", "x-vercel-id", "x-forwarded-for", "x-real-ip"]
    headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded_headers}
    
    # IMPORTANT: IP Spoofing
    # We take the client IP passed from Vercel and tell StreamRuby it's the real requester
    client_ip = request.headers.get("X-Forwarded-For") or request.headers.get("X-Real-IP")
    if client_ip:
        headers["X-Forwarded-For"] = client_ip
        headers["X-Real-IP"] = client_ip
        headers["Client-IP"] = client_ip
        headers["True-Client-IP"] = client_ip

    # Spoof Referer based on destination
    if "streamruby" in url or "egydead" in url:
        headers["Referer"] = f"https://egydead.live/"
    
    if "User-Agent" not in headers:
        headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, verify=False) as client:
        try:
            method = request.method
            body = await request.body()
            
            resp = await client.request(
                method=method,
                url=url,
                headers=headers,
                content=body if method == "POST" else None
            )
            
            content_type = resp.headers.get("content-type", "")
            is_m3u8 = "application/vnd.apple.mpegurl" in content_type.lower() or "application/x-mpegurl" in content_type.lower() or url.split('?')[0].endswith(".m3u8")

            # Clean response headers
            resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in ["content-encoding", "content-length", "transfer-encoding", "connection", "access-control-allow-origin"]}
            resp_headers["Access-Control-Allow-Origin"] = "*"
            
            if is_m3u8:
                proxy_base = str(request.url).split('?')[0]
                rewritten = rewrite_m3u8(resp.text, url, proxy_base, PROXY_TOKEN)
                return Response(content=rewritten, status_code=resp.status_code, headers=resp_headers)
            
            return Response(content=resp.content, status_code=resp.status_code, headers=resp_headers)
        except Exception as e:
            return Response(content=str(e), status_code=500, headers={"Access-Control-Allow-Origin": "*"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
