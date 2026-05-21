from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import ssl
import os
import json
from pathlib import Path
from urllib.parse import urlparse
import yaml


def _resolve_workspace_path(path_value: str, workspace_root: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((workspace_root / path).resolve())


def _build_runtime_config_js(config: dict) -> str:
    client_config = {
        "websocket": config.get("websocket", {}),
        "audio": config.get("audio", {}),
    }
    #return f"const CLIENT_CONFIG = {json.dumps(client_config, ensure_ascii=False, indent=2)};\n"
    return f"window.CLIENT_CONFIG = {json.dumps(client_config, ensure_ascii=False, indent=2)};\n"


class ConfigAwareHandler(SimpleHTTPRequestHandler):
    #runtime_config_js = "const CLIENT_CONFIG = {};\n"
    runtime_config_js = "window.CLIENT_CONFIG = {};\n"

    def do_GET(self):
        request_path = urlparse(self.path).path
        if request_path == "/runtime-config.js":
            body = self.runtime_config_js.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

def main():
    current_dir = Path(__file__).resolve().parent
    workspace_root = current_dir.parent
    os.chdir(current_dir)

    config_path = current_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ConfigAwareHandler.runtime_config_js = _build_runtime_config_js(config)

    client_config = config.get("client") or config.get("server")
    if client_config is None:
        raise KeyError("config.yaml must include 'client' (preferred) or legacy 'server' section")

    cert_path = _resolve_workspace_path(client_config["cert_path"], workspace_root)
    key_path = _resolve_workspace_path(client_config["key_path"], workspace_root)
    server_address = (client_config["host"], client_config["port"])

    httpd = ThreadingHTTPServer(
        server_address,
        ConfigAwareHandler
    )

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"🔒 Sample HTTPS running on {client_config['protocol']}://{client_config['host']}:{client_config['port']}")
    print(f"Config file: {str(config_path)}")
    print(f"Using cert: {cert_path}")
    print(f"Using key : {key_path}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()