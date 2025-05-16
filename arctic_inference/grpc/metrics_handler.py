import http.server
import requests
import threading
import logging
from typing import Any, List
import os
from prometheus_client import start_http_server  # type: ignore


_manager_metric_port = int(os.environ.get("CORVO_VLLMD_INTERNAL_METRIC_PORT", 8102))


# Append replica=<replica> to all metrics
def _add_replica_to_metrics(metrics: str, replica: int) -> List[str]:
    lines = metrics.splitlines()
    modified_lines = []
    for line in lines:
        if line.startswith("#"):
            modified_lines.append(line)
        else:
            parts = line.split()
            name = parts[0]
            value = parts[1]
            labels = {}
            if "{" in name:
                name, labels_str = name.split("{")
                labels_str = labels_str.strip("}").split(",")
                labels = {
                    label.split("=")[0]: label.split("=")[1].strip('"')
                    for label in labels_str
                }
            labels["replica"] = str(replica)
            labels_str = ",".join(f'{key}="{value}"' for key, value in labels.items())  # type: ignore
            modified_line = f"{name}{{{labels_str}}} {value}"
            modified_lines.append(modified_line)
    return modified_lines


# Deduplicate # HELP and # TYPE comments otherwise Prometheus will not able to parse the metrics.
def _deduplicate_comments(metrics: List[str]) -> List[str]:
    help_map: dict[str, str] = {}
    type_map: dict[str, str] = {}
    metrics_map: dict[str, List[str]] = {}

    for line in metrics:
        if line.startswith("# HELP"):
            metric_name = line.split()[2]
            if metric_name not in help_map:
                help_map[metric_name] = line
        elif line.startswith("# TYPE"):
            metric_name = line.split()[2]
            if metric_name not in type_map:
                type_map[metric_name] = line
        else:
            metric_name = line.split("{")[0] if "{" in line else line.split()[0]
            if metric_name not in metrics_map:
                metrics_map[metric_name] = [line]
            else:
                metrics_map[metric_name].append(line)

    metrics = []
    for metric in metrics_map:
        if metric in help_map:
            metrics.append(help_map[metric])
        if metric in type_map:
            metrics.append(type_map[metric])
        metrics.extend(metrics_map[metric])
    return metrics


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    ports: list[int] = []
    ports_lock = threading.Lock()

    def do_GET(self) -> None:
        if self.path != "/metrics" and self.path != "/":
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")
            return
        with self.ports_lock:
            ports = self.ports[:] + [_manager_metric_port]
        modified_metrics = []
        for i, port in enumerate(ports):
            url = f"http://localhost:{port}/metrics"
            try:
                response = requests.get(url)
                response.raise_for_status()
            except (requests.exceptions.RequestException, OSError) as e:
                logging.info(f"Error fetching metrics from {url}: {e}")
                continue
            metrics = response.text
            modified_metrics.extend(_add_replica_to_metrics(metrics, i))

        if len(ports) > 1:
            modified_metrics = _deduplicate_comments(modified_metrics)
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        try:
            # Prometheus requires metrics to be ended with a newline.
            self.wfile.write(("\n".join(modified_metrics) + "\n").encode())
        except BrokenPipeError:
            logging.info("Client disconnected.")

    def log_message(self, format: str, *args: Any) -> None:
        # Do nothing, effectively disabling logging
        pass


def _run_server(port: int) -> None:
    server_address = ("", port)
    httpd = http.server.HTTPServer(server_address, _MetricsHandler)
    logging.info(f"Starting httpd on port {port}...")
    httpd.serve_forever()


def set_ports(ports: int | list[int]) -> None:
    """
    Sets the list of ports to forward requests to when scraping metrics.

    Args:
        ports (list): A list of ports to forward requests to.
    """
    if isinstance(ports, int):
        ports = [ports]
    with _MetricsHandler.ports_lock:
        _MetricsHandler.ports = ports


def run_server(port: int) -> None:
    """
    Runs an HTTP server that forwards requests to the specified ports
    to scrape and aggregate metrics.
    set_ports must be called before running the server.
    The metrics would have an addition label as replica=<replica_number>
    to differentiate between replicas.

    Args:
        port (int): The port to run the server on.
    """
    if port == _manager_metric_port:
        raise ValueError(
            f"Cannot start metrics server on the same port as the manager metrics server: {port}, try setting CORVO_VLLMD_INTERNAL_METRIC_PORT to a different port."
        )
    logging.info(f"Starting manager metrics server on port {_manager_metric_port}")
    start_http_server(port=_manager_metric_port)
    server_thread = threading.Thread(target=_run_server, args=(port,))
    server_thread.daemon = True  # Set as daemon so that it exits when main thread exits
    server_thread.start()
