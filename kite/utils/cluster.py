"""
Cluster Management Utility
Handles node registration, heartbeat, and distributed task signaling using Redis.
"""

import time
import uuid
import json
import threading
from typing import Dict, Any, List, Optional
import redis

class ClusterNode:
    """Represents a node in the Kite cluster."""
    def __init__(self, node_id: str = None, redis_config: Dict = None):
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self.redis_config = redis_config or {"host": "localhost", "port": 6379, "db": 0}
        self.r = redis.Redis(**self.redis_config, decode_responses=True)
        self.is_active = False
        self._heartbeat_thread = None

    def join_cluster(self):
        """Register the node and start heartbeats."""
        try:
            self.is_active = True
            self.r.hset("kite:cluster:nodes", self.node_id, time.time())
            self._heartbeat_thread = threading.Thread(target=self._run_heartbeat, daemon=True)
            self._heartbeat_thread.start()
            print(f"[START] Node {self.node_id} joined the cluster.")
        except redis.exceptions.ConnectionError:
            self.is_active = False
            print(f"[WARN]  Could not connect to Redis. Cluster features disabled for Node {self.node_id}.")

    def leave_cluster(self):
        """Unregister the node."""
        self.is_active = False
        self.r.hdel("kite:cluster:nodes", self.node_id)
        print(f"  Node {self.node_id} left the cluster.")

    def _run_heartbeat(self):
        while self.is_active:
            self.r.hset("kite:cluster:nodes", self.node_id, time.time())
            time.sleep(10)

    def get_active_nodes(self) -> List[str]:
        """List currently healthy nodes."""
        nodes = self.r.hgetall("kite:cluster:nodes")
        now = time.time()
        active = []
        for nid, last_seen in nodes.items():
            if now - float(last_seen) < 30: # 30s timeout
                active.append(nid)
            else:
                self.r.hdel("kite:cluster:nodes", nid)
        return active

    def broadcast_signal(self, signal_type: str, data: Any):
        """Send a signal to all nodes in the cluster."""
        message = json.dumps({"sender": self.node_id, "type": signal_type, "data": data})
        self.r.publish("kite:cluster:signals", message)

    def listen_for_signals(self, callback):
        """Listen for cluster-wide signals."""
        pubsub = self.r.pubsub()
        pubsub.subscribe("kite:cluster:signals")
        
        def _listen():
            for message in pubsub.listen():
                if message['type'] == 'message':
                    callback(json.loads(message['data']))
        
        threading.Thread(target=_listen, daemon=True).start()
