import torch
import os
import atexit
try:
    # Only available in sglang runtime; guard import for reuse elsewhere
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
except Exception:  # pragma: no cover
    def get_is_capture_mode():
        return False

class FisherCollector:
    def __init__(self):
        self.fisher = {} # layer_id -> tensor (num_experts,)
        self.co_activation = {} # layer_id -> tensor (num_experts, num_experts)
        self.count = {} # layer_id -> int (number of tokens)
        self.save_path = os.environ.get("FISHER_SAVE_PATH", "/tmp/fisher_stats.pt")
        self.enabled = os.environ.get("ENABLE_FISHER_CALIBRATION") == "1"
        # If set, skip update entirely while current stream is capturing a CUDA graph.
        self.skip_in_cuda_graph = os.environ.get("FISHER_SKIP_IN_CUDAGRAPH", "1") == "1"
        # Optional device-side staging (no host transfer inside capture). Disabled by default.
        self.use_device_stage = os.environ.get("FISHER_DEVICE_STAGE", "0") == "1"
        self.save_interval = int(os.environ.get("FISHER_SAVE_INTERVAL", "0"))
        self.update_count = 0
        self.fisher_device = {}
        self.co_activation_device = {}

    def update(self, layer_id, router_logits, hidden_states, top_k):
        if not self.enabled:
            return

        # Check save interval
        self.update_count += 1
        if self.save_interval > 0 and self.update_count % self.save_interval == 0:
            self.save()

        # router_logits: (batch, num_experts)
        # hidden_states: (batch, hidden_size)
        # NOTE: Host transfers (.cpu()) inside a CUDA Graph capture will break capture.
        # We detect capture and either skip (default) or accumulate on device if FISHER_DEVICE_STAGE=1.
        is_capturing = False
        try:
            # torch.cuda.is_current_stream_capturing is available in recent PyTorch
            is_capturing = torch.cuda.is_current_stream_capturing() or get_is_capture_mode()
        except Exception:
            is_capturing = get_is_capture_mode()
        if is_capturing and self.skip_in_cuda_graph:
            return
        
        with torch.no_grad():
            probs = torch.softmax(router_logits, dim=-1) # (batch, num_experts)
            x_norm = torch.norm(hidden_states, p=2, dim=-1) # (batch,)
            
            # Fisher: sum((probs * x_norm)^2)
            # (batch, num_experts) * (batch, 1) -> (batch, num_experts)
            term = (probs * x_norm.unsqueeze(-1)) ** 2
            fisher_batch = term.sum(dim=0)
            
            if is_capturing and self.use_device_stage:
                # Device-side staging: allocate device buffers once (before capture ideally).
                if layer_id not in self.fisher_device:
                    # Allocation during capture may be problematic; assume user pre-warmed or allows it.
                    self.fisher_device[layer_id] = torch.zeros_like(fisher_batch, device=fisher_batch.device)
                    self.co_activation_device[layer_id] = torch.zeros(
                        (probs.shape[1], probs.shape[1]), device=probs.device
                    )
                    self.count[layer_id] = 0
                self.fisher_device[layer_id] += fisher_batch
            else:
                if layer_id not in self.fisher:
                    self.fisher[layer_id] = torch.zeros_like(fisher_batch, device='cpu')
                    self.co_activation[layer_id] = torch.zeros((probs.shape[1], probs.shape[1]), device='cpu')
                    self.count[layer_id] = 0
                self.fisher[layer_id] += fisher_batch.cpu()
            
            # Co-Activation
            # Identify dropped experts (indices NOT in top-k)
            # We need to zero out the top-k probabilities to get G_drop
            
            # Get top-k indices
            _, topk_indices = torch.topk(probs, k=top_k, dim=-1)
            
            # Create mask for top-k
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            
            # G_drop: zero out top-k
            probs_drop = probs.clone()
            probs_drop[mask] = 0.0
            
            # C = probs_drop.T @ probs_drop
            # (num_experts, batch) @ (batch, num_experts) -> (num_experts, num_experts)
            co_act_batch = probs_drop.T @ probs_drop
            if is_capturing and self.use_device_stage:
                self.co_activation_device[layer_id] += co_act_batch
            else:
                self.co_activation[layer_id] += co_act_batch.cpu()
            
            self.count[layer_id] += probs.shape[0]

    def save(self):
        has_cpu_stats = bool(self.fisher)
        has_device_stats = bool(
            self.use_device_stage and (self.fisher_device or self.co_activation_device)
        )
        if not self.enabled or not (has_cpu_stats or has_device_stats):
            return
        # Flush staged device stats if any
        if self.use_device_stage:
            if self.fisher_device:
                for layer_id, dev_tensor in self.fisher_device.items():
                    if layer_id not in self.fisher:
                        self.fisher[layer_id] = dev_tensor.cpu()
                    else:
                        self.fisher[layer_id] += dev_tensor.cpu()
            if self.co_activation_device:
                for layer_id, dev_tensor in self.co_activation_device.items():
                    if layer_id not in self.co_activation:
                        self.co_activation[layer_id] = dev_tensor.cpu()
                    else:
                        self.co_activation[layer_id] += dev_tensor.cpu()
            
        data = {
            "fisher": self.fisher,
            "co_activation": self.co_activation,
            "count": self.count
        }
        # Append rank to filename if distributed
        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        
        path = f"{self.save_path}.rank{rank}"
        torch.save(data, path)
        print(f"Saved fisher stats to {path}")

collector = FisherCollector()
atexit.register(collector.save)
