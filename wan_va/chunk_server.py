"""
chunk_server_unified.py — single chunk-server for all robot embodiments and
both aligned / notaligned WAN-VA server variants.

Usage
-----
python chunk_server_unified.py \
    --mode       notaligned \          # or "aligned"
    --embodiment ur5 \                 # selects OBS_KEY_MAP preset
    --prompt     "stack the block" \
    --ws-url     ws://127.0.0.1:9001 \
    --port       9002 \
    --save-root  ./received_data \
    --save-data

Input contract
--------------
The /inference endpoint expects a pickle-encoded dict sent as the raw POST body:

    {
        "frames": [
            {
                "<cam_key>": np.ndarray  uint8 (256, 256, 3)   # already resized
                "state":     list | np.ndarray               # joint positions
                "timestamp": float                           # optional
            },
            ...  # at least 4 frames
        ]
    }

Session lifecycle
-----------------
  POST /start      — begin a new inference job:
                       • triggers async save of the accumulated action sequence
                         from the previous job
                       • increments job_count
                       • resets call_count to 0
  POST /inference  — run one inference step (call_count advances automatically)
  GET  /health     — liveness check

Aligned vs notaligned
---------------------
  call_count == 0 (first step of a job)
    aligned:     send_infer(state=initial_state)          # no KV warm-up
    notaligned:  send_compute_kv_cache(1 frame, state) → send_infer()

  call_count > 0 (all subsequent steps, both modes)
    send_compute_kv_cache(4 frames ×4 repeat) → send_infer()
"""

import argparse
import atexit
import json
import pickle
import threading
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import websocket
from flask import Flask, request, jsonify


# ---------------------------------------------------------------------------
# msgpack helper — encodes numpy arrays into the binary format that the
# WAN-VA server's msgpack_numpy.unpackb / unpack_array decodes back.
# ---------------------------------------------------------------------------
def pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    return obj


# ---------------------------------------------------------------------------
# Embodiment presets
#   key   : robot name passed via --embodiment
#   value : dict mapping robot-side frame keys → model obs_cam_keys
# ---------------------------------------------------------------------------
EMBODIMENT_CAM_KEYS = {
    # UR5 with single wrist camera
    "ur5": {
        "front_head": "observation.images.top",
        "left_hand":  "observation.images.wrist",
    },
    # Bimanual setup with two wrist cameras
    "aloha": {
        "front_head": "observation.images.top",
        "left_hand":  "observation.images.left_wrist",
        "right_hand": "observation.images.right_wrist",
    },
    # Alias
    "arx5": {
        "front_head": "observation.images.top",
        "left_hand":  "observation.images.wrist",
        "right_hand": "observation.images.scene",
    },
    "franka": {
        "front_head": "observation.images.top",
        "left_hand":  "observation.images.left_wrist",
        "right_hand": "observation.images.right_wrist",
    },
}

DEFAULT_EMBODIMENT = "ur5"
DEFAULT_PROMPT     = "stack one colored block on top of another"


# ---------------------------------------------------------------------------
# Main server class
# ---------------------------------------------------------------------------
class ChunkInferenceServer:
    def __init__(
        self,
        mode: str = "notaligned",
        embodiment: str = DEFAULT_EMBODIMENT,
        app_host: str = "0.0.0.0",
        app_port: int = 9002,
        ws_server_url: str = "ws://127.0.0.1:9001",
        save_root: str = "./received_data",
        save_data: bool = False,
        prompt: str = DEFAULT_PROMPT,
    ):
        if mode not in ("aligned", "notaligned"):
            raise ValueError(f"--mode must be 'aligned' or 'notaligned', got {mode!r}")
        if embodiment not in EMBODIMENT_CAM_KEYS:
            raise ValueError(
                f"Unknown embodiment {embodiment!r}. "
                f"Known: {list(EMBODIMENT_CAM_KEYS)}"
            )

        self.mode           = mode
        self.notaligned     = (mode == "notaligned")
        self.OBS_KEY_MAP    = EMBODIMENT_CAM_KEYS[embodiment]
        self.APP_HOST       = app_host
        self.APP_PORT       = app_port
        self.WS_SERVER_URL  = ws_server_url
        self.save_data      = save_data
        self.DEFAULT_PROMPT = prompt

        self.SAVE_ROOT = Path(save_root)
        if self.save_data:
            self.SAVE_ROOT.mkdir(parents=True, exist_ok=True)

        # Session counters
        self.job_count  = 0   # incremented by /start; identifies each new experiment
        self.call_count = 0   # steps within the current job; reset by /start

        # Accumulated actions for the current job (one entry per infer call)
        self._action_chunks: list[np.ndarray] = []
        self._action_chunks_lock = threading.Lock()

        # WAN-VA server metadata received on connect
        self._action_per_frame: int | None = None
        self._frame_chunk_size: int = 4  # updated from server metadata on connect

        # WebSocket
        self._ws      = None
        self._ws_lock = threading.Lock()

        self.app = Flask(__name__)
        self._register_routes()
        atexit.register(self._atexit_save)

    # -----------------------------------------------------------------------
    # msgpack helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def send_obj(ws, obj):
        payload = msgpack.packb(obj, default=pack_array, use_bin_type=True)
        print(f"[send_obj] type={type(obj).__name__}, bytes={len(payload)}")
        ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)

    @staticmethod
    def recv_obj(ws):
        data = ws.recv()
        if isinstance(data, str):
            print(f"[recv_obj] text len={len(data)}")
            print(f"[recv_obj] text preview:\n{data[:1000]}")
            return data
        if isinstance(data, (bytes, bytearray)):
            data = bytes(data)
            print(f"[recv_obj] binary len={len(data)}")
            print(f"[recv_obj] binary head={data[:32]!r}")
            try:
                return msgpack.unpackb(data, raw=False)
            except Exception as e:
                print(f"[recv_obj] msgpack.unpackb failed: {e!r}")
                return data
        return data

    def recv_and_check(self, ws, tag: str):
        result = self.recv_obj(ws)
        if isinstance(result, str) and "Traceback" in result:
            raise RuntimeError(f"{tag} failed on server:\n{result}")
        return result

    # -----------------------------------------------------------------------
    # numpy decode (server responses may contain pack_array-encoded arrays)
    # -----------------------------------------------------------------------
    def decode_possible_ndarray(self, obj):
        if isinstance(obj, dict):
            keys = set(obj.keys())
            if b"__ndarray__" in keys and obj[b"__ndarray__"] is True:
                dtype = obj[b"dtype"]
                if isinstance(dtype, bytes):
                    dtype = dtype.decode()
                return np.frombuffer(obj[b"data"], dtype=np.dtype(dtype)).reshape(obj[b"shape"])
            if "__ndarray__" in keys and obj["__ndarray__"] is True:
                data = obj["data"]
                if isinstance(data, str):
                    data = data.encode("latin1")
                dtype = obj["dtype"]
                if isinstance(dtype, bytes):
                    dtype = dtype.decode()
                return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(obj["shape"])
            return {self.decode_possible_ndarray(k): self.decode_possible_ndarray(v)
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.decode_possible_ndarray(x) for x in obj]
        return obj

    # -----------------------------------------------------------------------
    # State normalisation
    # -----------------------------------------------------------------------
    def normalize_state(self, state) -> list:
        """Decode / flatten the state field into a plain float32 list."""
        state = self.decode_possible_ndarray(state)
        if state is None:
            raise ValueError("state is missing")
        if isinstance(state, np.ndarray):
            return state.astype(np.float32).reshape(-1).tolist()
        if isinstance(state, list):
            return np.asarray(state, dtype=np.float32).reshape(-1).tolist()
        raise ValueError(f"Unsupported state type: {type(state)}")

    # -----------------------------------------------------------------------
    # Frame → policy obs conversion
    # Images arrive as uint8 numpy arrays (H, W, C) already at 256×256.
    # Only re-key to match the model's obs_cam_keys.
    # -----------------------------------------------------------------------
    def frame_to_policy_frame(self, frame: dict) -> dict:
        out = {}
        for src_key, dst_key in self.OBS_KEY_MAP.items():
            img = frame.get(src_key)
            if img is None:
                raise ValueError(f"Missing image field in frame: {src_key!r}")
            if not isinstance(img, np.ndarray):
                img = np.asarray(img, dtype=np.uint8)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"Expected uint8 (H,W,3) for {src_key!r}, got shape {img.shape}"
                )
            out[dst_key] = img.astype(np.uint8)
        return out

    def build_latest_obs(self, frames: list) -> dict:
        """Single-frame obs from the most recent frame (used for infer)."""
        obs = {"obs": [self.frame_to_policy_frame(frames[-1])]}
        print(f"[build_latest_obs] 1 frame (latest)")
        return obs

    def build_kv_cache_obs(self, frames: list) -> dict:
        """KV-cache obs: each of the N frames repeated ×4 for the VAE temporal stride.

        The WAN VAE uses a 1,4,4,4... temporal compression scheme: after the
        anchor frame, each latent frame requires 4 input video frames.  At
        deployment the robot sends one representative image per executed action
        chunk, so we repeat it ×4 to keep the tensor shapes correct.
        """
        obs_list = []
        for frame in frames:
            pf = self.frame_to_policy_frame(frame)
            for _ in range(4):
                obs_list.append(pf)
        print(f"[build_kv_cache_obs] {len(frames)} frames → {len(obs_list)} (each ×4)")
        return {"obs": obs_list}

    # -----------------------------------------------------------------------
    # WAN-VA server RPC helpers
    # -----------------------------------------------------------------------
    def send_reset(self, ws, prompt: str):
        print("\n========== send_reset ==========")
        print(f"prompt: {prompt!r}")
        self.send_obj(ws, {"reset": True, "prompt": prompt})
        return self.recv_and_check(ws, "reset")

    def send_compute_kv_cache(self, ws, obs_payload: dict, state_payload=None):
        payload = dict(obs_payload)
        payload["compute_kv_cache"] = True
        payload["state"] = state_payload

        print("\n====== send_compute_kv_cache ======")
        print(f"[kv_cache] obs_frames={len(obs_payload['obs'])}")
        if state_payload is not None:
            print(f"[kv_cache] state provided ({len(state_payload)} values)")
        else:
            print("[kv_cache] state=None (server uses predicted_actions)")

        self.send_obj(ws, payload)
        result = self.recv_and_check(ws, "compute_kv_cache")
        result = self.decode_possible_ndarray(result)
        print(f"[kv_cache] result: {result}")
        return result

    def send_infer(self, ws, obs_payload: dict, state=None):
        payload = dict(obs_payload)
        if state is not None:
            payload["state"] = state

        print("\n========== send_infer ==========")
        self.send_obj(ws, payload)
        result = self.recv_and_check(ws, "infer")
        result = self.decode_possible_ndarray(result)
        print("[infer decoded result]")

        actions = None
        if isinstance(result, dict):
            if "action" in result:
                actions = result["action"]
            elif "actions" in result:
                actions = result["actions"]
        if actions is not None and not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)
        if isinstance(actions, np.ndarray):
            print(f"[actions shape] {actions.shape}, dtype={actions.dtype}")
        return actions, result

    # -----------------------------------------------------------------------
    # WebSocket lifecycle
    # -----------------------------------------------------------------------
    def init_ws_once(self, max_retries: int = 10, retry_delay: float = 2.0):
        with self._ws_lock:
            if self._ws is not None:
                return self._ws

            print(f"[connect] {self.WS_SERVER_URL}")
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    ws = websocket.create_connection(self.WS_SERVER_URL, timeout=600)
                    break
                except Exception as e:
                    last_exc = e
                    print(f"[connect] attempt {attempt}/{max_retries} failed: {e}; "
                          f"retrying in {retry_delay}s …")
                    time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Could not connect to {self.WS_SERVER_URL} "
                    f"after {max_retries} attempts"
                ) from last_exc

            metadata = self.recv_obj(ws)
            metadata = self.decode_possible_ndarray(metadata)
            print(f"[metadata] {metadata}")
            if isinstance(metadata, dict):
                self._action_per_frame  = metadata.get("action_per_frame", 12)
                self._frame_chunk_size  = metadata.get("frame_chunk_size", 4)
                print(f"[init_ws_once] action_per_frame={self._action_per_frame}  "
                      f"frame_chunk_size={self._frame_chunk_size}")
            else:
                self._action_per_frame = 12
                print(f"[init_ws_once] metadata unexpected; default action_per_frame=12")

            self._ws = ws
            print("[init_ws_once] websocket ready")
            return self._ws

    def get_ws(self):
        if self._ws is None:
            return self.init_ws_once()
        return self._ws

    def close_ws(self):
        with self._ws_lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                    print("[ws closed]")
                except Exception as e:
                    print(f"[ws close error] {e}")
                self._ws = None

    # -----------------------------------------------------------------------
    # Action chunk accumulation + async save
    # -----------------------------------------------------------------------
    def _save_actions_async(self, action_chunks: list, save_path: Path):
        """Background thread: concatenate all action chunks and save as .npy."""
        def _worker():
            try:
                all_actions = np.concatenate(action_chunks, axis=0)   # (T_total, action_dim)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(save_path), all_actions)
                print(f"[save_actions] saved {all_actions.shape} → {save_path}")
            except Exception as e:
                print(f"[save_actions] ERROR: {e}")
        threading.Thread(target=_worker, daemon=True).start()

    def _flush_action_chunks(self, job_count: int):
        """Snapshot and clear accumulated chunks, then save asynchronously."""
        with self._action_chunks_lock:
            chunks = list(self._action_chunks)
            self._action_chunks = []

        if not chunks or not self.save_data:
            return

        save_path = self.SAVE_ROOT / f"job_{job_count:04d}" / "actions_all.npy"
        print(f"[flush] job {job_count}: {len(chunks)} chunks → {save_path}")
        self._save_actions_async(chunks, save_path)

    def _atexit_save(self):
        """Called on process exit: flush any un-saved action chunks."""
        print("[atexit] flushing action chunks before exit …")
        self._flush_action_chunks(self.job_count)

    # -----------------------------------------------------------------------
    # Optional: save per-call debug images
    # -----------------------------------------------------------------------
    def _save_call_images(self, frames: list, job_count: int, call_count: int):
        if not self.save_data:
            return
        call_dir = self.SAVE_ROOT / f"job_{job_count:04d}" / f"call_{call_count:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        for fi, frame in enumerate(frames):
            for src_key in self.OBS_KEY_MAP:
                img = frame.get(src_key)
                if img is not None and isinstance(img, np.ndarray):
                    fpath = call_dir / f"frame{fi:02d}_{src_key}.jpg"
                    cv2.imwrite(str(fpath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        meta = {
            "job_count":  job_count,
            "call_count": call_count,
            "num_frames": len(frames),
            "states": [
                np.asarray(f["state"]).tolist() if f.get("state") is not None else None
                for f in frames
            ],
            "timestamps": [f.get("timestamp") for f in frames],
        }
        with open(call_dir / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)

    # -----------------------------------------------------------------------
    # Core inference logic
    # -----------------------------------------------------------------------
    def handle_inference(self, state_buffer: list, prompt: str | None = None) -> dict:
        if len(state_buffer) < self._frame_chunk_size:
            raise ValueError(
                f"Need at least {self._frame_chunk_size} frames, got {len(state_buffer)}"
            )

        frames_n = state_buffer[-self._frame_chunk_size:]

        print(f"\n========== [INFERENCE] job={self.job_count} call={self.call_count} "
              f"frame_chunk_size={self._frame_chunk_size} ==========")
        for i, f in enumerate(frames_n):
            for src_key in self.OBS_KEY_MAP:
                img = f.get(src_key)
                shape = img.shape if isinstance(img, np.ndarray) else None
                print(f"  frame[{i}] {src_key}: {shape}")
            state_arr = f.get("state")
            if state_arr is not None:
                arr = np.asarray(state_arr)
                print(f"  frame[{i}] state: shape={arr.shape}  first3={arr.flat[:3]}")

        self._save_call_images(frames_n, self.job_count, self.call_count)

        latest_obs = self.build_latest_obs(frames_n)
        kv_obs     = self.build_kv_cache_obs(frames_n)

        with self._ws_lock:
            ws = self.get_ws()
            t_start = time.time()

            if self.call_count == 0:
                # ── First step of a new job ──────────────────────────────────
                prompt = prompt or self.DEFAULT_PROMPT
                print(f"\n[call=0] mode={self.mode}  prompt={prompt!r}")

                t0 = time.time()
                print("[timing] >>> reset (T5 encoding) …")
                self.send_reset(ws, prompt)
                print(f"[timing] <<< reset  ({time.time()-t0:.2f}s)")

                initial_state_raw = frames_n[-1].get("state")
                initial_state = (
                    self.normalize_state(initial_state_raw)
                    if initial_state_raw is not None else None
                )
                if initial_state is None:
                    print("[WARNING] no initial state — action conditioning skipped")

                if self.notaligned:
                    # KV warm-up with 1-frame VAE anchor + actual robot state
                    single_frame_obs = self.build_latest_obs([frames_n[-1]])
                    t0 = time.time()
                    print("[timing] >>> KV warm-up (VAE anchor + transformer cache) …")
                    self.send_compute_kv_cache(ws, single_frame_obs,
                                               state_payload=initial_state)
                    print(f"[timing] <<< KV warm-up  ({time.time()-t0:.2f}s)")

                    t0 = time.time()
                    print("[timing] >>> infer …")
                    final_action, _ = self.send_infer(ws, latest_obs)
                    print(f"[timing] <<< infer  ({time.time()-t0:.2f}s)")

                else:
                    # aligned: pass state into infer for frame-0 action inpainting
                    state_arr = (
                        np.asarray(initial_state, dtype=np.float32)
                        if initial_state is not None else None
                    )
                    t0 = time.time()
                    print("[timing] >>> infer (aligned, frame-0 with state) …")
                    final_action, _ = self.send_infer(ws, latest_obs, state=state_arr)
                    print(f"[timing] <<< infer  ({time.time()-t0:.2f}s)")

            else:
                # ── All subsequent steps (identical for both modes) ──────────
                t0 = time.time()
                print(f"[timing] >>> compute_kv_cache (call={self.call_count}) …")
                self.send_compute_kv_cache(ws, kv_obs)   # state=None → predicted_actions
                print(f"[timing] <<< compute_kv_cache  ({time.time()-t0:.2f}s)")

                t0 = time.time()
                print(f"[timing] >>> infer (call={self.call_count}) …")
                final_action, _ = self.send_infer(ws, latest_obs)
                print(f"[timing] <<< infer  ({time.time()-t0:.2f}s)")

            self.call_count += 1
            print(f"[timing] === total call {self.call_count}  "
                  f"({time.time()-t_start:.2f}s) ===")

        # Accumulate for end-of-job async save
        if final_action is not None:
            final_action = np.asarray(final_action, dtype=np.float32)
            with self._action_chunks_lock:
                self._action_chunks.append(final_action)
            print(f"[SERVER] job={self.job_count} call={self.call_count}  "
                  f"actions={final_action.shape}")
        else:
            print(f"[SERVER] job={self.job_count} call={self.call_count}  actions=None")

        return {
            "actions": final_action.tolist() if final_action is not None else [],
        }

    # -----------------------------------------------------------------------
    # Flask routes
    # -----------------------------------------------------------------------
    def _register_routes(self):

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "status":           "ok",
                "mode":             self.mode,
                "ws_initialized":   self._ws is not None,
                "save_data":        self.save_data,
                "job_count":        self.job_count,
                "call_count":       self.call_count,
                "prompt":           self.DEFAULT_PROMPT,
                "frame_chunk_size": self._frame_chunk_size,
                "action_per_frame": self._action_per_frame,
            })

        @self.app.route("/inference", methods=["POST"])
        def inference():
            try:
                t0=time.time()
                # Payload is a pickle-encoded dict: {'frames': list[dict], 'prompt': str}
                data = pickle.loads(request.data)

                if not isinstance(data, dict):
                    raise ValueError(
                        f"Expected pickled dict with 'frames' key, got {type(data).__name__}"
                    )

                frames = data.get("frames")
                if frames is None:
                    raise ValueError("Missing 'frames' key in payload")

                prompt = data.get("prompt")  # may be None; handle_inference falls back to DEFAULT_PROMPT

                print(f"[inference] frames={len(frames)}  prompt={prompt!r}")
                response = self.handle_inference(frames, prompt=prompt)
                print(f"[timing][inference] TOTAL={time.time()-t0:.4f}s")
                return jsonify(response)

            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route("/start", methods=["POST"])
        def start():
            """Begin a new inference job.

            Saves the accumulated action sequence from the completed job
            asynchronously, then resets session counters.
            """
            print(f"\n[/start] ending job {self.job_count}, "
                  f"starting job {self.job_count + 1}")
            self._flush_action_chunks(self.job_count)  # async save of completed job
            self.job_count  += 1
            self.call_count  = 0
            print(f"[/start] job_count={self.job_count}  call_count={self.call_count}")
            return jsonify({"status": "ok", "job_count": self.job_count})

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------
    def run(self, debug: bool = False, threaded: bool = True):
        self.init_ws_once()
        print(f"\n🚀  WAN-VA Chunk Server  (unified)")
        print(f"   mode={self.mode}  cam_keys={list(self.OBS_KEY_MAP)}")
        print(f"   Flask  {self.APP_HOST}:{self.APP_PORT}")
        print(f"   WS     {self.WS_SERVER_URL}")
        print(f"   save   {self.SAVE_ROOT}  (save_data={self.save_data})")
        self.app.run(host=self.APP_HOST, port=self.APP_PORT,
                     debug=debug, threaded=threaded)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="WAN-VA unified chunk server")
    p.add_argument("--mode", choices=["aligned", "notaligned"],
                   default="aligned",
                   help="Inference mode (default: aligned)")
    p.add_argument("--embodiment", required=True,
                   choices=list(EMBODIMENT_CAM_KEYS),
                   help=f"Robot embodiment preset for camera key mapping "
                        f"(default: {DEFAULT_EMBODIMENT})")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Default task prompt sent to the WAN-VA server on reset")
    p.add_argument("--ws-url", default="ws://127.0.0.1:9001",
                   dest="ws_url",
                   help="WebSocket URL of the WAN-VA policy server")
    p.add_argument("--port", type=int, default=9002,
                   help="Flask HTTP port (default: 9002)")
    p.add_argument("--host", default="0.0.0.0",
                   help="Flask bind host (default: 0.0.0.0)")
    p.add_argument("--save-root", default="./received_data",
                   dest="save_root",
                   help="Root directory for saving images and actions")
    p.add_argument("--save-data", action="store_true",
                   dest="save_data",
                   help="Enable saving of images and action sequences")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    server = ChunkInferenceServer(
        mode          = args.mode,
        embodiment    = args.embodiment,
        app_host      = args.host,
        app_port      = args.port,
        ws_server_url = args.ws_url,
        save_root     = args.save_root,
        save_data     = args.save_data,
        prompt        = args.prompt,
    )
    server.run()
