import atexit
import base64
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

# ─────────────────────────────────────────────────────────────────────────────
# Prompt used for the WAN-VA reset call on the first inference of each session.
# Change this to match your task at launch time.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PROMPT = "stack the yellow block on top of the orange block"


class MultiFrameInferenceServer:
    def __init__(
        self,
        app_host="0.0.0.0",
        app_port=9002,
        ws_server_url="ws://127.0.0.1:29056",
        save_root="./received_data",
        save_data=True,
        prompt=DEFAULT_PROMPT,
    ):
        self.APP_HOST = app_host
        self.APP_PORT = app_port
        self.WS_SERVER_URL = ws_server_url
        self.DEFAULT_PROMPT = prompt

        self.job_id = None
        self.save_data = save_data
        self.call_count = 0
        self._action_per_frame = None  # populated from server metadata on connect

        self.SAVE_ROOT = Path(save_root)
        if self.save_data:
            self.SAVE_ROOT.mkdir(parents=True, exist_ok=True)

        # Maps robot-side image keys → dataset/model keys used in obs_cam_keys.
        self.OBS_KEY_MAP = {
            "front_head": "observation.images.top",
            "left_hand":  "observation.images.wrist",
        }
        # All image keys the robot may send (used only for saving debug images).
        self.ALL_IMAGE_KEYS = ["front_head", "left_hand", "right_hand"]

        self.app = Flask(__name__)

        self._ws = None
        self._ws_lock = threading.Lock()

        self._register_routes()
        atexit.register(self.close_ws)

    # =========================
    # Basic utils
    # =========================
    @staticmethod
    def decode_jpg_bytes(img_bytes) -> np.ndarray | None:
        """Decode raw JPEG bytes → RGB uint8 numpy array."""
        if img_bytes is None:
            return None
        try:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[DECODE ERROR] {e}")
            return None

    @staticmethod
    def encode_numpy_array(arr: np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
        return {
            "__numpy__": base64.b64encode(arr.tobytes()).decode("utf-8"),
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }

    def decode_possible_ndarray(self, obj):
        if isinstance(obj, dict):
            keys = set(obj.keys())

            if b"__ndarray__" in keys and obj[b"__ndarray__"] is True:
                data = obj[b"data"]
                dtype = obj[b"dtype"]
                shape = obj[b"shape"]
                if isinstance(dtype, bytes):
                    dtype = dtype.decode()
                arr = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
                return arr

            if "__ndarray__" in keys and obj["__ndarray__"] is True:
                data = obj["data"]
                dtype = obj["dtype"]
                shape = obj["shape"]
                if isinstance(data, str):
                    data = data.encode("latin1")
                arr = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
                return arr

            if "__numpy__" in keys:
                data = base64.b64decode(obj["__numpy__"].encode("utf-8"))
                dtype = np.dtype(obj["dtype"])
                shape = tuple(obj["shape"])
                arr = np.frombuffer(data, dtype=dtype).reshape(shape)
                return arr

            new_dict = {}
            for k, v in obj.items():
                new_dict[self.decode_possible_ndarray(k)] = self.decode_possible_ndarray(v)
            return new_dict

        if isinstance(obj, list):
            return [self.decode_possible_ndarray(x) for x in obj]

        return obj

    def normalize_state(self, state):
        state = self.decode_possible_ndarray(state)

        if state is None:
            raise ValueError("state is missing")

        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
            if state.ndim != 1:
                state = state.reshape(-1)
            return state.tolist()

        if isinstance(state, list):
            return np.asarray(state, dtype=np.float32).reshape(-1).tolist()

        raise ValueError(f"Unsupported state type: {type(state)}")

    def save_request_and_images(self, state_buffer, session_tag):
        if not self.save_data:
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        sample_dir = self.SAVE_ROOT / f"{ts}_{ms}_{session_tag}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        frame_meta_all = []
        for idx, state in enumerate(state_buffer):
            frame_dir = sample_dir / f"frame_{idx:03d}"
            frame_dir.mkdir(parents=True, exist_ok=True)

            saved_images = {}
            for key in self.ALL_IMAGE_KEYS:
                img_bytes = state.get(key)
                if img_bytes is not None:
                    img = self.decode_jpg_bytes(img_bytes)
                    if img is not None:
                        save_path = frame_dir / f"{key}.jpg"
                        ok = cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        if ok:
                            saved_images[key] = str(save_path)

            frame_meta = {
                "frame_index": idx,
                "state": str(state.get("state")),   # not JSON-serialisable if ndarray
                "saved_images": saved_images,
                "timestamp": state.get("timestamp"),
            }
            with open(frame_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(frame_meta, f, ensure_ascii=False, indent=2)

            frame_meta_all.append(frame_meta)

        meta = {
            "call_count": self.call_count,
            "num_frames": len(state_buffer),
            "frames": frame_meta_all,
            "timestamp": time.time(),
        }
        with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return sample_dir

    def maybe_save_npy(self, path, arr):
        if not self.save_data or path is None or arr is None:
            return
        np.save(path, arr)

    # =========================
    # Websocket utils
    # =========================
    @staticmethod
    def send_obj(ws, obj):
        payload = msgpack.packb(obj, use_bin_type=True)
        print(f"[send_obj] type={type(obj)}, bytes={len(payload)}")
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
                print(f"[recv_obj] msgpack.unpackb failed: {repr(e)}")
                return data

        return data

    def recv_and_check(self, ws, tag):
        result = self.recv_obj(ws)
        if isinstance(result, str) and "Traceback" in result:
            raise RuntimeError(f"{tag} failed on server:\n{result}")
        return result

    # =========================
    # Convert robot frame -> policy obs format
    # =========================
    def frame_to_policy_frame(self, frame: dict) -> dict:
        """Decode raw JPEG bytes for each camera and re-key to model names."""
        out = {}
        for src_key, dst_key in self.OBS_KEY_MAP.items():
            img_bytes = frame.get(src_key)
            if img_bytes is None:
                raise ValueError(f"Missing image field: {src_key}")
            img = self.decode_jpg_bytes(img_bytes)
            if img is None:
                raise ValueError(f"Failed to decode image for key: {src_key}")
            out[dst_key] = img.tolist()
        return out

    def build_latest_obs(self, frames: list) -> dict:
        latest_frame = frames[-1]
        obs = {
            "obs": [self.frame_to_policy_frame(latest_frame)],
        }
        print("[build_latest_obs] num_frames=1, use latest frame")
        return obs

    def build_kv_cache_obs(self, frames: list) -> dict:
        """Build the observation payload for compute_kv_cache.

        Each of the N observation frames is repeated ×4 before being passed to
        the server so that the WAN VAE (temporal stride 4) produces one latent
        frame per real observation frame.  With N=4 this yields 16 input frames
        → 4 output latents = frame_chunk_size.

        This is a deployment compromise: the robot sends one image per executed
        action chunk, but the VAE requires 4 temporally-adjacent frames per
        latent.  Repeating each frame ×4 keeps the tensor shapes correct while
        producing a slightly out-of-distribution (frozen-motion) latent.  The
        trade-off is documented in server_details.md.
        """
        obs_list = []
        for frame in frames:
            policy_frame = self.frame_to_policy_frame(frame)
            for _ in range(4):
                obs_list.append(policy_frame)

        obs = {"obs": obs_list}
        print(f"[build_kv_cache_obs] input_frames={len(frames)}, "
              f"repeated_frames={len(obs_list)} (each ×4)")
        return obs

    # =========================
    # Policy actions
    # =========================
    def send_reset(self, ws, prompt):
        payload = {
            "reset": True,
            "prompt": prompt,
        }
        print("\n========== send_reset ==========")
        print("prompt:", prompt)
        self.send_obj(ws, payload)
        return self.recv_and_check(ws, "reset")

    def send_infer(self, ws, obs_payload):
        payload = dict(obs_payload)

        print("\n========== send_infer ==========")
        self.send_obj(ws, payload)
        result = self.recv_and_check(ws, "infer")
        result = self.decode_possible_ndarray(result)

        print("[infer decoded result]")

        actions = None
        if isinstance(result, dict):
            if "action" in result:
                actions = result.get("action")
            elif "actions" in result:
                actions = result.get("actions")

        if isinstance(actions, np.ndarray):
            print(f"[actions shape] {actions.shape}, dtype={actions.dtype}")
        elif actions is not None:
            actions = np.asarray(actions)
            print(f"[actions shape after asarray] {actions.shape}, dtype={actions.dtype}")

        return actions, result

    def send_compute_kv_cache(self, ws, obs_payload, state_payload=None):
        payload = dict(obs_payload)
        payload["compute_kv_cache"] = True
        payload["state"] = state_payload  # None → server uses buffered predicted_actions

        print("\n====== send_compute_kv_cache ======")
        print(f"[compute_kv_cache obs frames] {len(obs_payload['obs'])}")
        if state_payload is not None:
            print(f"[compute_kv_cache state frames] {len(state_payload)}")
        else:
            print("[compute_kv_cache state] None (server will use predicted_actions)")

        self.send_obj(ws, payload)
        result = self.recv_and_check(ws, "compute_kv_cache")
        result = self.decode_possible_ndarray(result)

        print("[compute_kv_cache decoded result]")
        print(result)
        return result

    # =========================
    # Websocket lifecycle
    # =========================
    def init_ws_once(self):
        with self._ws_lock:
            if self._ws is not None:
                return self._ws

            print(f"[connect] {self.WS_SERVER_URL}")
            ws = websocket.create_connection(self.WS_SERVER_URL, timeout=600)

            metadata = self.recv_obj(ws)
            metadata = self.decode_possible_ndarray(metadata)
            print("[metadata]")
            print(metadata)

            if isinstance(metadata, dict):
                self._action_per_frame = metadata.get('action_per_frame', 12)
                print(f"[init_ws_once] action_per_frame={self._action_per_frame} "
                      f"(frame_chunk_size={metadata.get('frame_chunk_size', 4)})")
            else:
                self._action_per_frame = 12
                print(f"[init_ws_once] metadata not a dict; using default action_per_frame={self._action_per_frame}")

            self._ws = ws
            print("[init_ws_once] websocket initialized")
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

    # =========================
    # Business logic
    # =========================
    def handle_inference(self, state_buffer: list) -> dict:
        """
        state_buffer: list of frame dicts, each with keys:
            front_head  – raw JPEG bytes
            left_hand   – raw JPEG bytes
            right_hand  – raw JPEG bytes  (optional, not used by UR5 model)
            state       – joint state list / ndarray
            timestamp   – float (optional)
        """
        print("\n========== [SERVER RECEIVE] ==========")
        print(f"state_buffer len: {len(state_buffer)}")
        for i, s in enumerate(state_buffer):
            for key in self.ALL_IMAGE_KEYS:
                img_bytes = s.get(key)
                if img_bytes is not None:
                    img = self.decode_jpg_bytes(img_bytes)
                    print(f"[Frame {i}] {key}: shape={img.shape if img is not None else None}")
                else:
                    print(f"[Frame {i}] {key}: None")
            robot_state = s.get("state")
            if robot_state is not None:
                arr = np.array(robot_state)
                print(f"[Frame {i}] state: shape={arr.shape}, sample={arr[:3]}")
            ts = s.get("timestamp")
            if ts:
                print(f"[Frame {i}] delay: {time.time() - ts:.3f}s")

        if len(state_buffer) < 4:
            raise ValueError(f"state_buffer must contain at least 4 frames, got {len(state_buffer)}.")

        frames_4 = state_buffer[-4:]
        sample_dir = self.save_request_and_images(frames_4, session_tag=f"call{self.call_count}")

        latest_obs = self.build_latest_obs(frames_4)
        kv_obs = self.build_kv_cache_obs(frames_4)

        action_per_frame = self._action_per_frame or 12

        with self._ws_lock:
            ws = self.get_ws()

            if self.call_count == 0:
                # First call: send reset then infer directly (no KV history yet).
                # Frame-0 action slot is conditioned on zeros per training convention;
                # we patch those with the initial robot state so the robot holds position.
                prompt = frames_4[-1].get("prompt") or self.DEFAULT_PROMPT
                print(f"[handle_inference] call_count=0: sending reset, then infer directly")
                print(f"[handle_inference] prompt: {prompt!r}")
                self.send_reset(ws, prompt)

                raw_action, _ = self.send_infer(ws, latest_obs)

                # Patch the first action_per_frame timesteps with the robot's current
                # state to hold position instead of commanding zeros.
                if raw_action is not None:
                    raw_action = np.array(raw_action, dtype=np.float32)  # always copies → writeable
                    initial_state_raw = frames_4[-1].get("state")
                    if initial_state_raw is not None:
                        initial_state = np.asarray(
                            self.normalize_state(initial_state_raw), dtype=np.float32
                        )
                        n_action_dims = raw_action.shape[1] if raw_action.ndim > 1 else raw_action.shape[0]
                        n_copy = min(n_action_dims, initial_state.shape[0])
                        print(f"[handle_inference] patching first {action_per_frame} "
                              f"zero-slot actions with initial state (dims={n_copy})")
                        raw_action[:action_per_frame, :n_copy] = initial_state[:n_copy]
                    else:
                        print("[handle_inference] WARNING: no initial state in frame; "
                              "zero-slot actions left as zeros")

                final_action = raw_action
            else:
                # Subsequent calls: update KV cache with repeated real observations,
                # then run inference.
                _ = self.send_compute_kv_cache(ws, kv_obs)  # state=None → use predicted_actions
                final_action, _ = self.send_infer(ws, latest_obs)
                if final_action is not None:
                    final_action = np.array(final_action, dtype=np.float32)  # always copies → writeable

            self.call_count += 1

        if final_action is not None:
            self.maybe_save_npy(
                sample_dir / "actions.npy" if sample_dir is not None else None,
                final_action,
            )
            # Return as a plain list-of-lists so jsonify can serialise it.
            actions_out = final_action.tolist()
        else:
            actions_out = []

        print(f"\n[SERVER] call_count={self.call_count}, actions shape={final_action.shape if final_action is not None else None}")
        print("=====================================\n")

        return {"actions": actions_out}

    # =========================
    # Flask routes
    # =========================
    def _register_routes(self):
        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "status": "ok",
                "ws_initialized": self._ws is not None,
                "save_data": self.save_data,
                "call_count": self.call_count,
                "prompt": self.DEFAULT_PROMPT,
            })

        @self.app.route("/inference", methods=["POST"])
        def inference():
            try:
                t0 = time.time()

                # ── 1. Deserialise pickle payload ──────────────────────────────
                payload_bytes = request.data
                data = pickle.loads(payload_bytes)

                # client sends: payload = [state_buffer]
                state_buffer = list(data)
                print('state buffer')
                print(type(state_buffer))
                print(len(state_buffer))
                for it in state_buffer:
                    print(type(it))

                # ── 2. Run inference ───────────────────────────────────────────
                response = self.handle_inference(state_buffer)

                dt = time.time() - t0
                print(f"[SERVER] request handled in {dt:.3f}s")
                return jsonify(response)

            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({
                    "status": "error",
                    "message": str(e),
                }), 500

    # =========================
    # Main
    # =========================
    def run(self, debug=False, threaded=True):
        self.init_ws_once()
        self.app.run(
            host=self.APP_HOST,
            port=self.APP_PORT,
            debug=debug,
            threaded=threaded,
        )


if __name__ == "__main__":
    server = MultiFrameInferenceServer(
        app_host="0.0.0.0",
        app_port=9002,
        ws_server_url="ws://127.0.0.1:9001",
        save_root="./received_data",
        save_data=False,
        prompt=DEFAULT_PROMPT,
    )
    print(f"🚀 WAN-VA Chunk Server (UR5 local) running on {server.APP_HOST}:{server.APP_PORT}")
    print(f"   prompt: {server.DEFAULT_PROMPT}")
    server.run()
