import atexit
import base64
import json
import threading
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import websocket
from flask import Flask, request, jsonify


class MultiFrameInferenceServer:
    def __init__(
        self,
        app_host="0.0.0.0",
        app_port=9002,
        ws_server_url="ws://127.0.0.1:29056",
        prompt="place the pen on the table into the pencil case",
        save_root="./received_data",
        save_data=True,
    ):
        self.APP_HOST = app_host
        self.APP_PORT = app_port
        self.WS_SERVER_URL = ws_server_url
        self.PROMPT = prompt

        self.job_id = None
        self.save_data = save_data

        self.SAVE_ROOT = Path(save_root)
        if self.save_data:
            self.SAVE_ROOT.mkdir(parents=True, exist_ok=True)

        self.OBS_KEY_MAP = {
            "front_head": "observation.images.top",
            "left_hand": "observation.images.left_wrist",
            "right_hand": "observation.images.right_wrist",
        }
        self.REQUIRED_IMAGE_KEYS = ["front_head", "left_hand", "right_hand"]

        self.app = Flask(__name__)

        self._ws = None
        self._ws_lock = threading.Lock()

        self._register_routes()
        atexit.register(self.close_ws)

    # =========================
    # Basic utils
    # =========================
    @staticmethod
    def b64_to_bytes(b64_str: str) -> bytes:
        return base64.b64decode(b64_str.encode("utf-8"))

    def decode_image_from_base64(self, b64_str: str) -> np.ndarray:
        img_buf = self.b64_to_bytes(b64_str)
        img_np = np.frombuffer(img_buf, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from base64")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

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

    def save_request_and_images(self, data, job_id):
        if not self.save_data:
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        sample_dir = self.SAVE_ROOT / f"{ts}_{ms}_{job_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        with open(sample_dir / "request.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        frames = data.get("frames", [])
        frame_meta_all = []

        for idx, frame in enumerate(frames):
            frame_dir = sample_dir / f"frame_{idx:03d}"
            frame_dir.mkdir(parents=True, exist_ok=True)

            saved_images = {}
            for key in self.REQUIRED_IMAGE_KEYS:
                if key in frame and frame[key] is not None:
                    img = self.decode_image_from_base64(frame[key])
                    save_path = frame_dir / f"{key}.jpg"
                    ok = cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if ok:
                        saved_images[key] = str(save_path)

            frame_meta = {
                "frame_index": idx,
                "state": frame.get("state"),
                "saved_images": saved_images,
                "timestamp": time.time(),
            }
            with open(frame_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(frame_meta, f, ensure_ascii=False, indent=2)

            frame_meta_all.append(frame_meta)

        meta = {
            "job_id": job_id,
            "prompt": data.get("prompt"),
            "num_frames": len(frames),
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
    # Convert new JSON -> policy obs format
    # =========================
    def frame_to_policy_frame(self, frame: dict) -> dict:
        out = {}

        for src_key, dst_key in self.OBS_KEY_MAP.items():
            if src_key not in frame or frame[src_key] is None:
                raise ValueError(f"Missing image field: {src_key}")
            img = self.decode_image_from_base64(frame[src_key])
            out[dst_key] = img.tolist()

        return out

    def build_latest_obs(self, frames: list) -> dict:
        latest_frame = frames[-1]
        obs = {
            "obs": [self.frame_to_policy_frame(latest_frame)],
        }
        print("[build_latest_obs] num_frames=1, use latest frame")
        return obs

    def build_kv_cache_obs_and_state(self, frames: list):
        obs_list = []
        state_list = []

        for frame in frames:
            obs_list.append(self.frame_to_policy_frame(frame))
            state_list.append(self.normalize_state(frame.get("state")))

        obs = {"obs": obs_list}
        print(f"[build_kv_cache_obs_and_state] num_frames={len(obs_list)}")
        print(
            f"[build_kv_cache_obs_and_state] state_shape=({len(state_list)}, "
            f"{len(state_list[0]) if state_list else 0})"
        )
        return obs, state_list

    # =========================
    # Policy actions
    # =========================
    def send_reset(self, ws, prompt=None):
        if prompt is None:
            prompt = self.PROMPT

        payload = {
            "reset": True,
            "prompt": prompt,
        }
        print("\n========== send_reset ==========")
        print("got prompt: ", prompt)
        self.send_obj(ws, payload)
        return self.recv_and_check(ws, "reset")

    def send_infer(self, ws, obs_payload, prompt=None):
        payload = dict(obs_payload)
        if prompt is not None:
            payload["prompt"] = prompt

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

    def send_compute_kv_cache(self, ws, obs_payload, state_payload):
        payload = dict(obs_payload)
        payload["compute_kv_cache"] = True
        payload["state"] = state_payload

        print("\n====== send_compute_kv_cache ======")
        print(f"[compute_kv_cache obs frames] {len(obs_payload['obs'])}")
        print(f"[compute_kv_cache state frames] {len(state_payload)}")

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

            self.send_reset(ws, self.PROMPT)

            self._ws = ws
            print("[init_ws_once] websocket initialized and reset done")
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
    def handle_inference(self, data):
        job_id = data.get("job_id", "no_job_id")
        frames = data.get("frames", None)
        first_action = None

        if frames is None:
            raise ValueError("Missing 'frames' field in request.")
        if not isinstance(frames, list):
            raise ValueError("'frames' must be a list.")
        if len(frames) == 0:
            raise ValueError("'frames' is empty.")
        if len(frames) < 4:
            raise ValueError(f"'frames' must contain at least 4 frames, got {len(frames)}.")

        frames_4 = frames[-4:]
        sample_dir = self.save_request_and_images(data, job_id)

        latest_obs = self.build_latest_obs(frames_4)
        kv_obs, kv_state = self.build_kv_cache_obs_and_state(frames_4)

        with self._ws_lock:
            ws = self.get_ws()

            if self.job_id != job_id:
                print(f"[job_id changed] {self.job_id} -> {job_id}")
                first_action, _ = self.send_infer(ws, latest_obs, prompt=self.PROMPT)
                self.job_id = job_id

            _ = self.send_compute_kv_cache(ws, kv_obs, kv_state)
            second_action, _ = self.send_infer(ws, latest_obs)

        final_action = None
        if isinstance(second_action, np.ndarray):
            final_action = second_action
        elif second_action is not None:
            final_action = np.asarray(second_action, dtype=np.float32)

        if isinstance(first_action, np.ndarray):
            self.maybe_save_npy(
                sample_dir / "actions_first.npy" if sample_dir is not None else None,
                first_action,
            )
        elif first_action is not None:
            self.maybe_save_npy(
                sample_dir / "actions_first.npy" if sample_dir is not None else None,
                np.asarray(first_action, dtype=np.float32),
            )

        if final_action is not None:
            self.maybe_save_npy(
                sample_dir / "actions_second.npy" if sample_dir is not None else None,
                final_action,
            )
            result = [self.encode_numpy_array(final_action[t]) for t in range(final_action.shape[0])]
        else:
            result = None

        return {
            "status": "ok",
            "saved_dir": str(sample_dir) if sample_dir is not None else None,
            "num_frames": len(frames_4),
            "actions": result,
        }

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
                "current_job_id": self.job_id,
            })

        @self.app.route("/inference", methods=["POST"])
        def inference():
            try:
                data = request.get_json(force=True)
                response = self.handle_inference(data)
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
    def run(self, debug=True, threaded=True):
        self.init_ws_once()
        self.app.run(
            host=self.APP_HOST,
            port=self.APP_PORT,
            debug=debug,
            threaded=threaded,
        )


if __name__ == "__main__":
    server = MultiFrameInferenceServer(
        app_host="127.0.0.1",
        app_port=9002,
        ws_server_url="ws://127.0.0.1:9001",
        prompt="scan the QR code on the medicine box using the scanner",
        save_root="./received_data_request",
        save_data=True,  # True 就保存，False 就不保存
    )
    server.run()