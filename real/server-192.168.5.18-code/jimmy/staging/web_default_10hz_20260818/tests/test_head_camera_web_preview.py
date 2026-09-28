import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib import error, request


MODULE_PATH = Path(__file__).resolve().parents[1] / "head_camera_web_preview.py"
SPEC = importlib.util.spec_from_file_location("head_camera_web_preview", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LatestJpegTests(unittest.TestCase):
    def test_accepts_only_bounded_jpeg_frames(self):
        store = MODULE.LatestJpeg(clock=lambda: 10.0)
        self.assertFalse(store.update(b"not a jpeg"))
        self.assertFalse(store.update(b"\xff\xd8missing end"))
        self.assertFalse(store.update(b"\xff\xd8" + b"x" * MODULE.MAX_JPEG_BYTES + b"\xff\xd9"))
        self.assertTrue(store.update(b"\xff\xd8ok\xff\xd9"))
        self.assertEqual(store.snapshot().sequence, 1)

    def test_wait_after_returns_only_a_newer_frame(self):
        store = MODULE.LatestJpeg()
        self.assertIsNone(store.wait_after(0, 0.001))
        store.update(b"\xff\xd8one\xff\xd9")
        self.assertEqual(store.wait_after(0, 0.001).sequence, 1)
        self.assertIsNone(store.wait_after(1, 0.001))


class ArgumentTests(unittest.TestCase):
    def test_defaults_to_corrected_left_head_and_ten_hz(self):
        args = MODULE.build_parser().parse_args([])
        MODULE.validate_args(args)
        self.assertEqual(args.topic, MODULE.DEFAULT_TOPIC)
        self.assertEqual(args.fps, 10.0)

    def test_rejects_non_absolute_topic_and_more_than_ten_hz(self):
        parser = MODULE.build_parser()
        with self.assertRaises(ValueError):
            MODULE.validate_args(parser.parse_args(["--topic", "relative/topic"]))
        with self.assertRaises(ValueError):
            MODULE.validate_args(parser.parse_args(["--fps", "10.1"]))


class PreviewHTTPTests(unittest.TestCase):
    def setUp(self):
        self.store = MODULE.LatestJpeg()
        self.server = MODULE.PreviewHTTPServer(
            (MODULE.LOOPBACK_ADDRESS, 0),
            self.store,
            topic=MODULE.DEFAULT_TOPIC,
            fps=10.0,
            stale_after=2.0,
        )
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def test_index_is_display_only(self):
        with request.urlopen(self.base_url + "/", timeout=2) as response:
            body = response.read().decode("utf-8")
        self.assertIn("Corrected left-head camera", body)
        self.assertIn(MODULE.DEFAULT_TOPIC, body)
        self.assertNotIn("<button", body)
        self.assertNotIn("<form", body)

    def test_health_fails_closed_until_a_fresh_frame_arrives(self):
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(self.base_url + "/healthz", timeout=2)
        self.assertEqual(caught.exception.code, 503)
        caught.exception.close()

        self.store.update(b"\xff\xd8frame\xff\xd9")
        with request.urlopen(self.base_url + "/healthz", timeout=2) as response:
            payload = json.load(response)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["fps"], 10.0)

    def test_snapshot_returns_latest_jpeg_without_caching(self):
        jpeg = b"\xff\xd8frame\xff\xd9"
        self.store.update(jpeg)
        with request.urlopen(self.base_url + "/snapshot.jpg", timeout=2) as response:
            self.assertEqual(response.read(), jpeg)
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            self.assertIn("no-store", response.headers["Cache-Control"])

    def test_mutating_http_methods_are_not_available(self):
        call = request.Request(self.base_url + "/", method="POST", data=b"capture=true")
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(call, timeout=2)
        self.assertEqual(caught.exception.code, 405)
        self.assertEqual(caught.exception.headers["Allow"], "GET, HEAD")
        caught.exception.close()

    def test_mjpeg_stream_contains_latest_frame(self):
        jpeg = b"\xff\xd8" + b"x" * 512 + b"\xff\xd9"
        self.store.update(jpeg)
        with request.urlopen(self.base_url + "/stream.mjpg", timeout=2) as response:
            chunk = response.read(256)
        self.assertIn(MODULE.MJPEG_BOUNDARY.encode("ascii"), chunk)
        self.assertIn(b"Content-Type: image/jpeg", chunk)

    def test_server_rejects_non_loopback_binding(self):
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            MODULE.PreviewHTTPServer(
                ("0.0.0.0", 0),
                self.store,
                topic=MODULE.DEFAULT_TOPIC,
                fps=5.0,
                stale_after=2.0,
            )


if __name__ == "__main__":
    unittest.main()
