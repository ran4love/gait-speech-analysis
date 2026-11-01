"""Translation completeness without modifying measurements or user-authored data."""

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "adualnet_toolkit"
CJK = re.compile(r"[\u4e00-\u9fff]")


class StaticUI(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.text = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in ("meta", "br", "input"):
            self.tags.append(tag)
        for key, value in attrs:
            if key in ("placeholder", "title", "aria-label") and CJK.search(value or ""):
                self.text.append(value)

    def handle_endtag(self, tag):
        if tag in self.tags:
            self.tags = self.tags[:len(self.tags)-1-self.tags[::-1].index(tag)]

    def handle_data(self, value):
        if not set(self.tags).intersection({"script", "style", "code", "pre", "textarea"}):
            value = value.strip()
            if CJK.search(value) and value != "中文":
                self.text.append(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate translation key: " + key)
        result[key] = value
    return result


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((UI / "ui_translations.json").read_text(), object_pairs_hook=unique_object)

    def test_every_static_chinese_label_and_placeholder_has_english(self):
        parser = StaticUI()
        parser.feed((UI / "ui.html").read_text())
        self.assertFalse(set(parser.text) - self.catalog.keys())

    def test_placeholders_match_and_english_values_are_complete(self):
        for source, target in self.catalog.items():
            with self.subTest(source=source):
                self.assertTrue(target.strip())
                self.assertFalse(CJK.search(target))
                self.assertEqual(set(re.findall(r"\{\w+\}", source)), set(re.findall(r"\{\w+\}", target)))

    def test_backend_video_progress_has_translations(self):
        code = (UI / "video.py").read_text()
        messages = re.findall(r'state\(job, "[^"]+", "([^"]+)"', code)
        self.assertTrue(messages)
        self.assertFalse({s for s in messages if CJK.search(s)} - self.catalog.keys())
        self.assertIn("启动视频流程", self.catalog)

    def test_translator_does_not_scan_dynamic_user_data(self):
        code = (UI / "ui_i18n.js").read_text()
        self.assertNotIn("MutationObserver", code)
        self.assertNotIn("innerHTML", code)
        self.assertIn("script,style,textarea,input,pre,code", code)
        self.assertNotIn(".value =", code)
        self.assertIn("adualnet_language", code)

    def test_interface_catalog_is_embedded_and_translation_script_is_authenticated(self):
        from adualnet_toolkit.web import make_handler
        with tempfile.TemporaryDirectory() as tmp:
            handler = object.__new__(make_handler(tmp, "synthetic-session-token", 8765))
            handler.path = "/"
            handler.headers = {"Host": "127.0.0.1:8765"}
            replies = []
            handler.send = lambda content, content_type=None, status=200: replies.append((content, status))
            handler.do_GET()
            html = replies.pop()[0].decode()
            self.assertNotIn("__UI_TRANSLATIONS__", html)
            self.assertNotIn("__LOCAL_TOKEN__", html)
            self.assertIn("/assets/ui_i18n.js?token=synthetic-session-token", html)
            handler.path = "/assets/ui_i18n.js"
            handler.do_GET()
            self.assertEqual(replies.pop()[1], 403)
            handler.headers["X-ADualNet-Token"] = "synthetic-session-token"
            handler.do_GET()
            self.assertEqual(replies.pop()[0], (UI / "ui_i18n.js").read_bytes())

    def test_workspace_assets_are_served_only_by_explicit_allowlist(self):
        from adualnet_toolkit.web import make_handler
        with tempfile.TemporaryDirectory() as tmp:
            handler = object.__new__(make_handler(tmp, "synthetic-token", 8765))
            handler.headers = {"Host": "127.0.0.1:8765", "X-ADualNet-Token": "synthetic-token"}
            replies = []
            handler.send = lambda content, content_type=None, status=200: replies.append((content, content_type, status))
            for name, mime in [("workspace.css", "text/css"), ("workspace_ui.js", "text/javascript")]:
                handler.path = "/assets/" + name
                handler.do_GET()
                content, content_type, status = replies.pop()
                self.assertEqual(status, 200)
                self.assertTrue(content_type.startswith(mime))
                self.assertEqual(content, (UI / name).read_bytes())
            handler.path = "/assets/recognition.local.json"
            handler.do_GET()
            self.assertEqual(replies.pop()[2], 404)

    def test_workspace_preserves_controls_and_is_presentation_only(self):
        html = (UI / "ui.html").read_text()
        for control in ["video-start", "video-env", "gait-reviewed", "speech-reviewed", "confirm-features", "save-review"]:
            self.assertEqual(html.count('id="' + control + '"'), 1)
        for tab in ["video-panel", "demo", "data", "raw"]:
            self.assertEqual(html.count('data-tab="' + tab + '"'), 1)
        self.assertEqual(html.count("data-workflow-step="), 4)
        script = (UI / "workspace_ui.js").read_text()
        self.assertNotIn("fetch(", script)
        self.assertNotIn(".value =", script)
        self.assertNotIn(".checked =", script)
        self.assertIn("prefers-reduced-motion", script)


if __name__ == "__main__":
    unittest.main()
