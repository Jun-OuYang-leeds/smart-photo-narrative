"""Offline Streamlit smoke test for Library and the Story v3 evidence preview."""

from __future__ import annotations

from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def assert_no_exceptions(app: AppTest, stage: str) -> None:
    if app.exception:
        details = "; ".join(str(item.value) for item in app.exception)
        raise AssertionError(f"{stage} raised Streamlit exceptions: {details}")


def main() -> None:
    # Cold CLIP/Streamlit startup can exceed 60 seconds immediately after a
    # long Ollama run on the laptop GPU.  This remains an offline UI test; the
    # larger bound avoids treating resource contention as an app exception.
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert_no_exceptions(app, "Library")
    navigation = app.sidebar.radio[0]
    navigation.set_value("📖 Story").run()
    assert_no_exceptions(app, "Story basket")
    source = next(item for item in app.radio if item.label == "Story source")
    source.set_value("Date").run()
    assert_no_exceptions(app, "Story Date evidence plan")
    if not any("Evidence-group preview" in item.label for item in app.expander):
        raise AssertionError("Story v3 evidence-group preview was not rendered")
    if not any(item.label.startswith("Verified context") for item in app.text_area):
        raise AssertionError("Story verified-context input was not rendered")
    print("Streamlit AppTest: Library, Story basket, and Story Date preview passed with zero exceptions")


if __name__ == "__main__":
    main()
