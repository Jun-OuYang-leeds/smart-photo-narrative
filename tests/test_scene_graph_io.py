from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scene_graph_io import (
    RESULT_SCHEMA_VERSION,
    audit_scene_graph_results,
    export_scene_graph_batch,
    load_scene_graph_manifest,
    write_indexable_scene_graphs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLOUD_SCRIPT = PROJECT_ROOT / "scripts" / "cloud" / "generate_scene_graph_jsonl.py"
SBATCH_SCRIPT = PROJECT_ROOT / "scripts" / "cloud" / "run_scene_graph_array_3gpu.sbatch"


def _load_cloud_module():
    spec = importlib.util.spec_from_file_location("spn_cloud_scene_graph", CLOUD_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SceneGraphIoTests(unittest.TestCase):
    def make_jpeg(self, path: Path, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (9, 7), color=color).save(path, format="JPEG", quality=91)

    def write_jsonl(self, path: Path, rows: list[dict] | None = None, raw: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(row, ensure_ascii=False) for row in (rows or [])]
        lines.extend(raw or [])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def successful_result(self, manifest, *, triples=None, source="remote") -> dict:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "dataset_id": manifest.dataset_id,
            "photo_id": manifest.photo_id,
            "sha256": manifest.sha256,
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "prompt_version": "spn-scene-graph-v1",
            "max_triples": 15,
            "source": source,
            "status": "success",
            "scene_graph": triples
            if triples is not None
            else [{"subject": "person", "predicate": "hold", "object": "phone"}],
            "generated_at": "2026-01-01T00:00:00+00:00",
        }

    def test_export_contract_and_cloud_names_are_hash_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album = root / "album"
            batch = root / "batch"
            self.make_jpeg(album / "nested" / "holiday.jpeg", (10, 20, 30))

            report = export_scene_graph_batch(album, batch, "personal-main-v1")
            records = load_scene_graph_manifest(batch / "manifest.jsonl")

            self.assertEqual(report.exported, 1)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(
                set(record.to_dict()),
                {"schema_version", "dataset_id", "photo_id", "relative_path", "sha256", "file_size"},
            )
            self.assertEqual(record.relative_path, "nested/holiday.jpeg")
            self.assertEqual((batch / "images" / f"{record.photo_id}.jpg").read_bytes(), (album / "nested" / "holiday.jpeg").read_bytes())

    def test_legacy_geograph_against_personal_album_reports_67_collisions_and_applies_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            personal = root / "personal"
            legacy = root / "geograph"
            batch = root / "batch"
            rows = []
            for index in range(67):
                name = f"{index + 1}.jpg"
                self.make_jpeg(personal / name, ((index * 3) % 256, 10, 20))
                self.make_jpeg(legacy / name, (20, (index * 3 + 100) % 256, 30))
                rows.append(
                    {
                        "frame_id": str(index + 1),
                        "image_path": name,
                        "scene_graph": [
                            {"subject": "cliff", "predicate": "meet", "object": "sea"}
                        ],
                        "scene_graph_source": "remote",
                    }
                )
            export_scene_graph_batch(personal, batch, "personal-main-v1")
            results = root / "old_qwen.jsonl"
            self.write_jsonl(results, rows)

            report = audit_scene_graph_results(
                batch / "manifest.jsonl",
                [results],
                personal,
                legacy_root=legacy,
            )

            self.assertEqual(report.counts["checksum_conflict"], 67)
            self.assertEqual(report.counts["matched"], 0)
            self.assertEqual(report.counts["indexable"], 0)
            self.assertEqual(report.indexable_records, [])

    def test_same_name_or_stem_never_overrides_modern_checksum_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album = root / "album"
            batch = root / "batch"
            self.make_jpeg(album / "one" / "same.jpg", (1, 2, 3))
            self.make_jpeg(album / "two" / "same.jpg", (4, 5, 6))
            export_scene_graph_batch(album, batch, "personal-main-v1")
            manifests = load_scene_graph_manifest(batch / "manifest.jsonl")
            first, second = manifests
            row = self.successful_result(first)
            row["sha256"] = second.sha256
            results = root / "results.jsonl"
            self.write_jsonl(results, [row])

            report = audit_scene_graph_results(batch / "manifest.jsonl", [results], album)

            self.assertEqual(report.counts["checksum_conflict"], 1)
            self.assertEqual(report.counts["matched"], 0)

    def test_renamed_photo_matches_only_by_same_id_and_full_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album = root / "album"
            batch = root / "batch"
            original = album / "old-name.jpg"
            renamed = album / "renamed" / "new-name.jpg"
            self.make_jpeg(original, (30, 40, 50))
            export_scene_graph_batch(album, batch, "personal-main-v1")
            manifest = load_scene_graph_manifest(batch / "manifest.jsonl")[0]
            renamed.parent.mkdir(parents=True)
            original.rename(renamed)
            results = root / "results.jsonl"
            self.write_jsonl(results, [self.successful_result(manifest)])

            report = audit_scene_graph_results(batch / "manifest.jsonl", [results], album)

            self.assertEqual(report.counts["matched"], 1)
            self.assertEqual(report.indexable_records[0]["relative_path"], "renamed/new-name.jpg")
            self.assertEqual(report.indexable_records[0]["manifest_relative_path"], "old-name.jpg")

    def test_export_import_and_apply_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album = root / "album"
            batch = root / "batch"
            self.make_jpeg(album / "a.jpg", (70, 80, 90))
            export_scene_graph_batch(album, batch, "personal-main-v1")
            manifest_bytes = (batch / "manifest.jsonl").read_bytes()
            manifest = load_scene_graph_manifest(batch / "manifest.jsonl")[0]
            cloud_bytes = (batch / "images" / f"{manifest.photo_id}.jpg").read_bytes()

            second_export = export_scene_graph_batch(album, batch, "personal-main-v1")
            self.assertEqual(second_export.reused_cloud_images, 1)
            self.assertEqual((batch / "manifest.jsonl").read_bytes(), manifest_bytes)
            self.assertEqual((batch / "images" / f"{manifest.photo_id}.jpg").read_bytes(), cloud_bytes)

            results = root / "results.jsonl"
            self.write_jsonl(results, [self.successful_result(manifest)])
            first = audit_scene_graph_results(batch / "manifest.jsonl", [results], album)
            second = audit_scene_graph_results(batch / "manifest.jsonl", [results], album)
            self.assertEqual(first.to_dict(include_records=True), second.to_dict(include_records=True))

            clean = root / "clean.jsonl"
            write_indexable_scene_graphs(first, clean)
            clean_bytes = clean.read_bytes()
            write_indexable_scene_graphs(second, clean)
            self.assertEqual(clean.read_bytes(), clean_bytes)

    def test_malformed_empty_failed_and_duplicate_rows_are_not_silently_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album = root / "album"
            batch = root / "batch"
            self.make_jpeg(album / "a.jpg", (100, 110, 120))
            export_scene_graph_batch(album, batch, "personal-main-v1")
            manifest = load_scene_graph_manifest(batch / "manifest.jsonl")[0]

            empty = self.successful_result(manifest, triples=[])
            failed = self.successful_result(manifest)
            failed.update({"source": "failed", "status": "failed", "scene_graph": []})
            first_valid = self.successful_result(manifest)
            second_valid = self.successful_result(
                manifest,
                triples=[{"subject": "cup", "predicate": "be on", "object": "table"}],
                source="remote_repaired",
            )
            results = root / "results.jsonl"
            self.write_jsonl(results, [empty, failed, first_valid, second_valid], raw=["{not-json"])

            report = audit_scene_graph_results(batch / "manifest.jsonl", [results], album)

            self.assertEqual(report.counts["malformed"], 1)
            self.assertEqual(report.counts["empty"], 1)
            self.assertEqual(report.counts["failed"], 1)
            self.assertEqual(report.counts["duplicate"], 3)
            self.assertEqual(report.counts["matched"], 1)
            self.assertEqual(report.indexable_records[0]["source"], "remote_repaired")

    def test_verified_legacy_rows_are_indexable_only_for_geograph_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album = root / "geograph"
            batch = root / "batch"
            self.make_jpeg(album / "nested" / "1.jpg", (130, 140, 150))
            export_scene_graph_batch(album, batch, "geograph-v1")
            results = root / "legacy.jsonl"
            self.write_jsonl(
                results,
                [
                    {
                        "image_path": "nested/1.jpg",
                        "scene_graph": [
                            {"subject": "road", "predicate": "pass", "object": "field"}
                        ],
                        "scene_graph_source": "remote",
                    }
                ],
            )

            report = audit_scene_graph_results(
                batch / "manifest.jsonl", [results], album, legacy_root=album
            )

            self.assertEqual(report.counts["matched"], 1)
            self.assertEqual(report.counts["checksum_conflict"], 0)

    def test_checkpoint_skips_only_successful_nonempty_exact_provenance(self) -> None:
        module = _load_cloud_module()
        self.assertIn('"scene_graph"', module.SYSTEM_PROMPT.format(max_triples=15))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "checkpoint.jsonl"
            base = {
                "dataset_id": "personal-main-v1",
                "photo_id": "a" * 32,
                "sha256": "b" * 64,
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "prompt_version": "spn-scene-graph-v1",
                "max_triples": 15,
            }
            failed = dict(base, source="failed", status="failed", scene_graph=[])
            empty = dict(base, source="remote", status="success", scene_graph=[])
            valid = dict(
                base,
                source="remote",
                status="success",
                scene_graph=[{"subject": "a", "predicate": "on", "object": "b"}],
            )
            self.write_jsonl(output, [failed, empty])
            self.assertEqual(module.load_successful_checkpoint_keys(output), set())
            self.write_jsonl(output, [failed, empty, valid])
            self.assertEqual(len(module.load_successful_checkpoint_keys(output)), 1)

    def test_cloud_generator_primary_and_repair_paths_return_bounded_triples(self) -> None:
        module = _load_cloud_module()
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "photo.jpg"
            self.make_jpeg(image, (1, 20, 200))
            calls = []

            def fake_chat(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    return "not valid json"
                return json.dumps(
                    {
                        "scene_graph": [
                            {"subject": "person", "predicate": "hold", "object": "camera"}
                        ]
                    }
                )

            original = module._chat
            module._chat = fake_chat
            try:
                triples, source = module._generate_one(
                    image_path=image,
                    base_url="http://127.0.0.1:8000/v1",
                    api_key="not-logged",
                    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                    max_triples=15,
                    max_tokens=640,
                    timeout=1.0,
                )
            finally:
                module._chat = original

            self.assertEqual(source, "remote_repaired")
            self.assertEqual(triples, [{"subject": "person", "predicate": "hold", "object": "camera"}])
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["max_tokens"], 640)

    def test_array_job_uses_three_one_gpu_shards_and_validated_limits(self) -> None:
        text = SBATCH_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-2%3", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("--max-model-len 1536", text)
        self.assertIn("--max-tokens 640", text)
        self.assertIn("--max-triples 15", text)
        self.assertIn("--num-shards 3", text)


if __name__ == "__main__":
    unittest.main()
