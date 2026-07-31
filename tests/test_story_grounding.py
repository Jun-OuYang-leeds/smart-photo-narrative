from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config import STORY_NARRATIVE_MAX_PARAGRAPHS
from retrieval_engine import MultimodalRetriever
from storage import PhotoStorage
from story_agent import (
    ContextAggregator,
    DashScopeGenerator,
    FirstPersonMemoirPromptBuilder,
    FirstPersonMemoirValidator,
    GroundingValidator,
    LayeredCreativePromptBuilder,
    PhotographerMoodPromptBuilder,
    PhotographerMoodValidator,
    PromptBuilder,
    StoryGenerator,
)


class FakeVectors:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or {}

    def get_image_embeddings(self, photo_ids):
        return {photo_id: self.embeddings[photo_id] for photo_id in photo_ids if photo_id in self.embeddings}


class NoClip:
    pass


class FakeGenerator:
    model = "fake"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return next(self.outputs)

    def is_available(self):
        return True


class StoryGroundingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = PhotoStorage(Path(self.tmp.name) / "app.db")
        self.photo_id = self._add_photo(
            "one.jpg", "a", "2026-01-01T09:00:00", "a cup on a wooden table",
        )
        self.aggregator = self._aggregator()

    def tearDown(self):
        self.tmp.cleanup()

    def _add_photo(self, path, hash_letter, timestamp, caption, *, confidence=1.0, location=None):
        photo_id = self.storage.upsert_photo(
            relative_path=path, content_sha256=hash_letter * 64, captured_at=timestamp,
            date_local=timestamp[:10], timestamp_source="exif_original", timestamp_confidence=confidence,
            location=location, image_width=10, image_height=10,
        )
        self.storage.upsert_caption(photo_id, caption)
        return photo_id

    def _aggregator(self, embeddings=None):
        retriever = MultimodalRetriever(self.storage, FakeVectors(embeddings), NoClip())
        return ContextAggregator(retriever, self.storage)

    @staticmethod
    def _payload(context, texts=None, *, language="en", uncertain=None, transitions=None, unused=None):
        texts = texts or [
            ("证据组记录了可直接观察到的场景。" if language == "zh" else "The evidence group shows an observable scene.")
            for _ in context.groups
        ]
        return {
            "title": "事件记录" if language == "zh" else "Event record",
            "paragraphs": [{
                "text": text,
                "evidence_ids": list(group.evidence_ids),
                "group_id": group.group_id,
            } for text, group in zip(texts, context.groups)],
            "uncertain_observations": uncertain or [],
            "unused_evidence_ids": unused or [],
            "creative_transitions": transitions or [],
        }

    @staticmethod
    def _faithful_groups_payload(context, *, language="en", facts=None, title=None):
        """The grouped faithful contract the model now returns under the v3 schema:
        ``{title, groups[{group_id, factual_text}]}`` with no evidence bookkeeping."""
        if language == "zh":
            default_facts = [
                f"第{index}组照片记录了可直接观察到的场景与物体。"
                for index in range(1, len(context.groups) + 1)
            ]
            default_title = "事件记录"
        else:
            default_facts = [
                f"Group {index} records a directly observable scene and its objects."
                for index in range(1, len(context.groups) + 1)
            ]
            default_title = "Event record"
        selected = facts or default_facts[:len(context.groups)]
        return {
            "title": title or default_title,
            "groups": [
                {"group_id": group.group_id, "factual_text": fact}
                for group, fact in zip(context.groups, selected)
            ],
        }

    @staticmethod
    def _layered_payload(context, *, language="en", transitions=None):
        facts = (
            [f"第{index}组照片记录了清晰可见的场景与物体。" for index in range(1, len(context.groups) + 1)]
            if language == "zh"
            else [f"Group {index} records a distinct visible scene and its objects." for index in range(1, len(context.groups) + 1)]
        )
        if transitions is None:
            count = LayeredCreativePromptBuilder.transition_count(context)
            transitions = (
                [f"我把第{index}段间隙想象成一页轻轻翻过的相册。" for index in range(1, count + 1)]
                if language == "zh"
                else [f"I imagine pause {index} as a quiet turn of the album page." for index in range(1, count + 1)]
            )
        return {
            "title": "分层照片故事" if language == "zh" else "Layered photo story",
            "groups": [
                {"group_id": group.group_id, "factual_text": fact}
                for group, fact in zip(context.groups, facts)
            ],
            "creative_transitions": transitions,
        }

    @staticmethod
    def _memoir_payload(context, *, language="en", transitions=None, title=None, facts=None):
        if language == "zh":
            opening = "我重新翻看这组照片时，先把注意力放在画面中清楚可见的细节上，让这些片段按照原有顺序自然展开。"
            default_facts = [
                "我从画面中看到木桌上放着一只杯子，桌面的纹理、杯子的轮廓和周围留白共同组成了安静而具体的日常场景。",
                "我从下一幅画面中看到一扇蓝色门和周围清楚的墙面结构，颜色与前一组室内物体形成了明显变化。",
                "我在最后的画面中看到一棵绿色树木占据视线，枝叶和背景把这段由室内走向室外的观察完整收住。",
            ]
            default_transitions = [
                "我把视线从桌面的近处细节移向下一幅画面，场景的变化让我继续留意颜色和空间之间的联系。",
                "我继续沿着照片的顺序向后看，门边的人工结构逐渐让位于更开阔的树木和自然背景。",
            ]
            closing = "我看完这些照片后，留下的是一段由具体物体、空间变化和自然景象连接起来的平实记忆。"
            title = title or "照片里的连续回忆"
        else:
            opening = (
                "I return to these photographs with careful attention to the ordinary details that the camera preserved, "
                "letting the sequence guide the memory rather than adding an event that was never shown."
            )
            default_facts = [
                "Looking at the first photograph, I see a cup resting on a wooden table, with the clear rim, the surface "
                "grain, and the open space around it forming a specific record of an everyday scene.",
                "In the next photograph, I see a blue door set against a visible wall, and I notice how its color and "
                "straight edges change the visual rhythm established by the smaller object in the previous scene.",
                "In the final photograph, I see a green tree filling the view, with its branches and background bringing "
                "the movement from an interior object toward a broader outdoor scene to a clear end.",
            ]
            default_transitions = [
                "I move my attention from the close view of the table toward the next frame, carrying the earlier focus "
                "on shape into a scene defined more strongly by color and structure.",
                "I continue through the sequence as the built lines around the door give way to branches, foliage, and "
                "a more open arrangement in the final view.",
            ]
            closing = (
                "I finish the sequence with a grounded memory of objects and spaces changing from one frame to the next, "
                "held together by what remained plainly visible in the photographs."
            )
            title = title or "A continuous photo memory"
        selected_facts = facts or default_facts[:len(context.groups)]
        if transitions is None:
            transitions = default_transitions[:FirstPersonMemoirPromptBuilder.transition_count(context)]
        return {
            "title": title,
            "opening": opening,
            "groups": [
                {"group_id": group.group_id, "factual_text": fact}
                for group, fact in zip(context.groups, selected_facts)
            ],
            "transitions": transitions,
            "closing": closing,
        }

    def test_story_evidence_carries_no_clip_tag_observations(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        item = context.evidence[0]

        # CLIP preset tags were removed: they must not reach the model prompt,
        # the grouping text, or the person-evidence check that gates first
        # person narration in creative mode.
        self.assertNotIn("clip_tags_model_observations", item.observation_record())
        self.assertEqual(item.tags, ())
        self.assertNotIn("clip_tags", context.to_prompt_context())
        system, user = PromptBuilder.build_story_prompt(context, mode="faithful", language="en")
        self.assertNotIn("clip_tags", system + user)

    def test_unknown_citation_is_rejected(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        payload = self._payload(context)
        payload["paragraphs"][0]["evidence_ids"] = ["P999"]
        self.assertTrue(GroundingValidator.validate(payload, context, "faithful", "en"))

    def test_date_only_label_or_evidence_echo_is_rejected(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id], label="Trip")
        for degenerate in ("2026-01-01", "2026-01-01T09:00:00", "Trip", "P001"):
            payload = self._payload(context, [degenerate])
            self.assertTrue(GroundingValidator.validate(payload, context, "faithful", "en"))

    def test_faithful_validator_default_bans_speculation_and_first_person(self):
        # The validator itself still enforces the full ban by default (no
        # speculation_terms override). This is the behaviour the production
        # switch restores when STORY_FAITHFUL_SPECULATION_BAN is True, and it
        # stays in force for the frozen N0--N3 / QF--QC experiments.
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        for claim in (
            "A person is possibly washing.", "A person is likely working.",
            "A person might be studying.", "A person is getting ready.",
            "I felt happy because my daughter arrived.",
        ):
            payload = self._payload(context, [claim])
            errors = GroundingValidator.validate(payload, context, "faithful", "en")
            self.assertTrue(any("E_UNSUPPORTED_CLAIM" in error or "E_FIRST_PERSON" in error for error in errors))

    def test_faithful_production_accepts_speculation_when_ban_disabled(self):
        # Default switch is off: production faithful no longer hard-rejects the
        # 46 speculation/emotion/causal/kinship terms.
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        raw = self._faithful_groups_payload(
            context, facts=["A person is likely working beside the visible table."],
        )
        story = StoryGenerator(
            aggregator=self.aggregator, generator=FakeGenerator([json.dumps(raw)]), storage=self.storage,
        ).generate_story_with_context(context, mode="faithful", language="en", save=False)
        self.assertEqual(story.status, "ok", story.validation_codes)
        self.assertFalse(any("E_UNSUPPORTED_CLAIM" in code for code in story.validation_codes))

    def test_faithful_production_rejects_speculation_when_ban_enabled(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        raw = self._faithful_groups_payload(
            context, facts=["A person is likely working beside the visible table."],
        )
        invalid = json.dumps(raw)
        with patch("story_agent.STORY_FAITHFUL_SPECULATION_BAN", True):
            story = StoryGenerator(
                aggregator=self.aggregator, generator=FakeGenerator([invalid, invalid]), storage=self.storage,
            ).generate_story_with_context(
                context, mode="faithful", language="en", save=False, allow_deterministic_fallback=False,
            )
        self.assertEqual(story.status, "error")
        self.assertTrue(any(
            "E_UNSUPPORTED_CLAIM" in code for diag in story.attempt_diagnostics for code in diag["codes"]
        ))

    def test_verified_context_can_support_an_explicit_personal_fact(self):
        context = self.aggregator.aggregate_by_photo_ids(
            [self.photo_id], verified_context="The person is studying.",
        )
        payload = self._payload(context, ["The person is studying."])
        errors = GroundingValidator.validate(payload, context, "faithful", "en")
        self.assertFalse(any("E_UNSUPPORTED_CLAIM" in error for error in errors))

    def test_chinese_request_rejects_all_english_output(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        payload = self._payload(context, language="en")
        errors = GroundingValidator.validate(payload, context, "faithful", "zh")
        self.assertTrue(any("E_LANGUAGE_ZH" in error for error in errors))

    def test_english_request_rejects_chinese_output(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        payload = self._payload(context, language="zh")
        errors = GroundingValidator.validate(payload, context, "faithful", "en")
        self.assertTrue(any("E_LANGUAGE_EN" in error for error in errors))

    def test_creative_transition_is_marked_but_first_person_fact_is_rejected(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        valid = self._payload(context, transitions=["I remember the quiet pause between these frames."])
        self.assertFalse(GroundingValidator.validate(valid, context, "creative", "en"))
        invalid = self._payload(context, ["I remember a cup on the table."], transitions=[])
        self.assertTrue(any(
            "E_FIRST_PERSON_IN_FACT" in error
            for error in GroundingValidator.validate(invalid, context, "creative", "en")
        ))

    def test_creative_fact_rejects_unsupported_speculation(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        payload = self._payload(
            context,
            ["A person is likely working beside the visible table."],
            transitions=["I imagine the room becoming quiet after this frame."],
        )
        errors = GroundingValidator.validate(payload, context, "creative", "en")
        self.assertTrue(any("E_UNSUPPORTED_CLAIM" in error for error in errors))

    def test_layered_schema_fixes_group_and_transition_counts(self):
        ids = [self.photo_id]
        ids.append(self._add_photo("two.jpg", "b", "2026-01-01T11:00:00", "a blue door"))
        ids.append(self._add_photo("three.jpg", "c", "2026-01-01T13:00:00", "a green tree"))
        context = self.aggregator.aggregate_by_photo_ids(ids)
        schema = LayeredCreativePromptBuilder.output_schema(context)
        self.assertEqual(schema["properties"]["groups"]["minItems"], 3)
        self.assertEqual(schema["properties"]["groups"]["maxItems"], 3)
        self.assertEqual(schema["properties"]["creative_transitions"]["minItems"], 2)
        self.assertEqual(
            schema["properties"]["groups"]["items"]["properties"]["group_id"]["enum"],
            ["G001", "G002", "G003"],
        )

    def test_first_person_memoir_injects_citations_and_accepts_observer_voice(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        raw = self._memoir_payload(context)
        fake = FakeGenerator([json.dumps(raw)])
        story = StoryGenerator(
            aggregator=self.aggregator, generator=fake, storage=self.storage,
        ).generate_story_with_context(context, mode="creative", language="en", save=False)
        self.assertEqual(story.status, "ok", story.validation_codes)
        self.assertEqual(story.citations["1"], ["P001"])
        self.assertEqual(story.creative_transitions, raw["transitions"])
        self.assertEqual(story.opening, raw["opening"])
        self.assertEqual(story.closing, raw["closing"])
        self.assertEqual(story.prompt_version, "grounded-story-v6.2-deterministic-photographer-mood")
        self.assertIn("output_schema", fake.calls[0][1])
        self.assertNotIn("evidence_ids", json.dumps(raw))

    def test_first_person_memoir_chinese_validates_each_text_field(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        candidate = self._memoir_payload(context, language="zh")
        normalized = FirstPersonMemoirPromptBuilder.normalise_payload(candidate, context)
        self.assertFalse(FirstPersonMemoirValidator.validate(normalized, context, "zh"))

    def test_chinese_memoir_rejects_english_title_even_when_body_is_chinese(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        candidate = self._memoir_payload(context, language="zh", title="Antalya Horizon")
        normalized = FirstPersonMemoirPromptBuilder.normalise_payload(candidate, context)
        errors = FirstPersonMemoirValidator.validate(normalized, context, "zh")
        self.assertTrue(any("E_FIELD_LANGUAGE_ZH: title" in error for error in errors))

    def test_chinese_memoir_allows_implicit_first_person_in_closing(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        candidate = self._memoir_payload(context, language="zh")
        candidate["closing"] = "看完这些清楚可见的细节后，留下的是一段由普通物体和安静空间连接起来的平实记忆。"
        normalized = FirstPersonMemoirPromptBuilder.normalise_payload(candidate, context)
        errors = FirstPersonMemoirValidator.validate(normalized, context, "zh")
        self.assertFalse(any("E_FIRST_PERSON_VOICE" in error for error in errors))

    def test_chinese_memoir_rejects_english_first_person_voice(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        candidate = self._memoir_payload(context, language="zh")
        candidate["opening"] = "I return to these photographs，重新查看画面中清楚可见的普通细节，并让记录按照原有顺序自然展开。"
        candidate["groups"][0]["factual_text"] = (
            "I see 木桌上放着一只杯子，桌面的纹理、杯子的轮廓以及周围留白仍然构成一段清楚具体的日常记录。"
        )
        normalized = FirstPersonMemoirPromptBuilder.normalise_payload(candidate, context)
        errors = FirstPersonMemoirValidator.validate(normalized, context, "zh")
        # The English opening still fails first-person-voice in a Chinese request.
        # The observer role no longer mandates a fixed "我从画面中看到" template,
        # so E_OBSERVER_ROLE is not raised.
        self.assertTrue(any("E_FIRST_PERSON_VOICE: opening" in error for error in errors))
        self.assertFalse(any("E_OBSERVER_ROLE" in error for error in errors))

    def test_observer_allows_varied_first_person_without_forcing_a_template(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id], narrator_role="observer")
        valid = FirstPersonMemoirPromptBuilder.normalise_payload(self._memoir_payload(context), context)
        self.assertFalse(FirstPersonMemoirValidator.validate(valid, context, "en"))
        # A varied first-person opening that is not the old observer template is
        # accepted now; the model chooses and varies its own openings.
        varied_fact = (
            "I linger over the wooden table and the visible cup, letting the clear rim, the grain of the "
            "surface, and the quiet space around it settle into a calm, ordinary memory of the afternoon."
        )
        varied = FirstPersonMemoirPromptBuilder.normalise_payload(
            self._memoir_payload(context, facts=[varied_fact]), context,
        )
        errors = FirstPersonMemoirValidator.validate(varied, context, "en")
        self.assertFalse(any("E_OBSERVER_ROLE" in error for error in errors))
        self.assertFalse(any("E_UNSUPPORTED_CLAIM" in error for error in errors))

    def test_confirmed_subject_allows_evidenced_action_but_not_unsupported_identity(self):
        person_id = self._add_photo(
            "person.jpg", "b", "2026-01-02T10:00:00", "a woman sits on a chair beside a table",
        )
        context = self.aggregator.aggregate_by_photo_ids([person_id], narrator_role="confirmed_subject")
        action = (
            "I sit on the visible chair beside the table, with the chair frame, the tabletop, and the surrounding room "
            "remaining clear enough to anchor this part of the memory in what the photograph actually contains."
        )
        accepted = FirstPersonMemoirPromptBuilder.normalise_payload(
            self._memoir_payload(context, facts=[action]), context,
        )
        self.assertFalse(FirstPersonMemoirValidator.validate(accepted, context, "en"))
        # Light feeling is now allowed, but a concrete identity/relationship
        # claim without verified_context is still rejected.
        unsupported = FirstPersonMemoirPromptBuilder.normalise_payload(
            self._memoir_payload(context, facts=[action + " My wife is waiting nearby."]), context,
        )
        errors = FirstPersonMemoirValidator.validate(unsupported, context, "en")
        self.assertTrue(any("E_UNSUPPORTED_CLAIM" in error for error in errors))

    def test_confirmed_subject_chinese_allows_implicit_subject_for_object_only_group(self):
        context = self.aggregator.aggregate_by_photo_ids(
            [self.photo_id], narrator_role="confirmed_subject",
        )
        candidate = self._memoir_payload(context, language="zh")
        candidate["groups"][0]["factual_text"] = (
            "木桌上放着一只杯子，清楚可见的杯沿、桌面纹理和周围留白组成了安静具体的日常片段，"
            "也让这一段叙述自然延续开头已经建立的第一人称视角。"
        )
        normalized = FirstPersonMemoirPromptBuilder.normalise_payload(candidate, context)
        errors = FirstPersonMemoirValidator.validate(normalized, context, "zh")
        self.assertFalse(any("E_FIRST_PERSON_VOICE: factual_text" in error for error in errors))
        self.assertFalse(any("E_CONFIRMED_SUBJECT_NO_PERSON" in error for error in errors))

    def test_confirmed_subject_still_rejects_narrator_action_without_person_evidence(self):
        context = self.aggregator.aggregate_by_photo_ids(
            [self.photo_id], narrator_role="confirmed_subject",
        )
        candidate = self._memoir_payload(context, language="zh")
        candidate["groups"][0]["factual_text"] = (
            "我坐在木桌旁拿起杯子，桌面的纹理、杯子的轮廓和周围空间构成了一段清楚具体的室内记录。"
        )
        normalized = FirstPersonMemoirPromptBuilder.normalise_payload(candidate, context)
        errors = FirstPersonMemoirValidator.validate(normalized, context, "zh")
        self.assertTrue(any("E_CONFIRMED_SUBJECT_NO_PERSON" in error for error in errors))

    def test_confirmed_subject_prompt_marks_person_evidence_per_group(self):
        person_id = self._add_photo(
            "prompt-person.jpg", "d", "2026-01-02T11:00:00", "a person walking beside a table",
        )
        context = self.aggregator.aggregate_by_photo_ids(
            [self.photo_id, person_id], narrator_role="confirmed_subject",
        )
        system, user = FirstPersonMemoirPromptBuilder.build_story_prompt(context, language="zh")
        self.assertIn("has_visible_person_evidence", user)
        self.assertIn('"has_visible_person_evidence": false', user)
        self.assertIn('"has_visible_person_evidence": true', user)
        self.assertIn("Natural Chinese may omit 我", system)

    def test_opening_transition_theme_echo_is_not_a_fatal_duplicate(self):
        ids = [self.photo_id]
        ids.append(self._add_photo("echo2.jpg", "e", "2026-01-01T11:00:00", "a blue door"))
        ids.append(self._add_photo("echo3.jpg", "f", "2026-01-01T13:00:00", "a green tree"))
        context = self.aggregator.aggregate_by_photo_ids(ids)
        candidate = self._memoir_payload(context, language="zh")
        candidate["transitions"][0] = candidate["opening"]
        normalized = FirstPersonMemoirPromptBuilder.normalise_payload(candidate, context)
        errors = FirstPersonMemoirValidator.validate(normalized, context, "zh")
        self.assertFalse(any("E_MEMOIR_REPETITION" in error for error in errors))

    def test_confirmed_subject_regression_generates_when_chinese_facts_omit_pronoun(self):
        ids = [self.photo_id]
        ids.append(self._add_photo("regression2.jpg", "1", "2026-01-01T11:00:00", "a blue door"))
        ids.append(self._add_photo("regression3.jpg", "2", "2026-01-01T13:00:00", "a green tree"))
        context = self.aggregator.aggregate_by_photo_ids(ids, narrator_role="confirmed_subject")
        raw = self._memoir_payload(context, language="zh")
        raw["groups"][0]["factual_text"] = (
            "木桌上的杯子、清楚可见的桌面纹理和周围留白先构成一个安静具体的日常片段，叙述由这些细节开始。"
        )
        raw["groups"][2]["factual_text"] = (
            "最后一幅画面由绿色树木、层叠枝叶和开阔背景占据，视觉范围从室内物体转到自然景象并在这里结束。"
        )
        raw["transitions"][0] = raw["opening"]
        story = StoryGenerator(
            aggregator=self.aggregator,
            generator=FakeGenerator([json.dumps(raw, ensure_ascii=False)]),
            storage=self.storage,
        ).generate_story_with_context(context, mode="creative", language="zh", save=False)
        self.assertEqual(story.status, "ok", story.validation_codes)
        self.assertEqual(len(story.paragraphs), 3)

    def test_reduce_groups_merges_heterogeneous_groups_to_target(self):
        ids = [self.photo_id]
        for index in range(2, 7):
            ids.append(self._add_photo(
                f"scene{index}.jpg", format(index, "x")[-1], f"2026-01-03T{index:02d}:00:00",
                f"a distinct indoor scene number {index}",
            ))
        context = self.aggregator.aggregate_by_photo_ids(ids)
        self.assertGreater(len(context.groups), 3)
        reduced = StoryGenerator._reduce_groups(context.groups, 3)
        self.assertEqual(len(reduced), 3)
        self.assertEqual([group.group_id for group in reduced], ["G001", "G002", "G003"])
        original = {ev for group in context.groups for ev in group.evidence_ids}
        merged = {ev for group in reduced for ev in group.evidence_ids}
        self.assertEqual(original, merged)

    def test_creative_generation_weaves_many_scenes_into_few_paragraphs(self):
        ids = [self.photo_id]
        for index in range(2, 7):
            ids.append(self._add_photo(
                f"weave{index}.jpg", format(index, "x")[-1], f"2026-01-04T{index:02d}:00:00",
                f"a distinct visible scene number {index}",
            ))
        context = self.aggregator.aggregate_by_photo_ids(ids)
        self.assertGreater(len(context.groups), STORY_NARRATIVE_MAX_PARAGRAPHS)
        reduced_groups = StoryGenerator._reduce_groups(context.groups, STORY_NARRATIVE_MAX_PARAGRAPHS)
        reduced_context = replace(context, groups=reduced_groups)
        facts = [
            "I begin with a quiet cup resting on the wooden table, and I take my time with the rim, the soft grain "
            "running under it, and the open space around the object, letting this first plain detail set a calm and "
            "unhurried tone for the whole sequence.",
            "My attention then moves to a flight of stairs rising past a bare wall, where I follow the steps, the simple "
            "railing, and the even light on each tread, noticing how the vertical lines change the rhythm after the small "
            "object I looked at first.",
            "A wide window comes next, and I rest on the glass, the plain frame, and the row of distant rooftops beyond it, "
            "aware that the view opens outward here while the earlier frames stayed close and enclosed within a single "
            "quiet room.",
            "The last frame gives me a green tree filling the view, and I linger on the branches, the layered leaves, and "
            "the soft background, feeling that this outdoor close brings the run of ordinary, clearly seen scenes to a "
            "settled and gentle end.",
        ][:len(reduced_groups)]
        raw = self._memoir_payload(reduced_context, facts=facts, transitions=[])
        story = StoryGenerator(
            aggregator=self.aggregator, generator=FakeGenerator([json.dumps(raw)]), storage=self.storage,
        ).generate_story_with_context(context, mode="creative", language="en", save=False)
        self.assertEqual(story.status, "ok")
        self.assertLessEqual(len(story.paragraphs), STORY_NARRATIVE_MAX_PARAGRAPHS)
        self.assertEqual(len(story.paragraphs), len(reduced_groups))

    def test_memoir_transitions_are_optional(self):
        second = self._add_photo("t2.jpg", "b", "2026-01-01T14:00:00", "a green tree outdoors")
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id, second])
        self.assertEqual(len(context.groups), 2)
        candidate = self._memoir_payload(context, transitions=[])
        normalized = FirstPersonMemoirPromptBuilder.normalise_payload(candidate, context)
        errors = FirstPersonMemoirValidator.validate(normalized, context, "en")
        self.assertFalse(any("E_CREATIVE_TRANSITION_COUNT" in error for error in errors))

    def test_memoir_allows_light_feeling_but_blocks_identity_activity_and_cause(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        feeling = FirstPersonMemoirPromptBuilder.normalise_payload(
            self._memoir_payload(context, facts=[
                "I settle my gaze on the cup and the wooden table, and the plain, quiet arrangement leaves me with a "
                "calm, unhurried impression of an ordinary afternoon spent close to these small, clearly visible things."
            ]),
            context,
        )
        self.assertFalse(any(
            "E_UNSUPPORTED_CLAIM" in error
            for error in FirstPersonMemoirValidator.validate(feeling, context, "en")
        ))
        for banned in (
            "I keep working beside the visible cup on the table.",
            "My wife left the cup on the table.",
            "The cup is empty because I finished the tea.",
        ):
            candidate = FirstPersonMemoirPromptBuilder.normalise_payload(
                self._memoir_payload(context, facts=[banned]), context,
            )
            errors = FirstPersonMemoirValidator.validate(candidate, context, "en")
            self.assertTrue(any("E_UNSUPPORTED_CLAIM" in error for error in errors), banned)

    def test_memoir_rejects_metadata_and_generic_poetic_cliche(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        candidate = self._memoir_payload(context, language="zh")
        candidate["opening"] += " 我在2026-01-01 09:00再次翻到这里。"
        candidate["closing"] += " 我仿佛时间被海风拉长。"
        normalized = FirstPersonMemoirPromptBuilder.normalise_payload(candidate, context)
        errors = FirstPersonMemoirValidator.validate(normalized, context, "zh")
        self.assertTrue(any("E_METADATA_IN_NARRATIVE" in error for error in errors))
        self.assertTrue(any("E_CREATIVE_CLICHE" in error for error in errors))

    def test_v5_schema_requires_three_groups_optional_transitions_opening_and_closing(self):
        ids = [self.photo_id]
        ids.append(self._add_photo("memoir2.jpg", "b", "2026-01-01T11:00:00", "a blue door"))
        ids.append(self._add_photo("memoir3.jpg", "c", "2026-01-01T13:00:00", "a green tree"))
        context = self.aggregator.aggregate_by_photo_ids(ids)
        schema = FirstPersonMemoirPromptBuilder.output_schema(context)
        self.assertEqual(schema["properties"]["groups"]["minItems"], 3)
        # Transitions are optional now: at most group_count-1, but the model may
        # omit them so it does not fabricate a link between unrelated scenes.
        self.assertEqual(schema["properties"]["transitions"]["minItems"], 0)
        self.assertEqual(schema["properties"]["transitions"]["maxItems"], 2)
        self.assertIn("opening", schema["required"])
        self.assertIn("closing", schema["required"])
        self.assertNotIn('"pattern"', json.dumps(schema))

    def test_event_summary_uses_only_reliable_time_and_deduplicates_locations(self):
        low = self._add_photo(
            "low.jpg", "b", "2025-12-31T23:00:00", "a dark room",
            confidence=0.2, location="Leeds",
        )
        reliable = self._add_photo(
            "reliable.jpg", "c", "2026-01-01T11:15:00", "a green tree",
            confidence=0.6, location="Leeds",
        )
        context = self.aggregator.aggregate_by_photo_ids([low, reliable])
        summary = StoryGenerator._event_summary(context)
        self.assertEqual(summary["date"], "2026-01-01")
        self.assertEqual(summary["time_range"], "11:15")
        self.assertEqual(summary["locations"], ["Leeds"])
        self.assertEqual(summary["reliable_time_count"], 1)

    def test_creative_repair_prompt_contains_rejected_output_and_errors(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        invalid_payload = self._memoir_payload(context)
        invalid_payload["opening"] = ""
        invalid_raw = json.dumps(invalid_payload)
        valid_raw = json.dumps(self._memoir_payload(context))
        fake = FakeGenerator([invalid_raw, valid_raw])
        story = StoryGenerator(
            aggregator=self.aggregator, generator=fake, storage=self.storage,
        ).generate_story_with_context(context, mode="creative", language="en", save=False)
        self.assertEqual(story.status, "repaired")
        repair_prompt = fake.calls[1][0][1]
        self.assertIn(invalid_raw, repair_prompt)
        self.assertIn("E_MEMOIR_OPENING", repair_prompt)
        self.assertIn("Repair guidance", repair_prompt)

    def test_two_creative_failures_return_error_without_fallback_or_database_write(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        invalid_payload = self._memoir_payload(context)
        invalid_payload["closing"] = ""
        invalid = json.dumps(invalid_payload)
        story = StoryGenerator(
            aggregator=self.aggregator,
            generator=FakeGenerator([invalid, invalid]),
            storage=self.storage,
        ).generate_story_with_context(context, mode="creative", language="en", save=True)
        self.assertEqual(story.status, "error")
        self.assertEqual(story.paragraphs, [])
        self.assertNotEqual(story.model, "deterministic-group-fallback")
        self.assertTrue(any("E_MEMOIR_CLOSING" in value for value in story.validation_codes))
        with self.storage.transaction(write=False) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM stories").fetchone()[0], 0)

    def test_faithful_two_failures_surface_error_and_diagnostics_when_fallback_disabled(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        invalid = json.dumps(self._faithful_groups_payload(
            context, facts=["I see a cup resting on the visible wooden table."],
        ))
        story = StoryGenerator(
            aggregator=self.aggregator,
            generator=FakeGenerator([invalid, invalid]),
            storage=self.storage,
        ).generate_story_with_context(
            context, mode="faithful", language="en", save=True,
            allow_deterministic_fallback=False,
        )
        self.assertEqual(story.status, "error")
        self.assertEqual(story.paragraphs, [])
        self.assertNotEqual(story.model, "deterministic-group-fallback")
        self.assertEqual(len(story.attempt_diagnostics), 2)
        self.assertTrue(all(diag["phase"] == "validation" for diag in story.attempt_diagnostics))
        # First-person prose is still rejected in faithful (neutral third-person contract).
        self.assertTrue(any(
            "E_FIRST_PERSON_IN_FACT" in code for diag in story.attempt_diagnostics for code in diag["codes"]
        ))
        with self.storage.transaction(write=False) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM stories").fetchone()[0], 0)

    def test_faithful_deterministic_fallback_is_preserved_by_default(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        invalid = json.dumps(self._faithful_groups_payload(
            context, facts=["I am looking at a cup on the wooden table."],
        ))
        story = StoryGenerator(
            aggregator=self.aggregator,
            generator=FakeGenerator([invalid, invalid]),
            storage=self.storage,
        ).generate_story_with_context(context, mode="faithful", language="en", save=False)
        self.assertEqual(story.status, "fallback")
        self.assertEqual(story.model, "deterministic-group-fallback")
        self.assertTrue(story.paragraphs)
        self.assertEqual(len(story.attempt_diagnostics), 2)

    def test_length_truncation_is_reported_as_distinct_diagnostic(self):
        class LengthTruncatingGenerator:
            model = "fake"
            last_done_reason = "length"

            def generate(self, *args, **kwargs):
                return '{"title": "x", "paragraphs": [{"text": "truncat'  # invalid JSON

            def is_available(self):
                return True

        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        story = StoryGenerator(
            aggregator=self.aggregator,
            generator=LengthTruncatingGenerator(),
            storage=self.storage,
        ).generate_story_with_context(
            context, mode="faithful", language="en", save=False,
            allow_deterministic_fallback=False,
        )
        self.assertEqual(story.status, "error")
        self.assertTrue(
            any("E_LENGTH_TRUNCATED" in code for diag in story.attempt_diagnostics for code in diag["codes"])
        )

    def test_successful_creative_story_is_saved_as_v6_with_continuous_content(self):
        second = self._add_photo("saved2.jpg", "b", "2026-01-01T11:00:00", "a blue door")
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id, second])
        raw = self._memoir_payload(context)
        valid = json.dumps(raw)
        story = StoryGenerator(
            aggregator=self.aggregator,
            generator=FakeGenerator([valid]),
            storage=self.storage,
        ).generate_story_with_context(context, mode="creative", language="en", save=True)
        self.assertEqual(story.status, "ok")
        with self.storage.transaction(write=False) as connection:
            row = connection.execute("SELECT prompt_version, grounding_json FROM stories").fetchone()
        self.assertEqual(row["prompt_version"], "grounded-story-v6.2-deterministic-photographer-mood")
        grounding = json.loads(row["grounding_json"])
        self.assertEqual(grounding["prompt_version"], "grounded-story-v6.2-deterministic-photographer-mood")
        self.assertEqual(grounding["creative_transitions"], story.creative_transitions)
        self.assertEqual(grounding["opening"], raw["opening"])
        self.assertEqual(grounding["closing"], raw["closing"])
        with self.storage.transaction(write=False) as connection:
            content = connection.execute("SELECT content FROM stories").fetchone()["content"]
        self.assertIn(raw["opening"], content)
        self.assertIn(raw["transitions"][0], content)
        self.assertIn(raw["closing"], content)

    def test_m0_and_m1_share_system_prompt_and_only_m1_exposes_mood(self):
        self.storage.save_photo_moods({self.photo_id: "excited"})
        m0 = self.aggregator.aggregate_by_photo_ids(
            [self.photo_id], use_photographer_mood=False,
        )
        m1 = self.aggregator.aggregate_by_photo_ids(
            [self.photo_id], use_photographer_mood=True,
        )
        m0_system, m0_user = PhotographerMoodPromptBuilder.build_story_prompt(m0, language="zh")
        m1_system, m1_user = PhotographerMoodPromptBuilder.build_story_prompt(m1, language="zh")
        self.assertEqual(m0_system, m1_system)
        self.assertNotIn('"mood_label": "excited"', m0_user)
        self.assertIn('"mood_label": "excited"', m1_user)
        self.assertFalse(m0.use_photographer_mood)
        self.assertEqual(m1.photographer_moods[0]["subject_role"], "photographer")
        m0_schema = PhotographerMoodPromptBuilder.output_schema(m0, language="zh")
        m1_schema = PhotographerMoodPromptBuilder.output_schema(m1, language="zh")
        self.assertEqual(m0_schema, m1_schema)
        self.assertNotIn("mood_reflections", m0_schema["properties"])

    def test_mood_validator_rejects_m0_claim_and_accepts_confirmed_m1_claim(self):
        self.storage.save_photo_moods({self.photo_id: "excited"})
        m0 = self.aggregator.aggregate_by_photo_ids([self.photo_id], use_photographer_mood=False)
        m1 = self.aggregator.aggregate_by_photo_ids([self.photo_id], use_photographer_mood=True)
        m1_candidate = self._memoir_payload(m1, language="zh")
        m0_candidate = self._memoir_payload(m0, language="zh")
        m0_candidate["opening"] = (
            "我重新翻看这组照片时，也想起按下快门时自己感到兴奋，随后把注意力放回画面里清楚可见的细节。"
        )
        normalized_m1 = PhotographerMoodPromptBuilder.normalise_payload(m1_candidate, m1)
        normalized_m0 = PhotographerMoodPromptBuilder.normalise_payload(m0_candidate, m0)
        normalized_m1["mood_reflections"] = StoryGenerator._deterministic_mood_reflections(m1, "zh")
        m1_errors = PhotographerMoodValidator.validate(normalized_m1, m1, "zh")
        m0_errors = PhotographerMoodValidator.validate(normalized_m0, m0, "zh")
        self.assertFalse(m1_errors, m1_errors)
        self.assertTrue(any("E_UNSUPPORTED_MOOD" in error for error in m0_errors))

    def test_mood_validator_rejects_wrong_label_person_attribution_and_unused_mood(self):
        self.storage.save_photo_moods({self.photo_id: "calm"})
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id], use_photographer_mood=True)
        wrong = self._memoir_payload(context, language="zh")
        wrong["opening"] = "我重新翻看照片时感到开心，并继续留意画面中清楚可见的细节和物体。"
        wrong_normalized = PhotographerMoodPromptBuilder.normalise_payload(wrong, context)
        wrong_errors = PhotographerMoodValidator.validate(wrong_normalized, context, "zh")
        self.assertTrue(any("E_UNSUPPORTED_MOOD" in error for error in wrong_errors))

        person = self._memoir_payload(context, language="zh")
        person["opening"] = "我看到照片中的人物感到平静，于是继续留意画面中清楚可见的细节和物体。"
        person_normalized = PhotographerMoodPromptBuilder.normalise_payload(person, context)
        person_errors = PhotographerMoodValidator.validate(person_normalized, context, "zh")
        self.assertTrue(any("E_MOOD_SUBJECT" in error for error in person_errors))

        unused = PhotographerMoodPromptBuilder.normalise_payload(
            self._memoir_payload(context, language="zh"), context,
        )
        unused_errors = PhotographerMoodValidator.validate(unused, context, "zh")
        self.assertTrue(any("E_MOOD_UNUSED" in error for error in unused_errors))

    def test_photographer_mood_woven_into_narrative_and_saved(self):
        # Under the mood-woven behaviour (A), the LLM reflects the verified
        # photographer mood directly in the narrative; there is no separate
        # deterministic mood_reflections appendix.
        self.storage.save_photo_moods({self.photo_id: "excited"})
        context = self.aggregator.aggregate_by_photo_ids(
            [self.photo_id], use_photographer_mood=True,
        )
        candidate = self._memoir_payload(context, language="zh")
        candidate["opening"] = (
            "我重新翻看这组照片时，按下快门时我感到兴奋，先把注意力放在画面中清楚可见的细节上，"
            "让这些片段按照原有顺序自然展开。"
        )
        story = StoryGenerator(
            aggregator=self.aggregator,
            generator=FakeGenerator([json.dumps(candidate, ensure_ascii=False)]),
            storage=self.storage,
        ).generate_story_with_context(
            context, mode="creative", language="zh", save=True, seed=20260707,
        )
        self.assertEqual(story.status, "ok", story.validation_codes)
        self.assertIn("我感到兴奋", story.content)
        # No deterministic mood appendix under the woven-mood behaviour.
        self.assertEqual(story.mood_reflections, [])

    def test_structured_mood_reflection_rejects_mismatched_pair(self):
        self.storage.save_photo_moods({self.photo_id: "tense"})
        context = self.aggregator.aggregate_by_photo_ids(
            [self.photo_id], use_photographer_mood=True,
        )
        candidate = self._memoir_payload(context, language="zh")
        candidate["mood_reflections"] = [{
            "evidence_id": "P001", "mood_label": "happy",
            "text": "按下快门记录这幅画面时，我感到开心。",
        }]
        normalized = PhotographerMoodPromptBuilder.normalise_payload(candidate, context)
        normalized["mood_reflections"] = candidate["mood_reflections"]
        errors = PhotographerMoodValidator.validate(normalized, context, "zh")
        self.assertTrue(any("E_MOOD_REFLECTION_LABEL" in error for error in errors))

    def test_faithful_prompt_never_contains_mood_metadata(self):
        self.storage.save_photo_moods({self.photo_id: "sad"})
        context = self.aggregator.aggregate_by_photo_ids(
            [self.photo_id], use_photographer_mood=True,
        )
        system, user = PromptBuilder.build_story_prompt(context, mode="faithful", language="en")
        self.assertNotIn("photographer_mood", system)
        self.assertNotIn("photographer_mood", user)
        self.assertNotIn('"sad"', user)

    def test_failed_generation_uses_grouped_fallback(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        invalid = json.dumps(self._faithful_groups_payload(
            context, facts=["My daughter was likely happy beside the visible table."], title="x",
        ))
        generator = StoryGenerator(
            aggregator=self.aggregator, generator=FakeGenerator([invalid, invalid]), storage=self.storage,
        )
        story = generator.generate_story_with_context(context, language="en", save=False)
        self.assertEqual(story.status, "fallback")
        self.assertEqual(story.paragraphs[0].evidence_ids, ("P001",))
        self.assertNotIn("daughter", story.content.lower())

    def test_valid_structured_story_keeps_only_selected_photo(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        valid = json.dumps(self._faithful_groups_payload(
            context, facts=["A cup rests on a wooden table."],
        ))
        generator = StoryGenerator(
            aggregator=self.aggregator, generator=FakeGenerator([valid]), storage=self.storage,
        )
        story = generator.generate_story_with_context(context, language="en", save=False)
        self.assertEqual(story.status, "ok", story.validation_codes)
        self.assertEqual(story.photo_ids, [self.photo_id])
        self.assertEqual(story.citations["1"], ["P001"])
        self.assertEqual(story.paragraphs[0].group_id, "G001")

    def test_faithful_grouped_contract_injects_citations_and_accepts(self):
        second = self._add_photo("two.jpg", "b", "2026-01-01T11:00:00", "a blue door")
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id, second])
        raw = self._faithful_groups_payload(
            context, language="en",
            facts=[
                "A cup rests on the wooden table near the bright window.",
                "A blue door stands against a plain wall in the following frame.",
            ],
        )
        story = StoryGenerator(
            aggregator=self.aggregator, generator=FakeGenerator([json.dumps(raw)]), storage=self.storage,
        ).generate_story_with_context(context, mode="faithful", language="en", save=False)
        self.assertEqual(story.status, "ok", story.validation_codes)
        # The model wrote no evidence_ids; the application injected each group's
        # full evidence list deterministically, so the bookkeeping can never fail.
        self.assertEqual(story.citations["1"], ["P001"])
        self.assertEqual(story.citations["2"], ["P002"])
        self.assertEqual(story.creative_transitions, [])
        self.assertEqual(story.prompt_version, "grounded-story-v3-event")
        self.assertNotIn("evidence_ids", json.dumps(raw))

    def test_faithful_merges_many_groups_to_max_paragraphs(self):
        ids = [self.photo_id]
        for index in range(2, 7):
            ids.append(self._add_photo(
                f"faithful-weave{index}.jpg", format(index, "x")[-1], f"2026-01-04T{index:02d}:00:00",
                f"a distinct visible scene number {index}",
            ))
        context = self.aggregator.aggregate_by_photo_ids(ids)
        self.assertGreater(len(context.groups), STORY_NARRATIVE_MAX_PARAGRAPHS)
        reduced_groups = StoryGenerator._reduce_groups(context.groups, STORY_NARRATIVE_MAX_PARAGRAPHS)
        reduced_context = replace(context, groups=reduced_groups)
        raw = self._faithful_groups_payload(
            reduced_context, language="en",
            facts=[
                "A quiet cup rests on the wooden table, its rim and the soft grain catching the light.",
                "A flight of stairs rises past a bare wall with even light on each tread.",
                "A wide window opens toward distant rooftops beyond the plain frame.",
                "A green tree fills the final view with layered branches and a soft background.",
            ],
        )
        story = StoryGenerator(
            aggregator=self.aggregator, generator=FakeGenerator([json.dumps(raw)]), storage=self.storage,
        ).generate_story_with_context(context, mode="faithful", language="en", save=False)
        self.assertEqual(story.status, "ok", story.validation_codes)
        self.assertLessEqual(len(story.paragraphs), STORY_NARRATIVE_MAX_PARAGRAPHS)
        self.assertEqual(len(story.paragraphs), len(reduced_groups))

    def test_repeated_paragraph_unknown_reference_and_group_omission_trigger_repair(self):
        ids = [self.photo_id]
        for index in range(2, 4):
            ids.append(self._add_photo(
                f"photo{index}.jpg", chr(96 + index), f"2026-01-01T{9 + index:02d}:00:00",
                f"distinct scene number {index}",
            ))
        context = self.aggregator.aggregate_by_photo_ids(ids)
        payload = self._payload(context, ["The same observable scene."] * len(context.groups))
        payload["paragraphs"][0]["evidence_ids"] = ["P999"]
        payload["paragraphs"] = payload["paragraphs"][:-1]
        errors = GroundingValidator.validate(payload, context, "faithful", "en")
        self.assertTrue(any("E_GROUP_PARAGRAPH_COUNT" in error for error in errors))
        self.assertTrue(any("E_CITATION_UNKNOWN" in error for error in errors))
        self.assertTrue(any("E_GROUP_COVERAGE" in error for error in errors))

    def test_fourteen_photos_never_become_fourteen_fallback_paragraphs(self):
        ids = [self.photo_id]
        for index in range(2, 15):
            ids.append(self._add_photo(
                f"many{index}.jpg", format(index, "x")[-1], f"2026-01-02T{index:02d}:00:00",
                f"unique observable scene {index}",
            ))
        context = self.aggregator.aggregate_by_photo_ids(ids)
        self.assertLessEqual(len(context.groups), 5)
        invalid = json.dumps({"title": "x", "groups": []})
        generator = StoryGenerator(
            aggregator=self.aggregator, generator=FakeGenerator([invalid, invalid]), storage=self.storage,
        )
        story = generator.generate_story_with_context(context, language="en", save=False)
        # Faithful now also merges heterogeneous groups down to at most
        # STORY_NARRATIVE_MAX_PARAGRAPHS before generation or fallback.
        self.assertLessEqual(len(story.paragraphs), STORY_NARRATIVE_MAX_PARAGRAPHS)
        self.assertGreaterEqual(len(story.paragraphs), 1)

    def test_consecutive_laptop_photos_are_one_group_and_near_duplicates_compress(self):
        second = self._add_photo("laptop2.jpg", "b", "2026-01-01T09:03:00", "a laptop on a desk")
        third = self._add_photo("laptop3.jpg", "c", "2026-01-01T09:05:00", "a laptop on a desk")
        embeddings = {
            self.photo_id: [1.0, 0.0], second: [1.0, 0.01], third: [1.0, 0.02],
        }
        aggregator = self._aggregator(embeddings)
        # Give the first photo a matching observation for this grouping fixture.
        self.storage.upsert_caption(self.photo_id, "a laptop on a desk", model_version="fixture")
        context = aggregator.aggregate_by_photo_ids([self.photo_id, second, third])
        self.assertEqual(len(context.groups), 1)
        self.assertEqual(context.groups[0].near_duplicate_evidence_ids, ("P002", "P003"))
        self.assertEqual(context.groups[0].evidence_ids, ("P001", "P002", "P003"))

    def test_dish_rack_cat_cage_conflict_never_enters_factual_fallback(self):
        self.storage.upsert_caption(self.photo_id, "a person holding a dish rack in a kitchen", model_version="conflict")
        self.storage.upsert_scene_graph(
            self.photo_id, scene_graph_text="person stand cage ; cage contain cat",
            triples=[("person", "stand by", "cage"), ("cage", "contain", "cat")], model_name="qwen",
        )
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        self.assertTrue(context.groups[0].conflicts)
        invalid = json.dumps(self._faithful_groups_payload(
            context, facts=["A person holds a dish rack beside a cat cage."],
        ))
        generator = StoryGenerator(
            aggregator=self.aggregator, generator=FakeGenerator([invalid, invalid]), storage=self.storage,
        )
        story = generator.generate_story_with_context(context, language="en", save=False)
        self.assertEqual(story.status, "fallback")
        self.assertNotIn("rack", story.content.lower())
        self.assertNotIn("cage", story.content.lower())
        self.assertTrue(story.uncertain_observations)

    def test_fallback_is_saved_with_group_grounding(self):
        invalid = json.dumps({"title": "x", "paragraphs": []})
        generator = StoryGenerator(
            aggregator=self.aggregator, generator=FakeGenerator([invalid, invalid]), storage=self.storage,
        )
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        story = generator.generate_story_with_context(context, language="en", save=True)
        self.assertEqual(story.status, "fallback")
        self.assertIsNotNone(story.story_id)
        with self.storage.transaction(write=False) as connection:
            row = connection.execute("SELECT grounding_json, prompt_version FROM stories").fetchone()
        grounding = json.loads(row["grounding_json"])
        self.assertEqual(row["prompt_version"], "grounded-story-v3-event")
        self.assertEqual(grounding["evidence_groups"][0]["group_id"], "G001")

    def test_faithful_repair_includes_rejected_output_errors_and_schema_contract(self):
        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        rejected = json.dumps({
            "title": "Wrong legacy shape",
            "paragraphs": [{"text": "Legacy paragraph field"}],
        })
        accepted = json.dumps(self._faithful_groups_payload(context))
        fake = FakeGenerator([rejected, accepted])
        generator = StoryGenerator(
            aggregator=self.aggregator, generator=fake, storage=self.storage,
        )

        story = generator.generate_story_with_context(
            context, language="en", save=False, allow_deterministic_fallback=False,
        )

        self.assertEqual(story.status, "repaired")
        second_user_prompt = fake.calls[1][0][1]
        self.assertIn("<rejected_output>", second_user_prompt)
        self.assertIn("Wrong legacy shape", second_user_prompt)
        self.assertIn("E_PARAGRAPH_TEXT", second_user_prompt)
        self.assertIn("groups[{group_id, factual_text}]", second_user_prompt)
        self.assertEqual(
            fake.calls[1][1]["output_schema"],
            PromptBuilder.output_schema(context, language="en"),
        )

    def test_dashscope_prompt_contains_dynamic_story_schema(self):
        captured: dict[str, object] = {}

        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                message = types.SimpleNamespace(content='{"title":"ok","groups":[]}')
                choice = types.SimpleNamespace(message=message, finish_reason="stop")
                return types.SimpleNamespace(choices=[choice])

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        context = self.aggregator.aggregate_by_photo_ids([self.photo_id])
        schema = PromptBuilder.output_schema(context, language="en")
        fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        generator = DashScopeGenerator(
            model="qwen-test",
            base_url="https://workspace.example/compatible-mode/v1",
            api_key="test-only-key",
        )

        with patch.dict(sys.modules, {"openai": fake_module}):
            output = generator.generate(
                "Return JSON.", "Create the story.", output_schema=schema,
                temperature=0.25, num_predict=321,
            )

        self.assertEqual(output, '{"title":"ok","groups":[]}')
        system_content = captured["messages"][0]["content"]
        self.assertIn("REMOTE STRUCTURED OUTPUT CONTRACT", system_content)
        self.assertIn("<json_schema>", system_content)
        self.assertIn('"factual_text"', system_content)
        self.assertIn('"G001"', system_content)
        self.assertIn("do not substitute legacy fields", system_content)
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(captured["max_tokens"], 321)

    def test_single_event_date_and_basket_contexts_remain_available(self):
        second = self._add_photo("event.jpg", "b", "2026-01-01T09:04:00", "a cup on a table")
        event_id = self.storage.upsert_event(
            title="Morning", start_at="2026-01-01T09:00:00", end_at="2026-01-01T09:04:00",
            date_local="2026-01-01", method="test",
        )
        self.storage.replace_event_photos(event_id, [self.photo_id, second])
        self.assertEqual(self.aggregator.aggregate_by_photo_ids([self.photo_id]).photo_count, 1)
        self.assertEqual(self.aggregator.aggregate_by_event(event_id).photo_count, 2)
        date_context = self.aggregator.aggregate_by_date("2026-01-01")
        self.assertEqual(date_context.source_kind, "date")
        self.assertTrue(any(group.source_event_id == event_id for group in date_context.groups))
        self.assertEqual(
            self.aggregator.aggregate_by_photo_ids([second, self.photo_id], source_kind="basket").photo_count, 2,
        )


if __name__ == "__main__":
    unittest.main()
