"""Tests for the affect engine."""

import pytest
from engram.affect.engine import (
    AffectEngine, Temperament, Mood, Emotion,
    PLUTCHIK_PRIMARIES, PLUTCHIK_COMPOUNDS,
)


class TestTemperament:
    def test_defaults(self):
        t = Temperament()
        assert t.novelty_seeking == 0.5
        assert t.harm_avoidance == 0.5

    def test_preset(self):
        t = Temperament.preset("bold_prototyper")
        assert t.novelty_seeking == 0.9
        assert t.harm_avoidance == 0.2

    def test_unknown_preset_returns_neutral(self):
        t = Temperament.preset("nonexistent")
        assert t.novelty_seeking == 0.5

    def test_mutate_clamped(self):
        t = Temperament(novelty_seeking=0.99)
        t.mutate("novelty_seeking", 0.1, rate=0.005)  # huge delta, clamped to 0.005
        assert t.novelty_seeking == pytest.approx(0.995)

    def test_mutate_floor(self):
        t = Temperament(harm_avoidance=0.01)
        t.mutate("harm_avoidance", -0.1, rate=0.005)
        assert t.harm_avoidance == pytest.approx(0.005)

    def test_round_trip(self):
        t = Temperament(novelty_seeking=0.7, persistence=0.3)
        d = t.to_dict()
        t2 = Temperament.from_dict(d)
        assert t2.novelty_seeking == 0.7
        assert t2.persistence == 0.3

    def test_baseline_valence(self):
        t = Temperament(reward_dependence=0.8, harm_avoidance=0.2)
        # (0.8 - 0.2) * 0.3 = 0.18
        assert t.baseline_valence == pytest.approx(0.18)


class TestMood:
    def test_default_neutral(self):
        m = Mood()
        assert m.label == "neutral"

    def test_labels(self):
        assert Mood(valence=0.5, arousal=0.5).label == "excited"
        assert Mood(valence=0.3, arousal=0.3).label == "energized"
        assert Mood(valence=0.5, arousal=0.0).label == "content"
        assert Mood(valence=0.15, arousal=0.1).label == "pleasant"
        assert Mood(valence=-0.5, arousal=0.5).label == "angry"
        assert Mood(valence=-0.3, arousal=0.3).label == "frustrated"
        assert Mood(valence=-0.5, arousal=-0.5).label == "depleted"
        assert Mood(valence=-0.3, arousal=0.1).label == "sad"
        assert Mood(valence=-0.15, arousal=0.0).label == "uneasy"
        assert Mood(valence=0.0, arousal=0.5).label == "alert"
        assert Mood(valence=0.0, arousal=0.3).label == "engaged"
        assert Mood(valence=0.0, arousal=0.0).label == "neutral"

    def test_update_from_joy(self):
        m = Mood()
        m.update_from_emotion(Emotion(primary="joy", intensity=1.0))
        assert m.valence > 0  # joy increases valence

    def test_update_from_anger(self):
        m = Mood()
        m.update_from_emotion(Emotion(primary="anger", intensity=1.0))
        assert m.valence < 0  # anger decreases valence
        assert m.arousal > 0  # anger increases arousal

    def test_clamped(self):
        m = Mood()
        for _ in range(100):
            m.update_from_emotion(Emotion(primary="joy", intensity=1.0))
        assert m.valence <= 1.0
        assert m.arousal <= 1.0

    def test_decay_toward_baseline(self):
        m = Mood(valence=0.8, arousal=0.6)
        m.decay_toward_baseline(0.0, 0.0, hours=10.0, rate=0.1)
        assert m.valence < 0.8  # decayed toward 0
        assert m.arousal < 0.6


class TestEmotion:
    def test_decay(self):
        e = Emotion(primary="joy", intensity=1.0)
        e.decay(hours=2.0, rate=0.3)
        assert e.intensity < 1.0
        assert e.intensity > 0

    def test_rapid_decay(self):
        e = Emotion(primary="anger", intensity=0.5)
        e.decay(hours=24.0, rate=0.3)
        assert e.intensity < 0.01  # nearly gone after 24h


class TestAffectEngine:
    def test_trigger_emotion(self):
        engine = AffectEngine()
        emotion = engine.trigger_emotion("joy", 0.7, trigger="task completed")
        assert emotion.primary == "joy"
        assert emotion.intensity == 0.7
        assert len(engine.active_emotions) == 1

    def test_mood_shifts_on_emotion(self):
        engine = AffectEngine()
        initial_valence = engine.mood.valence
        engine.trigger_emotion("joy", 1.0)
        assert engine.mood.valence > initial_valence

    def test_get_current_state(self):
        engine = AffectEngine()
        engine.trigger_emotion("surprise", 0.5)
        state = engine.get_current_state()
        assert "mood_valence" in state
        assert "mood_label" in state
        assert "active_emotions" in state
        assert "temperament" in state

    def test_consolidation_feedback_positive(self):
        engine = AffectEngine()
        initial_rd = engine.temperament.reward_dependence
        engine.consolidation_feedback(
            positive_outcomes=10, negative_outcomes=2,
            novel_discoveries=5, total_events=20,
        )
        # Should slightly increase reward dependence
        assert engine.temperament.reward_dependence >= initial_rd

    def test_consolidation_feedback_negative(self):
        engine = AffectEngine()
        initial_ha = engine.temperament.harm_avoidance
        engine.consolidation_feedback(
            positive_outcomes=2, negative_outcomes=10,
            novel_discoveries=0, total_events=20,
        )
        assert engine.temperament.harm_avoidance >= initial_ha

    def test_custom_temperament(self):
        t = Temperament.preset("careful_reviewer")
        engine = AffectEngine(temperament=t)
        assert engine.temperament.harm_avoidance == 0.9
        assert engine.mood.valence == pytest.approx(t.baseline_valence, abs=0.01)


class TestPlutchikData:
    def test_all_primaries_have_opposites(self):
        for primary, data in PLUTCHIK_PRIMARIES.items():
            assert "opposite" in data
            assert data["opposite"] in PLUTCHIK_PRIMARIES

    def test_compounds_are_adjacent_primaries(self):
        primaries = list(PLUTCHIK_PRIMARIES.keys())
        for (a, b), compound in PLUTCHIK_COMPOUNDS.items():
            assert a in primaries
            assert b in primaries
            assert isinstance(compound, str)
