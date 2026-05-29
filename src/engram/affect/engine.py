"""Affect engine — temperament, mood, and emotion.

Neuroscience-grounded:
- Temperament: Cloninger's 4 dimensions (slowly evolving personality)
- Mood: Russell's Circumplex (valence × arousal, drifts over hours)
- Emotion: Plutchik's Wheel (8 primaries, triggered by events)

Self-mutation loop:
  Events → Emotions → Mood update → Consolidation → Temperament drift
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# --- Plutchik's Emotion Wheel ---

PLUTCHIK_PRIMARIES = {
    "joy": {"opposite": "sadness", "intensity_low": "serenity", "intensity_high": "ecstasy"},
    "trust": {"opposite": "disgust", "intensity_low": "acceptance", "intensity_high": "admiration"},
    "fear": {"opposite": "anger", "intensity_low": "apprehension", "intensity_high": "terror"},
    "surprise": {"opposite": "anticipation", "intensity_low": "distraction", "intensity_high": "amazement"},
    "sadness": {"opposite": "joy", "intensity_low": "pensiveness", "intensity_high": "grief"},
    "disgust": {"opposite": "trust", "intensity_low": "boredom", "intensity_high": "loathing"},
    "anger": {"opposite": "fear", "intensity_low": "annoyance", "intensity_high": "rage"},
    "anticipation": {"opposite": "surprise", "intensity_low": "interest", "intensity_high": "vigilance"},
}

PLUTCHIK_COMPOUNDS = {
    ("joy", "trust"): "love",
    ("trust", "fear"): "submission",
    ("fear", "surprise"): "awe",
    ("surprise", "sadness"): "disapproval",
    ("sadness", "disgust"): "remorse",
    ("disgust", "anger"): "contempt",
    ("anger", "anticipation"): "aggressiveness",
    ("anticipation", "joy"): "optimism",
}


@dataclass
class Emotion:
    """A single emotion instance (Plutchik)."""
    primary: str  # one of 8 primaries
    intensity: float = 0.5  # 0-1
    compound: str = ""  # optional compound emotion
    trigger: str = ""  # what caused this emotion
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def decay(self, hours: float, rate: float = 0.3) -> float:
        """Emotions decay faster than memories."""
        self.intensity *= math.exp(-rate * hours)
        return self.intensity


# --- Russell's Circumplex Model ---

@dataclass
class Mood:
    """Current mood state (Russell's Circumplex: valence × arousal)."""
    valence: float = 0.0  # -1 (negative) to +1 (positive)
    arousal: float = 0.0  # -1 (calm) to +1 (excited)
    confidence: float = 1.0  # how confident the system is in current mood assessment
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def label(self) -> str:
        """Human-readable mood label based on Russell's Circumplex."""
        v, a = self.valence, self.arousal
        # More granular labels with lower thresholds
        if v > 0.4 and a > 0.4:
            return "excited"
        elif v > 0.2 and a > 0.2:
            return "energized"
        elif v > 0.2 and a <= 0.2:
            return "content"
        elif v > 0.1 and a > 0.0:
            return "pleasant"
        elif v < -0.4 and a > 0.4:
            return "angry"
        elif v < -0.2 and a > 0.2:
            return "frustrated"
        elif v < -0.2 and a <= -0.2:
            return "depleted"
        elif v < -0.2 and a <= 0.2:
            return "sad"
        elif v < -0.1:
            return "uneasy"
        elif a > 0.4:
            return "alert"
        elif a > 0.2:
            return "engaged"
        else:
            return "neutral"

    def update_from_emotion(self, emotion: Emotion) -> None:
        """Shift mood based on an emotion event."""
        # Map emotions to valence/arousal shifts
        emotion_map = {
            "joy": (0.3, 0.1),
            "trust": (0.2, -0.1),
            "fear": (-0.2, 0.3),
            "surprise": (0.0, 0.3),
            "sadness": (-0.3, -0.2),
            "disgust": (-0.2, 0.1),
            "anger": (-0.3, 0.3),
            "anticipation": (0.1, 0.2),
        }
        shift = emotion_map.get(emotion.primary, (0.0, 0.0))
        self.valence += shift[0] * emotion.intensity * 0.5  # moderate damping
        self.arousal += shift[1] * emotion.intensity * 0.5
        # Clamp
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(-1.0, min(1.0, self.arousal))
        self.last_updated = datetime.now(timezone.utc)

    def decay_toward_baseline(self, baseline_valence: float = 0.0, baseline_arousal: float = 0.0,
                               hours: float = 1.0, rate: float = 0.1) -> None:
        """Mood decays toward temperament baseline over time."""
        decay = math.exp(-rate * hours)
        self.valence = baseline_valence + (self.valence - baseline_valence) * decay
        self.arousal = baseline_arousal + (self.arousal - baseline_arousal) * decay


# --- Cloninger's Temperament Model ---

@dataclass
class Temperament:
    """Personality dimensions (Cloninger's 4 dimensions).

    These evolve very slowly through consolidation cycles.
    """
    novelty_seeking: float = 0.5  # dopamine — willingness to try new things
    harm_avoidance: float = 0.5  # serotonin — risk aversion, verification
    reward_dependence: float = 0.5  # norepinephrine — sensitivity to feedback
    persistence: float = 0.5  # how long before pivoting

    @property
    def baseline_valence(self) -> float:
        """Mood baseline from temperament."""
        return (self.reward_dependence - self.harm_avoidance) * 0.3

    @property
    def baseline_arousal(self) -> float:
        """Arousal baseline from temperament."""
        return (self.novelty_seeking - 0.5) * 0.4

    def mutate(self, dimension: str, delta: float, rate: float = 0.005) -> None:
        """Tiny temperament drift from experience. Rate limits change."""
        clamped_delta = max(-rate, min(rate, delta))
        current = getattr(self, dimension)
        new_val = max(0.0, min(1.0, current + clamped_delta))
        setattr(self, dimension, new_val)

    def to_dict(self) -> dict:
        return {
            "novelty_seeking": self.novelty_seeking,
            "harm_avoidance": self.harm_avoidance,
            "reward_dependence": self.reward_dependence,
            "persistence": self.persistence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Temperament:
        return cls(
            novelty_seeking=d.get("novelty_seeking", 0.5),
            harm_avoidance=d.get("harm_avoidance", 0.5),
            reward_dependence=d.get("reward_dependence", 0.5),
            persistence=d.get("persistence", 0.5),
        )

    @classmethod
    def preset(cls, name: str) -> Temperament:
        """Named personality presets."""
        presets = {
            "neutral": cls(),
            "careful_reviewer": cls(novelty_seeking=0.2, harm_avoidance=0.9, reward_dependence=0.5, persistence=0.9),
            "bold_prototyper": cls(novelty_seeking=0.9, harm_avoidance=0.2, reward_dependence=0.4, persistence=0.4),
            "steady_operator": cls(novelty_seeking=0.3, harm_avoidance=0.5, reward_dependence=0.5, persistence=0.7),
            "curious_researcher": cls(novelty_seeking=0.95, harm_avoidance=0.3, reward_dependence=0.7, persistence=0.9),
            "empathetic_partner": cls(novelty_seeking=0.5, harm_avoidance=0.4, reward_dependence=0.9, persistence=0.6),
        }
        return presets.get(name, cls())


# --- Affect Engine ---

class AffectEngine:
    """Manages the full affect system: temperament → mood → emotions.

    Self-mutation loop:
    1. Events trigger emotions (Plutchik)
    2. Emotions shift mood (Russell)
    3. Consolidation reads mood/emotions for memory appraisal
    4. Over many cycles, temperament drifts (Cloninger)
    5. Temperament sets mood baseline → biases future emotions
    """

    def __init__(self, temperament: Temperament | None = None):
        self.temperament = temperament or Temperament()
        self.mood = Mood(
            valence=self.temperament.baseline_valence,
            arousal=self.temperament.baseline_arousal,
        )
        self.active_emotions: list[Emotion] = []
        self._emotion_history: list[Emotion] = []

    def trigger_emotion(self, primary: str, intensity: float = 0.5,
                        trigger: str = "", compound: str = "") -> Emotion:
        """Trigger an emotion from an event."""
        if primary not in PLUTCHIK_PRIMARIES:
            logger.warning("unknown primary emotion: %s", primary)
            primary = "surprise"  # default

        emotion = Emotion(
            primary=primary,
            intensity=min(max(intensity, 0.0), 1.0),
            compound=compound,
            trigger=trigger,
        )

        self.active_emotions.append(emotion)
        self._emotion_history.append(emotion)
        self.mood.update_from_emotion(emotion)

        return emotion

    def get_current_state(self) -> dict:
        """Get current affect state for memory encoding context."""
        # Decay old emotions
        now = datetime.now(timezone.utc)
        for em in self.active_emotions:
            hours = (now - em.ts).total_seconds() / 3600
            em.decay(hours)
        self.active_emotions = [e for e in self.active_emotions if e.intensity > 0.05]

        return {
            "mood_valence": self.mood.valence,
            "mood_arousal": self.mood.arousal,
            "mood_label": self.mood.label,
            "active_emotions": [
                {"primary": e.primary, "intensity": round(e.intensity, 3), "compound": e.compound}
                for e in self.active_emotions
            ],
            "temperament": self.temperament.to_dict(),
        }

    def consolidation_feedback(self, positive_outcomes: int, negative_outcomes: int,
                                novel_discoveries: int, total_events: int) -> None:
        """Called after consolidation to drift temperament.

        This is the self-mutation: experience slowly reshapes personality.
        """
        if total_events == 0:
            return

        rate = 0.005  # tiny rate

        # More positive outcomes → slightly increase reward dependence
        if positive_outcomes > negative_outcomes:
            self.temperament.mutate("reward_dependence", rate)
        elif negative_outcomes > positive_outcomes:
            self.temperament.mutate("harm_avoidance", rate)

        # Novel discoveries → slightly increase novelty seeking
        if novel_discoveries > total_events * 0.3:
            self.temperament.mutate("novelty_seeking", rate)

        # Decay mood toward new baseline
        self.mood.decay_toward_baseline(
            self.temperament.baseline_valence,
            self.temperament.baseline_arousal,
        )
