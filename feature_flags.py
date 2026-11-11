"""Features are resolved from trusted worker targets."""

from dataclasses import dataclass
import config


KNOWN_FEATURES = frozenset({"bulk_extraction", "batch_completion_email"})
ENVIRONMENTS = frozenset({"staging", "production", "development"})


@dataclass(frozen=True)
class FeatureContext:
    environment: str
    features: frozenset[str]

    def enabled(self, name: str) -> bool:
        if name not in KNOWN_FEATURES:
            raise ValueError(f"Unknown feature: {name}")
        return name in self.features

    def require(self, name: str) -> None:
        if not self.enabled(name):
            raise PermissionError(f"Feature {name} is disabled in {self.environment}")


def features_for_environment(environment: str) -> FeatureContext:
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown worker environment: {environment}")
    raw = config.get_env(f"{environment.upper()}_FEATURES", "") or ""
    enabled = frozenset() if raw.strip() == "none" else frozenset(
        part.strip() for part in raw.split(",") if part.strip()
    )
    unknown = enabled - KNOWN_FEATURES
    if unknown:
        raise ValueError(f"Unknown features for {environment}: {', '.join(sorted(unknown))}")
    return FeatureContext(environment, enabled)
