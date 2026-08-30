import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest


@pytest.fixture(scope="session")
def tz_rules():
    from tz_alpr.country_rules.tanzania import build

    return build()


@pytest.fixture(scope="session")
def decoder(tz_rules):
    from tz_alpr.postprocessing.tz_aware import TanzaniaAwareDecoder

    return TanzaniaAwareDecoder(tz_rules)


@pytest.fixture(scope="session")
def confidence_model():
    from tz_alpr.postprocessing.confidence import build_confidence_model

    return build_confidence_model()
