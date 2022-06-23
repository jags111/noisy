from pathlib import Path
from pytest import fixture


@fixture
def workdir(tmpdir: Path) -> Path:
    return Path(str(tmpdir)) / 'testing_wd'
