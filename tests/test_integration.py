from pathlib import Path
from click.testing import CliRunner

from cli import cli


def test_init(workdir: Path) -> None:
    runner = CliRunner()
    assert not workdir.exists()
    res = runner.invoke(cli, ['init',
                              '--workdir', str(workdir),
                              '--config', './configs/testing.yaml',
                              '--no-symlink'])
    assert res.exit_code == 0
    assert workdir.exists()
    assert (workdir / 'initial').exists()
    assert (workdir / '.latest').exists()
