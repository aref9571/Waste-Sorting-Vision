from typer.testing import CliRunner

from wastevision import __version__
from wastevision.cli import app

runner = CliRunner()


def test_version_command_prints_package_version():
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_unknown_command_fails():
    result = runner.invoke(app, ["does-not-exist"])

    assert result.exit_code != 0
