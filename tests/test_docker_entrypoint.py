import sys

import pytest

from scripts import docker_entrypoint


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ["python", "scripts/ask.py", "--help"]),
        (["--help"], ["python", "scripts/ask.py", "--help"]),
        (["-h"], ["python", "scripts/ask.py", "-h"]),
        (["--no-generation", "question"], ["python", "scripts/ask.py", "--no-generation", "question"]),
        (["What problem does SELF-RAG address?"], ["python", "scripts/ask.py", "What problem does SELF-RAG address?"]),
        (["pytest", "-q"], ["pytest", "-q"]),
        (["python", "-c", "print('ok')"], ["python", "-c", "print('ok')"]),
        (["python", "scripts/ask.py", "question"], ["python", "scripts/ask.py", "question"]),
        (["-m", "pytest", "-q"], ["python", "-m", "pytest", "-q"]),
        (["-c", "print('ok')"], ["python", "-c", "print('ok')"]),
        (["-V"], ["python", "-V"]),
        (["--version"], ["python", "--version"]),
    ],
)
def test_dispatches_arguments(monkeypatch, args, expected):
    calls = []

    class ExecCalled(Exception):
        pass

    def fake_execvp(file, command_args):
        calls.append((file, command_args))
        raise ExecCalled

    monkeypatch.setattr(sys, "argv", ["docker_entrypoint.py", *args])
    monkeypatch.setattr(docker_entrypoint.os, "execvp", fake_execvp)

    with pytest.raises(ExecCalled):
        docker_entrypoint.main()

    assert calls == [(expected[0], expected)]
