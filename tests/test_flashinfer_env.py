from pathlib import Path

from l20_stack import flashinfer_env


def test_find_cuda13_root_prefers_l20_env_override(monkeypatch, tmp_path):
    root = tmp_path / "cuda13"
    nvcc = root / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n")
    nvcc.chmod(0o755)

    monkeypatch.setenv("L20_FLASHINFER_CUDA_HOME", str(root))
    monkeypatch.setattr(flashinfer_env, "_python_site_roots", lambda: [])

    assert flashinfer_env.find_cuda13_root() == root.resolve()


def test_configure_flashinfer_cuda13_env_updates_build_vars(monkeypatch, tmp_path):
    root = tmp_path / "cuda13"
    nvcc = root / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n")
    nvcc.chmod(0o755)

    monkeypatch.setenv("L20_FLASHINFER_CUDA_HOME", str(root))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDACXX", raising=False)
    monkeypatch.setattr(flashinfer_env, "_python_site_roots", lambda: [])

    env = flashinfer_env.configure_flashinfer_cuda13_env()

    assert env is not None
    assert Path(env.cuda_home) == root.resolve()
    assert Path(env.nvcc) == nvcc.resolve()
    assert env.changed
    assert flashinfer_env.os.environ["CUDA_HOME"] == str(root.resolve())
    assert flashinfer_env.os.environ["CUDACXX"] == str(nvcc.resolve())
    assert flashinfer_env.os.environ["PATH"].split(":")[0] == str(root.resolve() / "bin")
