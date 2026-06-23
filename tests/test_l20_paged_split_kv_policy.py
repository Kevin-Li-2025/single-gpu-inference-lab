import ast
from pathlib import Path


def _load_policy():
    source = Path("integrations/vllm/l20_paged_split_kv.py").read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "should_use_l20_paged_split_kv"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, "<policy>", "exec"), namespace)
    return namespace["should_use_l20_paged_split_kv"]


def test_paged_split_kv_stays_disabled_until_it_beats_flashinfer():
    policy = _load_policy()
    assert not policy(1, 2048)
    assert not policy(8, 4096)


def test_paged_fp8_split_kv_stays_disabled_after_serving_regression():
    source = Path("integrations/vllm/l20_paged_split_kv.py").read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "should_use_l20_paged_fp8_split_kv"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, "<policy>", "exec"), namespace)
    policy = namespace["should_use_l20_paged_fp8_split_kv"]

    assert not policy(1, 4096)
    assert not policy(4, 4096)
    assert not policy(8, 2048)
    assert not policy(8, 4096)
