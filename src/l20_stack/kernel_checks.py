"""Cheap interface checks before running expensive KernelBench evaluation."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class KernelInterfaceReport:
    valid: bool
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def validate_kernelbench_interface(source: str) -> KernelInterfaceReport:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return KernelInterfaceReport(False, [f"syntax error: {error.msg}"], warnings)

    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if "ModelNew" not in class_names:
        errors.append("missing class ModelNew")

    forbidden_names = {"get_inputs", "get_init_inputs"}
    forbidden_calls = {"NotImplementedError"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            errors.append("contains pass statement")
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            errors.append(f"references evaluator helper {node.id}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
            errors.append(f"raises or constructs {node.func.id}")
    for kernel in _triton_jit_functions(tree):
        for node in ast.walk(kernel):
            if _is_tl_sum_keepdims(node):
                errors.append("Triton tl.sum uses unsupported keepdims keyword")
            if _is_block_tensor_reshape(node):
                errors.append("Triton kernel uses dynamic block tensor view/reshape")
            if _is_tl_arange_missing_end(node):
                errors.append("Triton tl.arange must use explicit start and end")
    errors.extend(_triton_launcher_arg_errors(tree))

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    wrapper = functions.get("triton_kernel_wrapper")
    model_new = classes.get("ModelNew")
    if any(_is_main_guard(node) for node in tree.body):
        errors.append("contains executable test harness")
    if model_new is not None:
        for method_name in ("__init__", "forward"):
            method = _find_method(model_new, method_name)
            if method is not None and _has_varargs(method):
                errors.append(f"ModelNew.{method_name} uses varargs")
    if wrapper is not None and model_new is not None:
        forward = next(
            (
                item
                for item in model_new.body
                if isinstance(item, ast.FunctionDef) and item.name == "forward"
            ),
            None,
        )
        if forward is not None and not _has_varargs(wrapper) and _calls_wrapper_with_varargs(forward):
            wrapper_args = _required_positional_args(wrapper, drop_self=False)
            forward_args = _required_positional_args(forward, drop_self=True)
            if len(wrapper_args) > len(forward_args):
                errors.append(
                    "triton_kernel_wrapper requires more positional args than ModelNew.forward"
                )

    return KernelInterfaceReport(not errors, sorted(set(errors)), sorted(set(warnings)))


def _required_positional_args(node: ast.FunctionDef, drop_self: bool) -> List[str]:
    args = list(node.args.posonlyargs) + list(node.args.args)
    if drop_self and args and args[0].arg == "self":
        args = args[1:]
    defaults = len(node.args.defaults)
    if defaults:
        args = args[:-defaults]
    return [arg.arg for arg in args]


def _has_varargs(node: ast.FunctionDef) -> bool:
    return node.args.vararg is not None or node.args.kwarg is not None


def _find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    if not isinstance(left, ast.Name) or left.id != "__name__":
        return False
    if not isinstance(right, ast.Constant) or right.value != "__main__":
        return False
    return isinstance(test.ops[0], ast.Eq)


def _triton_jit_functions(tree: ast.Module) -> List[ast.FunctionDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(_is_triton_jit_decorator(decorator) for decorator in node.decorator_list)
    ]


def _is_triton_jit_decorator(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "triton" and node.attr == "jit"
    if isinstance(node, ast.Call):
        return _is_triton_jit_decorator(node.func)
    return False


def _is_tl_sum_keepdims(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "sum":
        return False
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "tl":
        return False
    return any(keyword.arg == "keepdims" for keyword in node.keywords)


def _is_block_tensor_reshape(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in {"view", "reshape"}:
        return False
    return True


def _is_tl_arange_missing_end(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "arange":
        return False
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "tl":
        return False
    return len(node.args) == 1 and not any(keyword.arg == "end" for keyword in node.keywords)


def _triton_launcher_arg_errors(tree: ast.Module) -> List[str]:
    kernels = {node.name: node for node in _triton_jit_functions(tree)}
    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Subscript) or not isinstance(node.func.value, ast.Name):
            continue
        kernel = kernels.get(node.func.value.id)
        if kernel is None:
            continue
        required = _required_positional_args(kernel, drop_self=False)
        supplied = len(node.args) + sum(1 for keyword in node.keywords if keyword.arg in required)
        if supplied < len(required):
            errors.append(
                f"Triton launcher for {kernel.name} supplies {supplied} args for {len(required)} required args"
            )
    return errors


def _calls_wrapper_with_varargs(node: ast.FunctionDef) -> bool:
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        if not isinstance(item.func, ast.Name) or item.func.id != "triton_kernel_wrapper":
            continue
        has_star = any(isinstance(arg, ast.Starred) for arg in item.args)
        has_kwargs = any(keyword.arg is None for keyword in item.keywords)
        if has_star or has_kwargs:
            return True
    return False
