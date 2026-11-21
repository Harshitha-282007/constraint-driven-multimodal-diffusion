"""Source-level guard (no torch / diffusers needed): controlnet_text must be a plain static method.

A stray @torch.no_grad() on top of @staticmethod once turned it into a normal method, so run_role(cn_text="mix")
failed with 'takes 3 positional arguments but 4 were given' on Colab while the CPU tests stayed skipped."""
import ast
import os

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "pipeline.py")


def _method(name):
    tree = ast.parse(open(SRC, encoding="utf8").read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Pipeline")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _decorators(fn):
    return [ast.unparse(d) for d in fn.decorator_list]


def test_controlnet_text_is_only_a_staticmethod():
    assert _decorators(_method("controlnet_text")) == ["staticmethod"]


def test_run_role_keeps_no_grad_and_takes_the_adapter_arguments():
    fn = _method("run_role")
    assert _decorators(fn) == ["torch.no_grad()"]
    args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    assert {"cn_text", "adapter", "palette", "adapter_scale"} <= set(args)
