"""The whole self-test must pass — it is the project's definition of done."""
from mmlab import self_test


def test_self_test_passes():
    assert self_test(verbose=False) is True
