import numpy as np
from itertools import chain
from contextlib import contextmanager
from torch import nn, inference_mode


class Module(nn.Module):
    """Module"""

    def _iterate(self, fn, memo=None):
        if memo is None:
            memo = set()

        for module in filter(lambda x: isinstance(x, Module), self.children()):

            if module in memo:
                continue
            else:
                memo.add(module)

            yield from module._iterate(fn, memo)

        vals = fn(self)
        if vals is None:
            return
        else:
            yield from vals

    def _reset(self):
        return

    def _restart(self):
        return

    def reset(self):
        def fn(module):
            module._reset()

        all(self._iterate(fn))
        return self

    def restart(self):
        def fn(module):
            module._restart()

        all(self._iterate(fn))
        return self

    def dropout(self, p: float = 0):
        from .elements import Dropout

        def fn(module):
            if isinstance(module, Dropout):
                module.p = p

        all(self._iterate(fn))
        return self

    def freeze(self, mode: bool = True):
        def fn(module):
            module._frozen = bool(mode)
            module.requires_grad_(not mode)
            module.train(not mode)

        all(self._iterate(fn))
        return self

    def requires_grad_(self, requires_grad: bool = True):
        return super().requires_grad_(requires_grad and not self.frozen)

    def train(self, mode: bool = True):
        return super().train(mode and not self.frozen)

    @property
    def frozen(self):
        return getattr(self, "_frozen", False)

    @property
    def device(self):
        x = next(chain(self.parameters(), self.buffers(), [None]))
        if x is not None:
            return x.device

    @contextmanager
    def train_context(self, mode: bool = True):
        prev = self.training
        try:
            self.train(mode)
            if mode:
                yield
            else:
                with inference_mode():
                    yield
        finally:
            self.train(prev)

    def parallel_groups(self, group_size=1, shared=None):
        """
        Parameters
        ----------
        group_size : int
            parallel group size
        shared : Sequence[str] | None
            sequence of module names

        Yields
        ------
        fnn.train.parallel.ParameterGroup
        """
        from torch.distributed import is_initialized, get_world_size, get_rank, new_group
        from fnn.train.parallel import ParameterGroup

        if not is_initialized():
            assert group_size == 1
            return

        size = get_world_size()
        assert size % group_size == 0

        if not size > 1:
            return

        shared = set() if shared is None else set(shared)
        children = {k for k, _ in self.named_children()}

        if shared - children:
            raise ValueError("Shared module not recognized")

        shared_params = dict()
        group_params = dict()

        for name in children:

            module = getattr(self, name)

            if module.frozen:
                continue

            elif name in shared:
                shared_params.update({f"{name}.{k}": v for k, v in module.named_parameters()})

            elif group_size > 1:
                group_params.update({f"{name}.{k}": v for k, v in module.named_parameters()})

        if shared_params:
            ranks = np.arange(size)
            yield ParameterGroup(parameters=shared_params, group=new_group(ranks))

        if group_params:
            ranks = np.arange(group_size) + get_rank() // group_size * group_size
            yield ParameterGroup(parameters=group_params, group=new_group(ranks))


class Sequential(nn.Sequential, Module):
    """Sequential Module"""

    pass


class ModuleList(nn.ModuleList, Module):
    """Module List"""

    pass
