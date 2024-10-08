import torch


# -------------- Optimizer Base --------------


class Optimizer:
    """Module Optimizer"""

    def _init(self, scheduler):
        """
        Parameters
        ----------
        scheduler : fnn.train.schedulers.Scheduler
            hyperparameter scheduler
        """
        self.scheduler = scheduler

    @property
    def hyperparameters(self):
        """
        Returns
        -------
        dict
            dictionary of hyperparameters
        """
        raise NotImplementedError()

    def step(self, parameters, **kwargs):
        """Perform a gradient descent step

        Parameters
        ----------
        parameters : Mapping[str, fnn.model.parameters.Parameter]
            mapping of parameters
        **kwargs
            hyperparameters
        """
        raise NotImplementedError()

    def optimize(self, loader, objective, parameters, groups=None):
        """
        Parameters
        ----------
        loader : fnn.train.loaders.Loader
            data loader
        objective : fnn.train.objectives.Objective
            training objective
        parameters : Mapping[str, fnn.model.parameters.Parameter]
            mapping of parameters
        groups : None | List[fnn.train.parallel.ParameterGroup]
            none or list of parameter groups

        Yields
        ------
        int
            epoch number
        dict
            optimization info (hyperparameters and objectives)
        """
        raise NotImplementedError()


# -------------- Optimizer Types --------------


class RandomOptimizer(Optimizer):
    """Random Optimizer"""

    def __init__(self, seed=42):
        """
        Parameters
        ----------
        seed : int
            random seed for optimization
        """
        self.seed = int(seed)

    def optimize(self, loader, objective, parameters, groups=None):
        """
        Parameters
        ----------
        loader : fnn.train.loaders.Loader
            data loader
        objective : fnn.train.objectives.Objective
            training objective
        parameters : Mapping[str, fnn.model.parameters.Parameter]
            mapping of parameters
        groups : None | Iterable[fnn.train.parallel.ParameterGroup]
            None | parameter groups

        Yields
        ------
        int
            epoch number
        dict
            optimization info (seed, hyperparameters, and objectives)
        """
        parameters = dict(parameters)
        groups = [] if groups is None else list(groups)
        devices = list(range(torch.cuda.device_count()))

        while self.scheduler.step():

            epoch = self.scheduler.epoch
            seed = self.scheduler.seed + self.seed
            hyperparameters = self.scheduler(**self.hyperparameters)

            for g in groups:
                g.sync_params()

            for training in [True, False]:

                with torch.random.fork_rng(devices):
                    torch.manual_seed(seed)

                    for data in loader(training=training):

                        objective(training=training, **data)

                        if training:

                            for g in groups:
                                g.sync_grads()

                            self.step(parameters, **hyperparameters)

            objectives = objective.step()

            yield epoch, dict(seed=seed, **hyperparameters, **objectives)


class SgdClip(RandomOptimizer):
    """Stochastic Gradient Descent with Adaptive Gradient Clipping"""

    def __init__(self, lr=0.1, decay=0, momentum=0, nesterov=False, clip=float("inf"), eps=0.001, seed=42):
        """
        Parameters
        ----------
        lr : float
            learning rate
        decay : float
            weight decay
        momentum : float
            momentum factor [0, 1)
        nesterov : bool
            enables nesterov momentum
        clip : float
            adaptive gradient clipping factor
        eps : float
            adaptive gradient clipping minimum
        seed : int
            random seed
        """
        assert lr > 0
        assert decay >= 0
        assert 0 <= momentum < 1
        assert clip > 0
        assert eps > 0

        super().__init__(
            seed=seed,
        )
        self._hyperparameters = dict(
            lr=float(lr),
            decay=float(decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            clip=float(clip),
            eps=float(eps),
        )
        self.momentums = dict()

    @property
    def hyperparameters(self):
        """
        Returns
        -------
        dict
            sgd clip hyperparameters
        """
        return dict(self._hyperparameters)

    @torch.no_grad()
    def step(self, parameters, lr, momentum, nesterov, decay, clip, eps):
        """Perform a gradient descent step

        Parameters
        ----------
        parameters : Mapping[str, fnn.model.parameters.Parameter]
            mapping of parameters
        lr : float
            learning rate
        decay : float
            weight decay
        momentum : float
            momentum factor [0, 1)
        nesterov : bool
            enables nesterov momentum
        clip : float
            adaptive gradient clipping factor
        eps : float
            adaptive gradient clipping minimum
        """
        for k, p in parameters.items():

            d_p = p.grad

            if d_p is None:
                continue

            d_p = d_p.mul(p.scale)

            if clip < float("inf"):
                p_norm = p.norm(p=2, dim=p.norm_dim, keepdim=True)
                d_p_norm = d_p.norm(p=2, dim=p.norm_dim, keepdim=True)

                min_norm = eps * p.numel() ** 0.5
                max_norm = (clip * p_norm).clamp(min=min_norm)

                c = max_norm / torch.maximum(d_p_norm, max_norm)
                d_p = d_p.mul(c)

            if momentum > 0:
                m = self.momentums.get(k, None)

                if m is None:
                    m = self.momentums[k] = torch.clone(d_p)
                else:
                    m.mul_(momentum).add_(d_p)

                if nesterov:
                    d_p = d_p.add(m, alpha=momentum)
                else:
                    d_p = m

            if decay > 0 and p.decay:
                d_p = d_p.add(p, alpha=decay)

            p.add_(d_p, alpha=-lr)
            p.grad = None
