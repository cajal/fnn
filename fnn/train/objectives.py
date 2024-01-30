import numpy as np
import torch


# -------------- Objective Bases --------------


class Objective:
    """Objective"""

    def __call__(self, training=True, **data):
        """Perform an objective call

        Parameters
        ----------
        training : bool
            training or validation
        **data
            training or validation data
        """
        raise NotImplementedError()

    def step(self):
        """Perform an epoch step

        Returns
        -------
        dict
            epoch info
        """
        raise NotImplementedError()


# -------------- Objective Types --------------

# -- Network Objectives --


class NetworkObjective(Objective):
    """Network Objective"""

    def _init(self, network):
        """
        Parameters
        ----------
        network : fnn.model.networks.Network
            network module
        """
        self.network = network


class NetworkLoss(NetworkObjective):
    """Network Loss"""

    def __init__(self, sample_stream=True, burnin_frames=0):
        """
        Parameters
        ----------
        sample_stream : bool
            sample stream during training
        burnin_frames : int
            number of initial frames to discard
        """
        assert burnin_frames >= 0

        self.sample_stream = bool(sample_stream)
        self.burnin_frames = int(burnin_frames)

        self.log = dict(
            training_objective=[],
            training_regularize=[],
            validation_objective=[],
            validation_regularize=[],
        )

    def __call__(self, units, stimuli, perspectives=None, modulations=None, training=True):
        """Perform an objective call

        Parameters
        ----------
        stimulus : Iterable[ND array]
            either singular or batch
        perspective : Iterable[ND array] | None
            either singular or batch
        modulations : Iterable[ND array] | None
            either singular or batch
        training : bool
            training or validation
        """
        if training and self.sample_stream:
            stream = torch.randint(0, self.network.streams, (1,)).item()
        else:
            stream = None

        losses = self.network.generate_loss(
            units=units,
            stimuli=stimuli,
            perspectives=perspectives,
            modulations=modulations,
            stream=stream,
            training=training,
        )
        losses = list(losses)[self.burnin_frames :]

        regs = self.network.regularize()
        if regs:
            regs = torch.stack(regs)
            rsum = regs.sum()
        else:
            regs = None
            rsum = torch.tensor(0)

        if training:
            objective = torch.stack(losses).mean() + rsum
            objective.backward()

            obj = "training_objective"
            reg = "training_regularize"

        else:
            objective = np.stack(losses).mean() + rsum.item()

            obj = "validation_objective"
            reg = "validation_regularize"

        objective = objective.item()

        if not np.isfinite(objective):
            raise ValueError("Non-finite objective")

        self.log[obj].append(objective)

        if regs is not None:
            self.log[reg].append(regs.tolist())

    def step(self):
        """Perform an epoch step

        Returns
        -------
        dict[str, float]
            epoch objectives
        """
        ret = dict()

        for key in ["training_objective", "validation_objective"]:
            if self.log[key]:
                ret[key] = np.mean(self.log[key])
                self.log[key].clear()

        for key in ["training_regularize", "validation_regularize"]:
            if self.log[key]:
                ret[key] = np.stack(self.log[key], 0).mean(0)
                self.log[key].clear()

        return ret


# -- Stimulus Objectives --


class StimulusObjective(Objective):
    """Stimulus Objective"""

    def _init(self, stimulus, network, unit_index=None):
        """
        Parameters
        ----------
        module : fnn.model.networks.Network
            network module
        stimulus : fnn.model.stimuli.Stimulus
            stimulus module
        unit_index : int | List[int] | None
            unit index
        """
        self.stimulus = stimulus
        self.network = network.freeze(True)

        if unit_index is None:
            self.unit_index = None
        else:
            try:
                self.unit_index = int(unit_index)
            except:
                self.unit_index = list(map(int, unit_index))

        self.log = dict(
            training_loss=[],
            training_penalty=[],
            validation_loss=[],
            validation_penalty=[],
        )

    def step(self):
        """Perform an epoch step

        Returns
        -------
        dict[str, float]
            epoch objectives
        """
        objectives = dict()

        for key, value in self.log.items():
            if value:
                objectives[key] = np.mean(value)
                value.clear()

        return objectives


class Reconstruction(StimulusObjective):
    """Reconstruction"""

    def __init__(
        self,
        trial_perspectives,
        trial_modulations,
        trial_units,
        sample_trial=True,
        sample_stream=True,
        burnin_frames=0,
        stimulus_penalty=0,
    ):
        """
        Parameters
        ----------
        trial_perspectives : 3D array
            [frames, trials, perspectives]
        trial_modulations : 3D array
            [frames, trials, modulations]
        trial_units : 3D array
            [frames, trials, units]
        sample_trial : bool
            sample trial during training
        sample_stream : bool
            sample stream during training
        burnin_frames : int
            number of initial frames to discard
        stimulus_penalty : float
            stimulus penalty weight
        """
        assert burnin_frames >= 0
        assert stimulus_penalty >= 0

        self.trial_perspectives = torch.tensor(trial_perspectives, dtype=torch.float)
        self.trial_modulations = torch.tensor(trial_modulations, dtype=torch.float)
        self.trial_units = torch.tensor(trial_units, dtype=torch.float)

        (self.frames,) = {
            self.trial_perspectives.shape[0],
            self.trial_modulations.shape[0],
            self.trial_units.shape[0],
        }
        (self.trials,) = {
            self.trial_perspectives.shape[1],
            self.trial_modulations.shape[1],
            self.trial_units.shape[1],
        }

        self.sample_trial = bool(sample_trial)
        self.sample_stream = bool(sample_stream)
        self.burnin_frames = int(burnin_frames)
        self.stimulus_penalty = float(stimulus_penalty)

    def _init(self, stimulus, network, unit_index=None):
        """
        Parameters
        ----------
        module : fnn.model.networks.Network
            network module
        stimulus : fnn.model.stimuli.Stimulus
            stimulus module
        unit_index : int | List[int] | None
            unit index
        """
        super()._init(stimulus=stimulus, network=network, unit_index=unit_index)

        device = lambda x: x.to(device=self.network.device)

        self.trial_perspectives = device(self.trial_perspectives)
        self.trial_modulations = device(self.trial_modulations)
        self.trial_units = device(self.trial_units)

    def __call__(self, training=True):
        """Perform an objective call

        Parameters
        ----------
        training : bool
            training or validation
        """
        self.stimulus.reset()
        self.network.reset()

        if training and self.sample_trial:
            trial = torch.randint(0, self.trials, (1,)).item()
            select = lambda x: x[:, trial, None]
            expand = lambda x: x.unsqueeze(0)
        else:
            select = lambda x: x
            expand = lambda x: x.expand(self.trials, -1, -1, -1)

        if training and self.sample_stream:
            stream = torch.randint(0, self.network.streams, (1,)).item()
        else:
            stream = None

        with self.network.train_context(training):
            losses = []

            inputs = zip(
                self.stimulus(),
                select(self.trial_perspectives),
                select(self.trial_modulations),
                select(self.trial_units),
            )

            for frame, (stimulus, perspective, modulation, unit) in enumerate(inputs):
                loss = self.network.loss(
                    stimulus=expand(stimulus),
                    perspective=perspective,
                    modulation=modulation,
                    unit=unit,
                    stream=stream,
                )

                if frame < self.burnin_frames:
                    continue

                elif self.unit_index is None:
                    loss = loss.mean()

                else:
                    loss = loss[:, self.unit_index].mean()

                losses.append(loss)

            assert frame + 1 == self.frames, "Unexpected number of frames"

            loss = torch.stack(losses).sum()
            penalty = self.stimulus.penalty()

            loss_item = loss.item()
            penalty_item = penalty.item()

            if not np.isfinite(loss_item):
                raise ValueError("Non-finite loss")

            if not np.isfinite(penalty_item):
                raise ValueError("Non-finite penalty")

            if training:
                stream = (loss + self.stimulus_penalty * penalty).backward()

                self.log["training_loss"].append(loss_item)
                self.log["training_penalty"].append(penalty_item)

            else:
                self.log["validation_loss"].append(loss_item)
                self.log["validation_penalty"].append(penalty_item)


class Excitation(StimulusObjective):
    """Excitation"""

    def __init__(self, temperature, sample_stream=True, burnin_frames=0, stimulus_penalty=0):
        """
        Parameters
        ----------
        temperature : float
            exponential temperature
        sample_stream : bool
            sample stream during training
        burnin_frames : int
            number of initial frames to discard
        stimulus_penalty : float
            stimulus penalty weight
        """
        assert temperature >= 0
        assert burnin_frames >= 0
        assert stimulus_penalty >= 0

        self.temperature = float(temperature)
        self.sample_stream = bool(sample_stream)
        self.burnin_frames = int(burnin_frames)
        self.stimulus_penalty = float(stimulus_penalty)

    def _init(self, stimulus, network, unit_index=None):
        """
        Parameters
        ----------
        module : fnn.model.networks.Network
            network module
        stimulus : fnn.model.stimuli.Stimulus
            stimulus module
        unit_index : int | List[int] | None
            unit index
        """
        super()._init(stimulus=stimulus, network=network, unit_index=unit_index)

        tensor = lambda x: torch.tensor(x, dtype=torch.float, device=self.network.device).unsqueeze(0)

        self.perspective = tensor(self.network.default_perspective)
        self.modulation = tensor(self.network.default_modulation)

    def __call__(self, training=True):
        """Perform an objective call

        Parameters
        ----------
        training : bool
            training or validation
        """
        self.stimulus.reset()
        self.network.reset()

        if training and self.sample_stream:
            stream = torch.randint(0, self.network.streams, (1,)).item()
        else:
            stream = None

        with self.network.train_context(training):
            losses = []

            for frame, stimulus in enumerate(self.stimulus()):
                out = self.network(
                    stimulus=stimulus.unsqueeze(0),
                    perspective=self.perspective,
                    modulation=self.modulation,
                    stream=stream,
                )

                if frame < self.burnin_frames:
                    continue

                if self.unit_index is not None:
                    out = out[:, self.unit_index]

                loss = out.pow(-self.temperature).mean()
                losses.append(loss)

            loss = torch.stack(losses).sum()
            penalty = self.stimulus.penalty()

            loss_item = loss.item()
            penalty_item = penalty.item()

            if not np.isfinite(loss_item):
                raise ValueError("Non-finite loss")

            if not np.isfinite(penalty_item):
                raise ValueError("Non-finite penalty")

            if training:
                stream = (loss + self.stimulus_penalty * penalty).backward()

                self.log["training_loss"].append(loss_item)
                self.log["training_penalty"].append(penalty_item)

            else:
                self.log["validation_loss"].append(loss_item)
                self.log["validation_penalty"].append(penalty_item)
