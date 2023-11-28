from __future__ import annotations

__all__ = ["StartorchExampleDataPipeCreator"]

import logging
from collections.abc import Sequence

from coola.utils import str_indent, str_mapping
from gravitorch import distributed as dist
from gravitorch.creators.datapipe import BaseDataPipeCreator
from gravitorch.engines import BaseEngine
from gravitorch.utils.seed import get_torch_generator
from redcat.datapipes.iter import MiniBatcher
from startorch.example import BaseExampleGenerator, setup_example_generator
from torch.utils.data import IterDataPipe

logger = logging.getLogger(__name__)


class StartorchExampleDataPipeCreator(BaseDataPipeCreator):
    r"""Implements a ``DataPipe`` creator that generates startorch
    examples and then instantiates a ``DataPipe`` to create batches.

    Args:
    ----
        example_generator (``BaseExampleGenerator`` or dict):
            Specifies the example generator or its configuration.
        num_examples (int): Specifies the number of examples.
        batch_size (int): Specifies the batch size.
        drop_last (bool, optional): If ``True``, it drops the last
            incomplete batch, if the number of examples is not
            divisible by the batch size. If ``False`` and the number
            of examples is not divisible by the batch size, then the
            last batch will be smaller. Default: ``False``
        shuffle (bool, optional): If ``True``, the batches are
            shuffled before to create the mini-batches. The
            shuffling is done per batch. Default: ``False``
        random_seed (int, optional): Specifies the random seed used to
            shuffle the batch before to split it.
            Default: ``11182458820758237424``
        example_seed (int, optional): Specifies the random seed used to
            generate the examples.
            Default: ``12620731324495683517``

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import SwissRollExampleGenerator
        >>> from gravitorchdata.creators.datapipe import StartorchExampleDataPipeCreator
        >>> creator = StartorchExampleDataPipeCreator(
        ...     SwissRollExampleGenerator(), num_examples=32, batch_size=8
        ... )
        >>> creator
        StartorchExampleDataPipeCreator(
          (example_generator): SwissRollExampleGenerator(noise_std=0.0, spin=1.5, hole=False)
          (num_examples): 32
          (batch_size): 8
          (drop_last): False
          (shuffle): False
          (random_seed): 11182458820758237424
          (example_seed): 12620731324495683517
        )
        >>> datapipe = creator.create()
        >>> datapipe
        MiniBatcherIterDataPipe
    """

    def __init__(
        self,
        example_generator: BaseExampleGenerator | dict,
        num_examples: int,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
        random_seed: int = 11182458820758237424,
        example_seed: int = 12620731324495683517,
    ) -> None:
        self._example_generator = setup_example_generator(example_generator)
        self._num_examples = int(num_examples)

        self._batch_size = int(batch_size)
        self._drop_last = bool(drop_last)
        self._shuffle = bool(shuffle)
        self._random_seed = random_seed
        self._example_seed = example_seed

        # This variable is used to store generated examples
        self._cached_examples = None

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "example_generator": self._example_generator,
                    "num_examples": f"{self._num_examples:,}",
                    "batch_size": f"{self._batch_size:,}",
                    "drop_last": self._drop_last,
                    "shuffle": self._shuffle,
                    "random_seed": self._random_seed,
                    "example_seed": self._example_seed,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe:
        if self._cached_examples is None:
            logger.info(f"Generating {self._num_examples:,} examples...")
            self._cached_examples = self._example_generator.generate(
                batch_size=self._num_examples,
                rng=get_torch_generator(self._example_seed),
            )

        random_seed = 0
        if engine:
            random_seed = engine.epoch + engine.max_epochs * dist.get_rank()
        return MiniBatcher(
            self._cached_examples,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
            random_seed=self._random_seed + random_seed,
        )
