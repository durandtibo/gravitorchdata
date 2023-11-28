from __future__ import annotations

from unittest.mock import Mock, patch

from coola import objects_are_equal
from gravitorch.engines import BaseEngine
from redcat.datapipes.iter import MiniBatcher
from startorch.example import SwissRollExampleGenerator

from gravitorchdata.creators.datapipe import StartorchExampleDataPipeCreator

#####################################################
#     Tests for StartorchExampleDataPipeCreator     #
#####################################################


def test_startorch_example_datapipe_creator_str() -> None:
    assert str(
        StartorchExampleDataPipeCreator(SwissRollExampleGenerator(), num_examples=32, batch_size=8)
    ).startswith("StartorchExampleDataPipeCreator(")


def test_startorch_example_datapipe_creator_create() -> None:
    creator = StartorchExampleDataPipeCreator(
        SwissRollExampleGenerator(), num_examples=32, batch_size=8
    )
    datapipe = creator.create()
    assert isinstance(datapipe, MiniBatcher)
    assert datapipe.batch_size == 8
    assert datapipe.random_seed == 11182458820758237424


@patch("gravitorchdata.creators.datapipe.startorch_example.dist.get_rank", lambda *args: 0)
def test_startorch_example_datapipe_creator_create_engine_rank_0() -> None:
    creator = StartorchExampleDataPipeCreator(
        SwissRollExampleGenerator(), num_examples=32, batch_size=8
    )
    datapipe = creator.create(engine=Mock(spec=BaseEngine, epoch=5, max_epochs=10))
    assert isinstance(datapipe, MiniBatcher)
    assert datapipe.batch_size == 8
    assert datapipe.random_seed == 11182458820758237429


@patch("gravitorchdata.creators.datapipe.startorch_example.dist.get_rank", lambda *args: 1)
def test_startorch_example_datapipe_creator_create_engine_rank_1() -> None:
    creator = StartorchExampleDataPipeCreator(
        SwissRollExampleGenerator(), num_examples=32, batch_size=8
    )
    datapipe = creator.create(engine=Mock(spec=BaseEngine, epoch=5, max_epochs=10))
    assert isinstance(datapipe, MiniBatcher)
    assert datapipe.batch_size == 8
    assert datapipe.random_seed == 11182458820758237439


def test_startorch_example_datapipe_creator_create_repeat() -> None:
    creator = StartorchExampleDataPipeCreator(
        SwissRollExampleGenerator(), num_examples=32, batch_size=8
    )
    datapipe1 = creator.create()
    datapipe2 = creator.create()
    assert objects_are_equal(tuple(datapipe1), tuple(datapipe2))
