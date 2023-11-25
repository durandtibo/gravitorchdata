from __future__ import annotations

from unittest.mock import patch

from gravitorch.testing import torchdata_available
from objectory import OBJECT_TARGET
from pytest import mark, raises
from torch.utils.data.datapipes.iter import IterableWrapper, Shuffler
from torch.utils.data.graph import DataPipe
from torchdata.dataloader2 import (
    DataLoader2,
    InProcessReadingService,
    ReadingServiceInterface,
)
from torchdata.dataloader2.adapter import Adapter, Shuffle

from gravitorchdata.dataloaders import (
    create_dataloader2,
    is_dataloader2_config,
    setup_dataloader2,
)

########################################
#     Tests for create_dataloader2     #
########################################


@torchdata_available
@mark.parametrize(
    "datapipe",
    (
        IterableWrapper(range(10)),
        {OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper", "iterable": range(10)},
    ),
)
def test_create_dataloader2_datapipe(datapipe: DataPipe | dict) -> None:
    dataloader = create_dataloader2(datapipe)
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, IterableWrapper)
    assert dataloader.datapipe_adapter_fns is None
    assert dataloader.reading_service is None


@torchdata_available
@mark.parametrize(
    "datapipe_adapter_fn",
    (
        Shuffle(),
        {OBJECT_TARGET: "torchdata.dataloader2.adapter.Shuffle"},
    ),
)
def test_create_dataloader2_datapipe_adapter_fn(datapipe_adapter_fn: Adapter | dict) -> None:
    dataloader = create_dataloader2(
        Shuffler(IterableWrapper(range(10))), datapipe_adapter_fn=datapipe_adapter_fn
    )
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, Shuffler)
    assert len(dataloader.datapipe_adapter_fns) == 1
    assert isinstance(dataloader.datapipe_adapter_fns[0], Shuffle)
    assert dataloader.reading_service is None


@torchdata_available
@mark.parametrize(
    "reading_service",
    (
        InProcessReadingService(),
        {OBJECT_TARGET: "torchdata.dataloader2.InProcessReadingService"},
    ),
)
def test_create_dataloader2_reading_service(
    reading_service: ReadingServiceInterface | dict,
) -> None:
    dataloader = create_dataloader2(IterableWrapper(range(10)), reading_service=reading_service)
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, IterableWrapper)
    assert dataloader.datapipe_adapter_fns is None
    assert isinstance(dataloader.reading_service, InProcessReadingService)


def test_create_dataloader2_without_torchdata() -> None:
    with patch("gravitorch.utils.imports.is_torchdata_available", lambda *args: False):
        with raises(RuntimeError, match="`torchdata` package is required but not installed."):
            create_dataloader2(IterableWrapper(range(10)))


###########################################
#     Tests for is_dataloader2_config     #
###########################################


@torchdata_available
def test_is_dataloader2_config_true() -> None:
    assert is_dataloader2_config({OBJECT_TARGET: "torchdata.dataloader2.DataLoader2"})


@torchdata_available
def test_is_dataloader2_config_false() -> None:
    assert not is_dataloader2_config({OBJECT_TARGET: "torch.nn.Identity"})


@torchdata_available
def test_is_dataloader2_config_false_dataloader() -> None:
    assert not is_dataloader2_config({OBJECT_TARGET: "torch.utils.data.DataLoader"})


def test_is_dataloader2_config_without_torchdata() -> None:
    with patch("gravitorch.utils.imports.is_torchdata_available", lambda *args: False):
        with raises(RuntimeError, match="`torchdata` package is required but not installed."):
            is_dataloader2_config({OBJECT_TARGET: "torchdata.dataloader2.DataLoader2"})


#######################################
#     Tests for setup_dataloader2     #
#######################################


@torchdata_available
def test_setup_dataloader2_object() -> None:
    creator = DataLoader2(IterableWrapper((1, 2, 3, 4, 5)))
    assert setup_dataloader2(creator) is creator


@torchdata_available
def test_setup_dataloader2_dict() -> None:
    assert isinstance(
        setup_dataloader2(
            {
                OBJECT_TARGET: "torchdata.dataloader2.DataLoader2",
                "datapipe": {
                    OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                    "examples": (1, 2, 3, 4, 5),
                },
            },
        ),
        DataLoader2,
    )


def test_setup_dataloader2_without_torchdata() -> None:
    with patch("gravitorch.utils.imports.is_torchdata_available", lambda *args: False):
        with raises(RuntimeError, match="`torchdata` package is required but not installed."):
            setup_dataloader2(
                {
                    OBJECT_TARGET: "torchdata.dataloader2.DataLoader2",
                    "datapipe": {
                        OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                        "examples": (1, 2, 3, 4, 5),
                    },
                },
            )
