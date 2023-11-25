from __future__ import annotations

from unittest.mock import Mock

from objectory import OBJECT_TARGET
from torchdata.dataloader2 import DataLoader2

from gravitorchdata.creators.dataloader2 import (
    DataLoader2Creator,
    is_dataloader2_creator_config,
    setup_dataloader2_creator,
)

###################################################
#     Tests for is_dataloader2_creator_config     #
###################################################


def test_is_dataloader2_creator_config_true() -> None:
    assert is_dataloader2_creator_config(
        {OBJECT_TARGET: "gravitorchdata.creators.dataloader2.DataLoader2Creator"}
    )


def test_is_dataloader2_creator_config_false() -> None:
    assert not is_dataloader2_creator_config({"_target_": "torch.nn.Identity"})


###############################################
#     Tests for setup_dataloader2_creator     #
###############################################


def test_setup_dataloader2_creator_object() -> None:
    creator = DataLoader2Creator(Mock(spec=DataLoader2))
    assert setup_dataloader2_creator(creator) is creator


def test_setup_dataloader2_creator_dict() -> None:
    assert isinstance(
        setup_dataloader2_creator(
            {
                OBJECT_TARGET: "gravitorchdata.creators.dataloader2.DataLoader2Creator",
                "dataloader": Mock(spec=DataLoader2),
            }
        ),
        DataLoader2Creator,
    )
