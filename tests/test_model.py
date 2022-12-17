import pytest
import torch

from src.ml_utils import get_model


@pytest.mark.parametrize('batch_size', [1])
def test_model(batch_size: str) -> None:
    model = get_model()
    image = torch.rand((batch_size, 3, 192, 192))
    prediction = model(image)
