import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from wastevision.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from wastevision.data.transforms import denormalize, evaluation_transforms, training_transforms

IMAGE_SIZES: tuple[tuple[int, int], ...] = ((512, 384), (384, 512), (224, 224), (100, 80))


def noise_image(seed: int, size: tuple[int, int] = (512, 384)) -> Image.Image:
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 256, (size[1], size[0], 3), dtype=torch.uint8, generator=generator)
    return Image.fromarray(pixels.numpy())


def mean_color_image(size: tuple[int, int] = (512, 384)) -> Image.Image:
    color = tuple(round(value * 255) for value in IMAGENET_MEAN)
    return Image.new("RGB", size, color)


@pytest.mark.parametrize("size", IMAGE_SIZES)
@pytest.mark.parametrize("transform_factory", [training_transforms, evaluation_transforms])
def test_transforms_return_normalized_float_tensor(size, transform_factory):
    tensor = transform_factory()(noise_image(seed=1, size=size))

    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_evaluation_transform_subtracts_imagenet_mean():
    tensor = evaluation_transforms()(mean_color_image())

    assert tensor.abs().max() < 0.01


def test_evaluation_transform_divides_by_imagenet_std():
    tensor = evaluation_transforms()(Image.new("RGB", (512, 384), (255, 255, 255)))

    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)):
        assert torch.allclose(tensor[channel], torch.tensor((1.0 - mean) / std), atol=1e-5)


def test_evaluation_transform_is_deterministic():
    image = noise_image(seed=2)
    transform = evaluation_transforms()

    assert torch.equal(transform(image), transform(image))


def test_evaluation_transform_crops_the_center():
    image = Image.new("RGB", (512, 384), (255, 0, 0))
    image.paste(mean_color_image((480, 352)), (16, 16))

    tensor = evaluation_transforms()(image)

    assert tensor.abs().max() < 0.01


def test_training_transform_changes_between_calls():
    image = noise_image(seed=3)
    transform = training_transforms()
    torch.manual_seed(0)

    assert not torch.equal(transform(image), transform(image))


def test_training_transform_is_reproducible_with_a_seed():
    image = noise_image(seed=4)
    transform = training_transforms()

    torch.manual_seed(7)
    first = transform(image)
    torch.manual_seed(7)
    second = transform(image)

    assert torch.equal(first, second)


def test_denormalize_reverses_normalisation():
    image = Image.new("RGB", (224, 224), (30, 128, 250))
    expected = torch.tensor([30, 128, 250], dtype=torch.float32).view(3, 1, 1) / 255

    restored = denormalize(evaluation_transforms()(image))

    assert torch.allclose(restored, expected.expand(3, IMAGE_SIZE, IMAGE_SIZE), atol=1e-5)


def test_denormalize_clamps_to_displayable_range():
    restored = denormalize(torch.full((3, 2, 2), 10.0))

    assert restored.max() == 1.0
