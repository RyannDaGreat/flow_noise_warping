import torch
import torch.nn.functional as F
from einops import rearrange
from torch_scatter import scatter


def unique_pixels(image):
    """
    Find unique pixel values in an image tensor and return their RGB values, counts, and inverse indices.

    Args:
        image (torch.Tensor): Image tensor of shape [c, h, w], where c is the number of channels (e.g., 3 for RGB),
                              h is the height, and w is the width of the image.

    Returns:
        tuple: A tuple containing three tensors:
            - unique_colors (torch.Tensor): Tensor of shape [u, c] representing the unique RGB values found in the image,
                                            where u is the number of unique colors.
            - counts (torch.Tensor): Tensor of shape [u] representing the counts of each unique color.
            - index_matrix (torch.Tensor): Tensor of shape [h, w] representing the inverse indices of each pixel,
                                           mapping each pixel to its corresponding unique color index.
    """
    c, h, w = image.shape

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Find unique RGB values, counts, and inverse indices
    unique_colors, inverse_indices, counts = torch.unique(flattened_pixels, dim=0, return_inverse=True, return_counts=True)

    # Get the number of unique indices
    u = unique_colors.shape[0]

    # Reshape the inverse indices back to the original image dimensions [h, w] using einops
    index_matrix = rearrange(inverse_indices, "(h w) -> h w", h=h, w=w)

    # Assert the shapes of the output tensors
    assert unique_colors.shape == (u, c), f"Expected unique_colors shape: ({u}, {c}), but got: {unique_colors.shape}"
    assert counts.shape == (u,), f"Expected counts shape: ({u},), but got: {counts.shape}"
    assert index_matrix.shape == (h, w), f"Expected index_matrix shape: ({h}, {w}), but got: {index_matrix.shape}"

    return unique_colors, counts, index_matrix


def sum_indexed_values(image, index_matrix):
    """
    Sum the values in the CHW image tensor based on the indices specified in the HW index matrix.

    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W], where C is the number of channels,
                              H is the height, and W is the width of the image.
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique value.
                                     Indices range [0, U), where U is the number of unique indices

    Returns:
        torch.Tensor: Tensor of shape [U, C] representing the sum of values in the image tensor
                      based on the indices in the index matrix, where U is the number of unique
                      indices in the index matrix.
    """
    c, h, w = image.shape
    u = index_matrix.max() + 1

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Scatter sum the flattened pixel values using the index matrix
    output = scatter(flattened_pixels, index_matrix.view(-1), dim=0, dim_size=u, reduce="sum")

    # Assert the shapes of the input and output tensors
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"
    assert index_matrix.shape == (h, w), f"Expected index_matrix shape: ({h}, {w}), but got: {index_matrix.shape}"
    assert output.shape == (u, c), f"Expected output shape: ({u}, {c}), but got: {output.shape}"

    return output


def indexed_to_image(index_matrix, unique_colors):
    """
    Create a CHW image tensor from an HW index matrix and a UC unique_colors matrix.

    Args:
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique color.
        unique_colors (torch.Tensor): Unique colors matrix tensor of shape [U, C] containing
                                      the unique color values, where U is the number of unique
                                      colors and C is the number of channels.

    Returns:
        torch.Tensor: Image tensor of shape [C, H, W] representing the reconstructed image
                      based on the index matrix and unique colors matrix.
    """
    h, w = index_matrix.shape
    u, c = unique_colors.shape

    # Assert the shapes of the input tensors
    assert index_matrix.max() < u, f"Index matrix contains indices ({index_matrix.max()}) greater than the number of unique colors ({u})"

    # Gather the colors based on the index matrix
    flattened_image = unique_colors[index_matrix.view(-1)]

    # Reshape the flattened image to [h, w, c]
    image = rearrange(flattened_image, "(h w) c -> h w c", h=h, w=w)

    # Rearrange the image tensor from [h, w, c] to [c, h, w] using einops
    image = rearrange(image, "h w c -> c h w")

    # Assert the shape of the output tensor
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"

    return image


def demo_pixellation_via_proxy():
    real_image = as_torch_image(
        rp.cv_resize_image(
            load_image("https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_square.jpg"),
            (512, 512),
        )
    )

    c, h, w = real_image.shape

    noise_image = torch.randn(c, h // 4, w // 4)

    # Resize noise_image using nearest-neighbor interpolation to match the dimensions of real_image
    pixelated_noise_image = torch_resize_image(noise_image, 4, "nearest")
    assert pixelated_noise_image.shape==(c,h,w)

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(pixelated_noise_image)

    # Sum the color values from real_image based on the indices of the unique noise pixels
    summed_colors = sum_indexed_values(real_image, index_matrix)

    # Divide the summed color values by the counts to get the average color for each unique pixel
    average_colors = summed_colors / rearrange(counts, "u -> u 1")

    # Create a new pixelated image using the average colors and the index matrix
    pixelated_dog_image = indexed_to_image(index_matrix, average_colors)

    display_image(pixelated_dog_image)

def demo():
    real_image = as_torch_image(
        rp.cv_resize_image(
            load_image("https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_square.jpg"),
            (512, 512),
        )
    )

    c, h, w = real_image.shape

    noise_image = torch.randn(c, h // 4, w // 4)

    # Resize noise_image using nearest-neighbor interpolation to match the dimensions of real_image
    pixelated_noise_image = torch_resize_image(noise_image, 4, "nearest")
    assert pixelated_noise_image.shape==(c,h,w)

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(pixelated_noise_image)

    # Sum the color values from real_image based on the indices of the unique noise pixels
    summed_colors = sum_indexed_values(real_image, index_matrix)

    # Divide the summed color values by the counts to get the average color for each unique pixel
    average_colors = summed_colors / rearrange(counts, "u -> u 1")

    # Create a new pixelated image using the average colors and the index matrix
    pixelated_dog_image = indexed_to_image(index_matrix, average_colors)

    display_image(pixelated_dog_image)