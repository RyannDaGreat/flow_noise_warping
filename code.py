import torch
from einops import rearrange
import rp
from rp import as_torch_image, display_image, load_image, torch_resize_image
from icecream import ic

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
    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.shape == (h, w)
    assert index_matrix.min() == 0
    assert index_matrix.max() == u - 1

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

    # Create an output tensor of shape [u, c] initialized with zeros
    output = torch.zeros((u, c), dtype=flattened_pixels.dtype, device=flattened_pixels.device)

    # Scatter sum the flattened pixel values using the index matrix
    output.index_add_(0, index_matrix.view(-1), flattened_pixels)

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
    
def calculate_wave_pattern(h, w, frame):
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    
    # Calculate the distance from the center of the image
    center_x, center_y = w // 2, h // 2
    dist_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Calculate the angle from the center of the image
    angle_from_center = torch.atan2(y - center_y, x - center_x)
    
    # Calculate the wave pattern based on the distance and angle
    wave_freq = 0.05  # Frequency of the waves
    wave_amp = 10.0   # Amplitude of the waves
    wave_offset = frame * 0.05  # Offset for animation
    
    dx = wave_amp * torch.cos(dist_from_center * wave_freq + angle_from_center + wave_offset)
    dy = wave_amp * torch.sin(dist_from_center * wave_freq + angle_from_center + wave_offset)
    
    return dx, dy

def starfield_zoom(h, w, frame):
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    
    # Calculate the distance from the center of the image
    center_x, center_y = w // 2, h // 2
    dist_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Calculate the angle from the center of the image
    angle_from_center = torch.atan2(y - center_y, x - center_x)
    
    # Calculate the starfield zoom effect
    zoom_speed = 0.01  # Speed of the zoom effect
    zoom_scale = 1.0 + frame * zoom_speed  # Scale factor for the zoom effect
    
    # Calculate the displacement based on the distance and angle
    dx = dist_from_center * torch.cos(angle_from_center) / zoom_scale
    dy = dist_from_center * torch.sin(angle_from_center) / zoom_scale
    
    return dx, dy

def warp_noise(noise, dx, dy, s=4):
    #This is *certainly* imperfect. We need to have particle swarm in addition to this.

    c, h, w = noise.shape
    assert dx.shape==(h,w)
    assert dy.shape==(h,w)

    #s is scaling factor
    hs = h * s
    ws = w * s
    
    #Upscale the warping with linear interpolation. Also scale it appropriately.
    up_dx = rp.torch_resize_image(dx[None], (hs, ws), interp="bilinear")[0]
    up_dy = rp.torch_resize_image(dy[None], (hs, ws), interp="bilinear")[0]
    up_dx *= s
    up_dy *= s

    up_noise = rp.torch_resize_image(noise, (hs, ws), interp="nearest")
    assert up_noise.shape == (c, hs, ws)

    up_noise = rp.torch_remap_image(up_noise, up_dx, up_dy, relative=True, interp="nearest", add_alpha_mask=True)
    up_noise, alpha = up_noise[:-1], up_noise[-1:]
    assert up_noise.shape == (c, hs, ws)
    assert alpha.shape == (1, hs, ws)
    
    # Fill occluded regions with noise...
    fill_noise = torch.randn_like(up_noise)
    up_noise = up_noise * alpha + fill_noise * (1-alpha)
    assert up_noise.shape == (c, hs, ws)

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(up_noise)
    u = len(unique_colors)
    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.max() == u - 1
    assert index_matrix.min() == 0
    assert index_matrix.shape == (hs, ws)

    foreign_noise = torch.randn_like(up_noise)
    assert foreign_noise.shape == up_noise.shape == (c, hs, ws)

    summed_foreign_noise_colors = sum_indexed_values(foreign_noise, index_matrix)
    assert summed_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise_colors = summed_foreign_noise_colors / rearrange(counts, "u -> u 1")
    assert meaned_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise = indexed_to_image(index_matrix, meaned_foreign_noise_colors)
    assert meaned_foreign_noise.shape == (c, hs, ws)

    zeroed_foreign_noise = foreign_noise - meaned_foreign_noise
    assert zeroed_foreign_noise.shape == (c, hs, ws)

    counts_as_colors = rearrange(counts, "u -> u 1")
    counts_image = indexed_to_image(index_matrix, counts_as_colors)
    assert counts_image.shape == (1, hs, ws)

    #To upsample noise, we must first divide by the area then add zero-sum-noise
    output = up_noise
    output = output / counts_image ** .5
    output = output + zeroed_foreign_noise

    #Now we resample the noise back down again
    #PLEASE HOPE AREA DOWNSAMPLING WORKS PROPERLY...UNVERIFIED...I think I remember it not working?
    output = rp.torch_resize_image(output, (h, w), interp='area')
    output = output * s #Adjust variance by multiplying by sqrt of area, aka sqrt(s*s)=s

    return output
    
def demo_noise_warp():
    d=rp.JupyterDisplayChannel()
    d.display()
    device='cuda'
    h=w=256
    noise=torch.randn(3,h,w).to(device)
    wdx,wdy=calculate_wave_pattern(h,w,frame=0)
    sdx,sdy=starfield_zoom(h,w,frame=1)

    dx=sdx+2*wdx
    dy=sdy+2*wdy
    
    dx/=dx.max()
    dy/=dy.max()
    Q=-6
    dy*=Q
    dx*=Q
    dx=dx.to(device)
    dy=dy.to(device)
    new_noise=noise


    for _ in range(10000):
        # ic(new_noise.device,dx.device,dy.device)

        new_noise=warp_noise(new_noise,dx,dy,2)
        # display_image(new_noise)
        d.update(rp.as_numpy_image(new_noise/4+.5))


def webcam_demo():
    import cv2
    import numpy as np
    
    def draw_hsv(flow, scale=8):
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        scaled_mag = mag * scale
        scaled_mag = np.clip(scaled_mag, 0, 255)  # Ensure it fits in the value range
        hsv[..., 2] = scaled_mag.astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
    
    def resize_frame(frame, target_height=128):
        aspect_ratio = frame.shape[1] / frame.shape[0]
        target_width = int(target_height * aspect_ratio)
        resized_frame = cv2.resize(frame, (target_width, target_height))
        return resized_frame
    
    def main():
        cap = cv2.VideoCapture(0)
        ret, prev_frame = cap.read()
        prev_frame = resize_frame(prev_frame)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
        # Initialize DeepFlow Optical Flow
        optical_flow = cv2.optflow.createOptFlow_DeepFlow()
    
    
        d=rp.JupyterDisplayChannel()
        d.display()
        device='cpu'
        h,w=get_image_dimensions(prev_frame)
        noise=torch.randn(3,h,w).to(device)
        wdx,wdy=calculate_wave_pattern(h,w,frame=0)
        sdx,sdy=starfield_zoom(h,w,frame=1)
    
        dx=sdx+2*wdx
        dy=sdy+2*wdy
        
        dx/=dx.max()
        dy/=dy.max()
        Q=-6
        dy*=Q
        dx*=Q
        dx=dx.to(device)
        dy=dy.to(device)
        new_noise=noise
    
    
    
    
        while True:
            ret, frame = cap.read()
            frame = resize_frame(frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            # Compute the optical flow
            flow = optical_flow.calc(prev_gray, frame_gray, None)
    
            # Visualization
            flow_bgr = draw_hsv(flow)
    
            x=flow[:,:,0]
            y=flow[:,:,1]
    
            dx=-torch.Tensor(x)
            dy=-torch.Tensor(y)
            
            # Display the original and flow side by side
            combined_img = np.hstack((frame, flow_bgr))
            cv2.imshow('Frame and Optical Flow', combined_img)
            
            prev_gray = frame_gray.copy()
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
            new_noise=warp_noise(new_noise,dx,dy,1)
            display_image(
                tiled_images(
                    [
                        as_numpy_image(new_noise / 2 + 0.5),
                        cv_bgr_rgb_swap(frame),
                        cv_bgr_rgb_swap(flow_bgr),
                    ]
                )
            )
    
            # d.update(rp.as_numpy_image(new_noise/4+.5))
    
        cap.release()
        cv2.destroyAllWindows()
    
    main()