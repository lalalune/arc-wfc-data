import json
import uuid
import numpy as np
import os
from random import randint
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool
from .wfc import WaveFunctionCollapse

# Color palette
colors_rgb = {
    0: (0x00, 0x00, 0x00),
    1: (0x00, 0x74, 0xD9),
    2: (0xFF, 0x41, 0x36),
    3: (0x2E, 0xCC, 0x40),
    4: (0xFF, 0xDC, 0x00),
    5: (0xA0, 0xA0, 0xA0),
    6: (0xF0, 0x12, 0xBE),
    7: (0xFF, 0x85, 0x1B),
    8: (0x7F, 0xDB, 0xFF),
    9: (0x87, 0x0C, 0x25),
}

# make a custom cmap
cmap = plt.cm.colors.ListedColormap([np.array(c) / 255 for c in colors_rgb.values()])


def generate_random_pattern(size, max_colors=3):
    colors = np.random.choice(range(1, 10), size=max_colors, replace=False)
    colors = np.insert(colors, 0, 0)
    pattern = np.random.choice(colors, size=size)
    for _ in range(2):
        shape_color = np.random.choice(colors[1:])
        shape_size = (randint(1, size[0] // 2), randint(1, size[1] // 2))
        shape_pos = (
            randint(0, size[0] - shape_size[0]),
            randint(0, size[1] - shape_size[1]),
        )
        pattern[
            shape_pos[0] : shape_pos[0] + shape_size[0],
            shape_pos[1] : shape_pos[1] + shape_size[1],
        ] = shape_color
    return pattern.astype(np.uint8)


def save_pattern_as_image(pattern, filename):
    img = Image.fromarray(pattern)
    img.save(filename)


def wfc_pattern_expansion(input_pattern, output_size, start_point):
    # Generate pattern image for visualization purposes
    pattern_image = np.zeros(
        (input_pattern.shape[0], input_pattern.shape[1], 3), dtype=np.uint8
    )
    print("pattern image shape", pattern_image.shape)
    for i in range(input_pattern.shape[0]):
        for j in range(input_pattern.shape[1]):
            pattern_image[i, j] = colors_rgb[input_pattern[i, j]]

    # Generate a unique filename for each run to avoid collisions
    temp_filename = f"temp_{uuid.uuid4()}.png"
    pil_image = Image.fromarray(pattern_image)
    pil_image.save(temp_filename)
    sample = plt.imread(temp_filename)
    os.remove(temp_filename)  # Clean up after reading
    sample = np.expand_dims(sample, axis=0)[:, :, :, :3]

    wfc = WaveFunctionCollapse(
        (1, *output_size), sample, (1, 2, 2), use_multiprocessing=False
    )

    # Fetch the cell at the start_point
    cell = wfc.grid.get_cell(start_point)

    # Execute WFC algorithm
    while True:
        if wfc.step():
            break

    # Generate the output pattern
    output_pattern_rgb = np.squeeze(wfc.get_image(), axis=0)
    output_pattern = np.zeros(output_size, dtype=np.uint8)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            rgb_value = tuple(output_pattern_rgb[i, j] * 255)
            min_distance = float("inf")
            closest_color_index = None
            for color_index, color_rgb in colors_rgb.items():
                distance = sum(abs(c1 - c2) for c1, c2 in zip(rgb_value, color_rgb))
                if distance < min_distance:
                    min_distance = distance
                    closest_color_index = color_index
            output_pattern[i, j] = closest_color_index
    return output_pattern


def save_plots(input_pattern, output_pattern, filename, title="Pattern"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    ax1.imshow(input_pattern, interpolation="nearest", cmap=cmap)
    ax1.set_title("Input")
    ax1.axis("off")
    ax2.imshow(output_pattern, interpolation="nearest", cmap=cmap)
    ax2.set_title("Output")
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def generate_and_save_plots(input_pattern, output_pattern, index, output_dir):
    plot_filename = os.path.join(output_dir, f"example_{index}_plot.png")
    save_plots(input_pattern, output_pattern, plot_filename, title=f"Example {index}")
    return plot_filename


def generate_challenge(args):
    (input_size, output_size, num_examples, output_dir, is_train) = args
    output_pattern = generate_random_pattern(input_size, randint(2, 4))

    # Calculate maximum valid start points
    max_start_x = max(0, (input_size[0] - output_size[0]) - 1)
    max_start_y = max(0, (input_size[1] - output_size[1]) - 1)

    # Generate random start point within the valid range
    start_point = (randint(0, max_start_x), randint(0, max_start_y))

    input_pattern = wfc_pattern_expansion(output_pattern, output_size, start_point)

    examples = []
    for i in range(num_examples):
        flip_h = np.random.choice([True, False])
        flip_v = np.random.choice([True, False])
        input_flipped = np.flip(
            input_pattern, axis=(0 if flip_v else 1) if flip_h else ()
        )
        output_flipped = np.flip(
            output_pattern, axis=(0 if flip_v else 1) if flip_h else ()
        )
        color_map = np.random.permutation(np.arange(len(colors_rgb)))
        input_color_changed = color_map[input_flipped]
        output_color_changed = color_map[output_flipped]
        plot_filename = generate_and_save_plots(
            input_color_changed, output_color_changed, i, output_dir
        )
        example = {
            "input": input_color_changed.tolist(),
            "output": output_color_changed.tolist(),
            "plot": plot_filename,
        }
        examples.append(example)
    challenge = {
        "train": examples,
        "test": [{"input": input_pattern.tolist(), "output": output_pattern.tolist()}],
    }
    challenge_hash = f"{randint(0, 0xFFFFFFFF):08x}"
    with open(os.path.join(output_dir, f"{challenge_hash}.json"), "w") as f:
        json.dump(challenge, f, indent=4)
    return f"Challenge saved: {challenge_hash}.json"


def generate_fewshot_challenges(num_challenges, train_ratio, output_dir):
    num_train_challenges = int(num_challenges * train_ratio)
    num_eval_challenges = num_challenges - num_train_challenges
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "training")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    args = []
    for i in range(num_train_challenges):
        input_size = (randint(2, 3), randint(2, 3))
        output_size = (randint(5, 16), randint(5, 16))
        num_examples_per_challenge = randint(2, 4)
        args.append(
            (input_size, output_size, num_examples_per_challenge, train_dir, True)
        )
    for i in range(num_eval_challenges):
        input_size = (randint(2, 6), randint(2, 6))
        output_size = (randint(input_size[0], 10), randint(input_size[1], 10))
        num_examples_per_challenge = randint(2, 4)
        args.append(
            (input_size, output_size, num_examples_per_challenge, eval_dir, False)
        )
    with Pool(processes=10) as pool:
        results = pool.map(generate_challenge, args)
    print(f"Challenges generated and saved in {output_dir}")


if __name__ == "__main__":
    import argparse
    # num_challenges and train_ratio should be argparsed
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_challenges", type=int, default=1000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default="data_wfc")
    args = parser.parse_args()
    
    num_challenges = args.num_challenges
    train_ratio = args.train_ratio
    output_dir = args.output_dir
    generate_fewshot_challenges(num_challenges, train_ratio, output_dir)
    print(f"Fewshot challenges generated and saved in {output_dir}")
