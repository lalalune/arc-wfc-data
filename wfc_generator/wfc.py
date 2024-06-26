# From https://github.com/mxgmn/WaveFunctionCollapse

from multiprocessing import Pool
import os
import time
import numpy as np


class Pattern:
    """
    Pattern is a configuration of tiles from the input image.
    """

    index_to_pattern = {}
    color_to_index = {}
    index_to_color = {}

    def __init__(self, data, index):
        self.index = index
        self.data = np.array(data)
        self.legal_patterns_index = {}  # offset -> [pattern_index]

    def get(self, index=None):
        if index is None:
            return self.data.item(0)
        return self.data[index]

    def set_legal_patterns(self, offset, legal_patterns):
        self.legal_patterns_index[offset] = legal_patterns

    @property
    def shape(self):
        return self.data.shape

    def is_compatible(self, candidate_pattern, offset):
        """
        Check if pattern is compatible with a candidate pattern for a given offset
        :param candidate_pattern:
        :param offset:
        :return: True if compatible
        """
        assert self.shape == candidate_pattern.shape

        # Precomputed compatibility
        if offset in self.legal_patterns_index:
            return candidate_pattern.index in self.legal_patterns_index[offset]

        # Computing compatibility
        ok_constraint = True
        start = tuple([max(offset[i], 0) for i, _ in enumerate(offset)])
        end = tuple(
            [
                min(self.shape[i] + offset[i], self.shape[i])
                for i, _ in enumerate(offset)
            ]
        )
        for index in np.ndindex(end):  # index = (x, y, z...)
            start_constraint = True
            for i, d in enumerate(index):
                if d < start[i]:
                    start_constraint = False
                    break
            if not start_constraint:
                continue

            if candidate_pattern.get(
                tuple(np.array(index) - np.array(offset))
            ) != self.get(index):
                ok_constraint = False
                break

        return ok_constraint

    def to_image(self):
        return Pattern.index_to_img(self.data)

    @staticmethod
    def from_sample(sample, pattern_size):
        """
        Compute patterns from sample
        :param pattern_size:
        :param sample:
        :return: list of patterns
        """

        sample = Pattern.sample_img_to_indexes(sample)

        shape = sample.shape
        patterns = []
        pattern_index = 0

        for index, _ in np.ndenumerate(sample):
            # Checking if index is out of bounds
            out = False
            for i, d in enumerate(index):  # d is a dimension, e.g.: x, y, z
                if d > shape[i] - pattern_size[i]:
                    out = True
                    break
            if out:
                continue

            pattern_location = [
                range(d, pattern_size[i] + d) for i, d in enumerate(index)
            ]
            pattern_data = sample[np.ix_(*pattern_location)]

            datas = [pattern_data, np.fliplr(pattern_data)]
            if shape[1] > 1:  # is 2D
                datas.append(np.flipud(pattern_data))
                datas.append(np.rot90(pattern_data, axes=(1, 2)))
                datas.append(np.rot90(pattern_data, 2, axes=(1, 2)))
                datas.append(np.rot90(pattern_data, 3, axes=(1, 2)))

            if shape[0] > 1:  # is 3D
                datas.append(np.flipud(pattern_data))
                datas.append(np.rot90(pattern_data, axes=(0, 2)))
                datas.append(np.rot90(pattern_data, 2, axes=(0, 2)))
                datas.append(np.rot90(pattern_data, 3, axes=(0, 2)))

            # Checking existence
            # TODO: more probability to multiple occurrences when observe phase
            for data in datas:
                exist = False
                for p in patterns:
                    if (p.data == data).all():
                        exist = True
                        break
                if exist:
                    continue

                pattern = Pattern(data, pattern_index)
                patterns.append(pattern)
                Pattern.index_to_pattern[pattern_index] = pattern
                pattern_index += 1

        # Pattern.plot_patterns(patterns)
        return patterns

    @staticmethod
    def sample_img_to_indexes(sample):
        """
        Convert a rgb image to a 2D array with pixel index
        :param sample:
        :return: pixel index sample
        """
        Pattern.color_to_index = {}
        Pattern.index_to_color = {}
        sample_index = np.zeros(sample.shape[:-1])  # without last rgb dim
        color_number = 0
        for index in np.ndindex(sample.shape[:-1]):
            color = tuple(sample[index])
            if color not in Pattern.color_to_index:
                Pattern.color_to_index[color] = color_number
                Pattern.index_to_color[color_number] = color
                color_number += 1

            sample_index[index] = Pattern.color_to_index[color]

        return sample_index

    @staticmethod
    def index_to_img(sample):
        color = next(iter(Pattern.index_to_color.values()))

        image = np.zeros(sample.shape + (len(color),))
        for index in np.ndindex(sample.shape):
            pattern_index = sample[index]
            if pattern_index == -1:
                image[index] = [0.5 for _ in range(len(color))]  # Grey
            else:
                image[index] = Pattern.index_to_color[pattern_index]
        return image

    @staticmethod
    def from_index(pattern_index):
        return Pattern.index_to_pattern[pattern_index]


class Cell:
    """
    Cell is a pixel or tile (in 2d) that stores the possible patterns
    """

    def __init__(self, num_pattern, position, grid):
        self.num_pattern = num_pattern
        self.allowed_patterns = [i for i in range(self.num_pattern)]

        self.position = position
        self.grid = grid
        self.offsets = [
            (z, y, x) for x in range(-1, 2) for y in range(-1, 2) for z in range(-1, 2)
        ]

    def entropy(self):
        return len(self.allowed_patterns)

    def choose_rnd_pattern(self):
        chosen_index = np.random.randint(len(self.allowed_patterns))
        self.allowed_patterns = [self.allowed_patterns[chosen_index]]

    def is_stable(self):
        return len(self.allowed_patterns) == 1

    def get_value(self):
        if self.is_stable():
            pattern = Pattern.from_index(self.allowed_patterns[0])
            return pattern.get()
        return -1

    def get_neighbors(self):
        neighbors = []
        for offset in self.offsets:
            neighbor_pos = tuple(np.array(self.position) + np.array(offset))
            out = False
            for i, d in enumerate(neighbor_pos):
                if not 0 <= d < self.grid.size[i]:
                    out = True
            if out:
                continue

            neighbors.append((self.grid.get_cell(neighbor_pos), offset))

        return neighbors


class Grid:
    """
    Grid is made of Cells
    """

    def __init__(self, size, num_pattern):
        self.size = size
        self.grid = np.empty(self.size, dtype=object)
        for position in np.ndindex(self.size):
            self.grid[position] = Cell(num_pattern, position, self)

        # self.grid = np.array([[Cell(num_pattern, (x, y), self) for x in range(self.size)] for y in range(self.size)])
        # self.grid = np.array([Cell(num_pattern, (x,), self) for x in range(self.size)])

    def find_lowest_entropy(self):
        min_entropy = 999999
        lowest_entropy_cells = []
        for cell in self.grid.flat:
            if cell.is_stable():
                continue

            entropy = cell.entropy()

            if entropy == min_entropy:
                lowest_entropy_cells.append(cell)
            elif entropy < min_entropy:
                min_entropy = entropy
                lowest_entropy_cells = [cell]

        if len(lowest_entropy_cells) == 0:
            return None
        cell = lowest_entropy_cells[np.random.randint(len(lowest_entropy_cells))]
        return cell

    def get_cell(self, index):
        """
        Returns the cell contained in the grid at the provided index
        :param index: (...z, y, x)
        :return: cell
        """
        return self.grid[index]

    def get_image(self):
        """
        Returns the grid converted from index to back to color
        :return:
        """
        image = np.vectorize(lambda c: c.get_value())(self.grid)
        image = Pattern.index_to_img(image)
        return image

    def check_contradiction(self):
        for cell in self.grid.flat:
            if len(cell.allowed_patterns) == 0:
                return True
        return False

    def print_allowed_pattern_count(self):
        grid_allowed_patterns = np.vectorize(lambda c: len(c.allowed_patterns))(
            self.grid
        )
        print(grid_allowed_patterns)


class Propagator:
    """
    Propagator that computes and stores the legal patterns relative to another
    """

    def __init__(self, patterns, use_multiprocessing=True):
        self.patterns = patterns
        self.offsets = [
            (z, y, x) for x in range(-1, 2) for y in range(-1, 2) for z in range(-1, 2)
        ]
        self.use_multiprocessing = use_multiprocessing

        start_time = time.time()
        self.precompute_legal_patterns()
        print(
            "Patterns constraints generation took %s seconds"
            % (time.time() - start_time)
        )

    def precompute_legal_patterns(self):
        if self.use_multiprocessing:
            with Pool(os.cpu_count()) as pool:
                patterns_offsets = [
                    (pattern, offset)
                    for pattern in self.patterns
                    for offset in self.offsets
                ]
                patterns_compatibility = pool.starmap(
                    self.legal_patterns, patterns_offsets
                )
        else:
            patterns_compatibility = [
                self.legal_patterns(pattern, offset)
                for pattern in self.patterns
                for offset in self.offsets
            ]

        for pattern_index, offset, legal_patterns in patterns_compatibility:
            self.patterns[pattern_index].set_legal_patterns(offset, legal_patterns)

    def legal_patterns(self, pattern, offset):
        legal_patt = []
        for candidate_pattern in self.patterns:
            if pattern.is_compatible(candidate_pattern, offset):
                legal_patt.append(candidate_pattern.index)
        return pattern.index, offset, legal_patt

    @staticmethod
    def propagate(cell):
        to_update = [neighbour for neighbour, _ in cell.get_neighbors()]
        while len(to_update) > 0:
            cell = to_update.pop(0)
            for neighbour, offset in cell.get_neighbors():
                for pattern_index in cell.allowed_patterns:
                    pattern = Pattern.from_index(pattern_index)
                    pattern_still_compatible = False
                    for neighbour_pattern_index in neighbour.allowed_patterns:
                        neighbour_pattern = Pattern.from_index(neighbour_pattern_index)

                        if pattern.is_compatible(neighbour_pattern, offset):
                            pattern_still_compatible = True
                            break

                    if not pattern_still_compatible:
                        cell.allowed_patterns.remove(pattern_index)

                        for neigh, _ in cell.get_neighbors():
                            if neigh not in to_update:
                                to_update.append(neigh)


class WaveFunctionCollapse:
    """
    WaveFunctionCollapse encapsulates the wfc algorithm
    """

    def __init__(self, grid_size, sample, pattern_size, use_multiprocessing=True):
        self.patterns = Pattern.from_sample(sample, pattern_size)
        self.grid = self._create_grid(grid_size)
        self.propagator = Propagator(
            self.patterns, use_multiprocessing=use_multiprocessing
        )

    def run(self):
        start_time = time.time()

        done = False
        while not done:
            done = self.step()

        print("WFC run took %s seconds" % (time.time() - start_time))

    def step(self):
        # self.grid.print_allowed_pattern_count()
        cell = self.observe()
        if cell is None:
            return True
        self.propagate(cell)
        return False

    def get_image(self):
        return self.grid.get_image()

    def get_patterns(self):
        return [pattern.to_image() for pattern in self.patterns]

    def observe(self):
        if self.grid.check_contradiction():
            return None
        cell = self.grid.find_lowest_entropy()

        if cell is None:
            return None

        cell.choose_rnd_pattern()

        return cell

    def propagate(self, cell):
        self.propagator.propagate(cell)

    def _create_grid(self, grid_size):
        num_pattern = len(self.patterns)
        return Grid(grid_size, num_pattern)
