from numpy import array, ceil, floor, ones


def limited_cuda_args(matrix, dimensions=2, number_of_kernels=1024):
    """
    Define the cuda args based on parameters.
    Arguments:
        matrix(object): array to be used as parallel process base.
        dimensions(int): number of dimensions to be considered.
        number_of_kernels(int): number of kernels supported by GPU. 
    Return:
        tuple: cuda grid (system composed by multiple blocks) dimensions.
        tuple: cuda block (parallel kernels) dimensions.
    """
    matrix_dimensions = array(matrix.shape)[:dimensions]
    return tuple(matrix_dimensions), (1, 1, 1)


def cuda_args(matrix, dimensions=2, number_of_kernels=1024):
    """
    Define the cuda args based on parameters.
    Arguments:
        matrix(object): array to be used as parallel process base.
        dimensions(int): number of dimensions to be considered.
        number_of_kernels(int): number of kernels supported by GPU. 
    Return:
        tuple: cuda grid (system composed by multiple blocks) dimensions.
        tuple: cuda block (parallel kernels) dimensions.
    """
    block_dimension = int(floor((number_of_kernels+1)**(1 / dimensions)))
    matrix_dimensions = array(matrix.shape)[:dimensions]
    block = tuple((block_dimension * ones(dimensions)).astype(int))
    grid = tuple(ceil(matrix_dimensions / block_dimension).astype(int))
    return grid, block
