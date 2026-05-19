from numba import cuda

device = cuda.get_current_device()

print(device.MAX_THREADS_PER_BLOCK)
print(device.WARP_SIZE)
print(device.MAX_BLOCK_DIM_X)
print(device.MAX_BLOCK_DIM_Y)