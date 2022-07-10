import cv2
import numpy as np

class HwMeanFilter:
    def __init__(self, img_shape, kernel) -> None:
        self.kernel_size = kernel
        self.kernel_dim = kernel**2
        self.width = img_shape[0]
        self.height = img_shape[1]
        #self.depth = img_shape[2]
        self.margin = kernel//2
        self.h_limits = []
        self.buffer = None
        self.buffer_size = 0
        self.move_src = 0
        self.move_dst = 0
        self.set_buffer_size()
        self.index = -1
        self.conv_pixel = 0
        self.rows = []
        self.cols = []
        self.set_start_index()

    # compute first position for conv, based on the kernel and img size
    def set_start_index(self):
        k = self.kernel_size
        w = self.width
        self.conv_pixel = (k + (w-k)) * (k//2) + (k//2)
        self.h_limits = [self.margin, self.width - 1 - self.margin]

    def set_buffer_size(self):
        # compute buffer size
        self.buffer_size = self.width*(self.kernel_size-1) + self.kernel_size + 1
        # create buffer
        self.buffer = np.zeros(shape=self.buffer_size, dtype=np.uint8)
        # get address position of the first element in the last row of buffer
        self.move_src = self.buffer_size - self.kernel_size - 1

    def input_buffer(self, pixel):
        # increment index while filling buffer
        if self.index < self.buffer_size - 1:
            self.index += 1
        # once filled, recycle addresses with new data
        else:
            # replace address space with data and shift left last row
            self.buffer[self.move_dst] = self.buffer[self.move_src]
            self.buffer[self.move_src:-1] = self.buffer[self.move_src+1:]
            self.move_dst += 1
            if self.move_dst >= self.move_src:
                self.move_dst = 0

        self.buffer[self.index] = pixel

    def check_buffer(self):
        # check if buffer is filled
        if self.index == self.buffer_size - 1:
            # check if pixel is valid for convolution
            # if not, wait for buffer to be ready with valid pixel
            col_id = self.conv_pixel%self.width
            if (col_id < self.h_limits[0]) or (col_id > self.h_limits[1]):
                self.conv_pixel += 1
                return False 
            return True
        else:
            return False

    def conv(self):
        sum = 0
        col_id = self.conv_pixel%self.width
        # get k-1 rows to add
        buff_idx = col_id - 1
        for r in range(self.kernel_size - 1):
                sum += self.buffer[buff_idx: buff_idx+self.kernel_size].sum()
                buff_idx += (self.width - self.kernel_size) + self.kernel_size
        # get last row sum
        sum += self.buffer[-(self.kernel_size+1): -1].sum()

        self.conv_pixel += 1
        return sum//self.kernel_dim
