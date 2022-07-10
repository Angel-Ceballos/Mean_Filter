import numpy as np
import cv2
from mean_filter import HwMeanFilter
from clk import Clk

class Simulation():
    def __init__(self, raster, kernel, frequency):
        self.clk = Clk(frequency)
        self.src_img = raster
        self.kernel = kernel
        self.mean_filter = None
        self.golden_img = None
        self.output_buffer = None
        self.output_shape = None
        self.set_io_images()

    def set_io_images(self):
        h, w = self.src_img.shape
        m_kernel = (1.0/(self.kernel**2))*np.ones((self.kernel, self.kernel)) # Normalize box filter kernel
        mean_filter_img = cv2.filter2D(self.src_img, -1, m_kernel) # Convolve
        h = np.floor(np.array(m_kernel.shape)/2).astype(int) # Find half dims of kernel
        self.golden_img = mean_filter_img[h[0]:-h[0],h[1]:-h[1]] # Cut away unwanted informatio
        self.output_shape = self.golden_img.shape
        buffer_size = self.output_shape[0] * self.output_shape[1]
        self.output_buffer = np.zeros(shape=buffer_size, dtype=np.uint8)
        self.set_hw_simulation(img_shape=(w, h))
 
    def set_hw_simulation(self, img_shape):
        self.mean_filter = HwMeanFilter(img_shape, self.kernel)

    def run_hw_simulation(self):
        buff_id = 0
        h, w = self.src_img.shape
        for row in range(h):
            for col in range(w):
                # feed buffer
                self.mean_filter.input_buffer(self.src_img[row][col])
                # check if buffer is full and conv is posssible
                if self.mean_filter.check_buffer():
                    # do convolution with median filter
                    self.output_buffer[buff_id] = self.mean_filter.conv()
                    buff_id += 1
                self.clk.update()
        # push left last pixel and do last convolution            
        self.mean_filter.input_buffer(0)
        self.output_buffer[buff_id] = self.mean_filter.conv()
        self.clk.update()

    def show_results(self):
        test_image = np.reshape(self.output_buffer, (self.output_shape[0], self.output_shape[1]))
        print(f"Size of ouput image: {self.output_shape}")
        cv2.namedWindow("Golden Mean Filtered")
        cv2.imshow("Golden Mean Filtered", self.golden_img)
        cv2.namedWindow("HW Mean Filtered")
        cv2.imshow("HW Mean Filtered", test_image)
        golden = self.golden_img.astype('int32')
        test = test_image.astype('int32')
        comparison = (abs(golden - test)/255).sum()/(self.output_shape[0] * self.output_shape[1])
        print(f"Golden vs Implementation Similarity: {round(1 - comparison, 3)*100}%")
        print(f"Latency: {self.mean_filter.buffer_size} clks")
        print(f"Clock cycles for process: {self.clk.get_cycles()} clks")
        cv2.waitKey(0)
        cv2.destroyAllWindows()