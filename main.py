import cv2
from simulation import Simulation

if __name__ == "__main__":

    # load grayscale img
    src_img = cv2.imread("dog.jpg", 0)

    # initialize and run hw simulation
    hw_mean_filter_sim = Simulation(raster=src_img, kernel=5, frequency=1e9)
    hw_mean_filter_sim.run_hw_simulation()
    hw_mean_filter_sim.show_results()