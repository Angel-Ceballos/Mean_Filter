class Clk:
    def __init__(self, frequency) -> None:
        self.frequency = frequency
        self.time = 0
        self.clk_cycles = 0

    def update(self):
        self.time += 1/self.frequency
        self.clk_cycles += 1

    def get_time(self):
        return self.time

    def get_cycles(self):
        return self.clk_cycles