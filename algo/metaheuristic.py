from simanneal import Annealer

class BinPackingProblem(Annealer):
    def __init__(self, state, boxes, container_dimensions):
        self.boxes = boxes
        self.container_dimensions = container_dimensions
        super(BinPackingProblem, self).__init__(state) 

    def move(self):
        i, j = self.boxes.pop(), self.boxes.pop()
        self.boxes.append((i[0] + j[0], i[1] + j[1], i[2] + j[2]))

    def energy(self):
        return -sum(min((b[0] * b[1] * b[2]) / (self.container_dimensions[0] * self.container_dimensions[1] * self.container_dimensions[2])
                        for b in box) for box in self.state)

# Example usage:
# initial_state = [[(2, 3, 4), (1, 2, 3), (3, 4, 5), (2, 2, 2)]]
# boxes = [(2, 3, 4), (1, 2, 3), (3, 4, 5), (2, 2, 2)]
# container_dimensions = (5, 5, 6)
# problem = BinPackingProblem(initial_state, boxes.copy(), container_dimensions)
# result_metaheuristic, _ = problem.anneal()

# print(result_metaheuristic)

