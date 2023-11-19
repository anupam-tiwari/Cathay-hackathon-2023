from ortools.linear_solver import pywraplp

def exact_bin_packing(boxes, container_dimensions):
    solver = pywraplp.Solver.CreateSolver('SCIP')

    x = {}
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            x[i, j] = solver.BoolVar('x[%i,%i]' % (i, j))

    # Each box is in exactly one bin.
    for i in range(len(boxes)):
        solver.Add(sum(x[i, j] for j in range(len(boxes))) == 1)

    # The size of the boxes should fit within the container.
    for j in range(len(boxes)):
        solver.Add(sum(boxes[i][k] * boxes[i][l] * x[i, j] for i in range(len(boxes)) for k in range(3) for l in range(3)) <=
                   container_dimensions[0] * container_dimensions[1] * container_dimensions[2])

    # Solve the problem.
    status = solver.Solve()

    containers = []
    if status == pywraplp.Solver.OPTIMAL:
        for j in range(len(boxes)):
            container = {'boxes': []}
            for i in range(len(boxes)):
                if x[i, j].solution_value() > 0.5:
                    container['boxes'].append(boxes[i])
            containers.append(container)

    return containers

