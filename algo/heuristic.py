def heuristic_bin_packing(boxes, container_dimensions):
    containers = []
    current_container = {'remaining_space': container_dimensions[0] * container_dimensions[1] * container_dimensions[2],
                         'boxes': []}

    for box in boxes:
        if (box[0] <= current_container['remaining_space'] / (container_dimensions[1] * container_dimensions[2]) and
                box[1] <= current_container['remaining_space'] / (container_dimensions[0] * container_dimensions[2]) and
                box[2] <= current_container['remaining_space'] / (container_dimensions[0] * container_dimensions[1])):
            current_container['boxes'].append(box)
            current_container['remaining_space'] -= box[0] * box[1] * box[2]
        else:
            containers.append(current_container)
            current_container = {'remaining_space': container_dimensions[0] * container_dimensions[1] * container_dimensions[2],
                                 'boxes': [box]}

    if current_container['boxes']:
        containers.append(current_container)

    return containers


