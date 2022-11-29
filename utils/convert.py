def pascal_to_coco(annotations):
    for i, a in enumerate(annotations):
        annotations[i] = [a[0], a[1], a[2] - a[0], a[3] - a[1]]
    return annotations


def coco_to_pascal(annotations):
    for i, a in enumerate(annotations):
        annotations[i] = [a[0], a[1], a[2] + a[0], a[3] + a[1]]
    return annotations


def circle_to_bbox(circle):
    return [circle[0]-circle[2], circle[1]-circle[2], circle[0]+circle[2], circle[1]+circle[2]]