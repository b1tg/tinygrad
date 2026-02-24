# Chapter 26: YOLO — Object Detection

YOLO (You Only Look Once) detects objects in images — it tells you *what* is in the image and *where*. This chapter explains how object detection works using tinygrad's YOLOv8 implementation.

## Classification vs Detection

| Task | Output | Example |
|------|--------|---------|
| Classification | One label per image | "This is a dog" |
| Detection | Multiple bounding boxes + labels | "Dog at (100,50,300,400), Cat at (400,200,550,350)" |

Detection outputs a list of `(x1, y1, x2, y2, confidence, class)` tuples.

## YOLOv8 Architecture

YOLOv8 has three parts:

```
Input image (3, 640, 640)
        │
   ┌────┴────┐
   │ Backbone │  Darknet: extract features at multiple scales
   │ (Darknet)│
   └────┬────┘
        │
   ┌────┴────┐
   │  Neck   │  FPN: fuse features across scales
   │  (FPN)  │
   └────┬────┘
        │
   ┌────┴────┐
   │  Head   │  DetectionHead: predict boxes + classes
   └─────────┘
```

### Backbone: Darknet

The backbone extracts features at three scales:

```python
class Darknet:
    def __init__(self, w, r, d):
        # Progressively downsample and increase channels
        self.b1 = [Conv_Block(3, int(64*w), 3, stride=2),
                   Conv_Block(int(64*w), int(128*w), 3, stride=2)]
        self.b2 = [C2f(int(128*w), int(128*w), round(3*d), shortcut=True),
                   Conv_Block(int(128*w), int(256*w), 3, 2),
                   C2f(int(256*w), int(256*w), round(6*d), True)]
        self.b3 = [Conv_Block(int(256*w), int(512*w), 3, stride=2),
                   C2f(int(512*w), int(512*w), round(6*d), True)]
        self.b4 = [Conv_Block(int(512*w), int(512*w*r), 3, stride=2),
                   C2f(int(512*w*r), int(512*w*r), round(3*d), True)]
        self.b5 = [SPPF(int(512*w*r), int(512*w*r), 5)]

    def __call__(self, x):
        x1 = x.sequential(self.b1)
        x2 = x1.sequential(self.b2)  # scale 1: 80x80 features
        x3 = x2.sequential(self.b3)  # scale 2: 40x40 features
        x4 = x3.sequential(self.b4)
        x5 = x4.sequential(self.b5)  # scale 3: 20x20 features
        return (x2, x3, x5)          # three scales
```

**Why multiple scales?** Small objects are detected by high-resolution features (80x80), large objects by low-resolution features (20x20).

### C2f Block

C2f (Cross Stage Partial with 2 convolutions) is a bottleneck block that balances speed and accuracy:

```python
class C2f:
    def __init__(self, c1, c2, n=1, shortcut=False):
        self.c = int(c2 * 0.5)
        self.cv1 = Conv_Block(c1, 2 * self.c, 1)
        self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)
        self.bottleneck = [Bottleneck(self.c, self.c, shortcut) for _ in range(n)]

    def __call__(self, x):
        y = list(self.cv1(x).chunk(2, 1))      # split channels in half
        y.extend(m(y[-1]) for m in self.bottleneck)  # process and accumulate
        return self.cv2(y[0].cat(*y[1:], dim=1))     # concatenate all
```

### SPPF (Spatial Pyramid Pooling - Fast)

SPPF captures context at multiple scales using cascaded max pooling:

```python
class SPPF:
    def __call__(self, x):
        x = self.cv1(x)
        x2 = self.maxpool(x)     # 5x5 receptive field
        x3 = self.maxpool(x2)    # 10x10 receptive field
        x4 = self.maxpool(x3)    # 15x15 receptive field
        return self.cv2(x.cat(x2, x3, x4, dim=1))  # concatenate all
```

### Neck: Feature Pyramid Network (FPN)

The FPN fuses features across scales using top-down and bottom-up paths:

```python
class Yolov8NECK:
    def __call__(self, p3, p4, p5):
        # Top-down: high-level features -> low-level
        x = self.n1(self.up(p5).cat(p4, dim=1))        # upsample p5 + concat with p4
        head_1 = self.n2(self.up(x).cat(p3, dim=1))    # upsample + concat with p3

        # Bottom-up: add back spatial detail
        head_2 = self.n4(self.n3(head_1).cat(x, dim=1))
        head_3 = self.n6(self.n5(head_2).cat(p5, dim=1))
        return [head_1, head_2, head_3]
```

This gives three detection heads operating at 80x80, 40x40, and 20x20 resolution.

### Detection Head

The detection head predicts bounding boxes and class probabilities:

```python
class DetectionHead:
    def __init__(self, nc=80, filters=()):
        self.ch = 16                    # DFL channels
        self.nc = nc                    # 80 COCO classes
        self.dfl = DFL(self.ch)         # Distribution Focal Loss
        self.cv2 = [...]               # box regression branches
        self.cv3 = [...]               # classification branches

    def __call__(self, x):
        for i in range(self.nl):
            x[i] = x[i].sequential(self.cv2[i]).cat(
                   x[i].sequential(self.cv3[i]), dim=1)

        # Decode boxes from anchor points
        box, cls = x_cat[:, :self.ch * 4], x_cat[:, self.ch * 4:]
        dbox = dist2bbox(self.dfl(box), self.anchors, xywh=True) * self.strides
        return dbox.cat(cls.sigmoid(), dim=1)
```

### Anchor-Free Detection

YOLOv8 is **anchor-free** — it doesn't use predefined box shapes. Instead, each grid cell directly predicts the distance from the cell center to the four box edges:

```python
def dist2bbox(distance, anchor_points, xywh=True):
    # distance: left, top, right, bottom from anchor point
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt    # top-left corner
    x2y2 = anchor_points + rb    # bottom-right corner
    return center_xy.cat(wh, dim=1)
```

**DFL (Distribution Focal Loss)** predicts each edge distance as a probability distribution over 16 possible values, then takes a weighted average. This is more accurate than direct regression.

## Non-Maximum Suppression (NMS)

The model predicts hundreds of overlapping boxes. NMS removes duplicates:

```python
def postprocess(output, max_det=300, conf_threshold=0.25, iou_threshold=0.45):
    # 1. Filter low-confidence predictions
    probs = Tensor.where(probs >= conf_threshold, probs, 0)

    # 2. Keep top-K predictions
    boxes = boxes[Tensor.topk(probs, max_det)[1]]

    # 3. Compute IoU (Intersection over Union) between all pairs
    iou = compute_iou_matrix(boxes[:, :4])

    # 4. Remove boxes that overlap too much with a higher-confidence box
    high_iou_mask = (iou > iou_threshold) & same_class_mask
    no_overlap_mask = high_iou_mask.sum(axis=0) == 0
    boxes = boxes * no_overlap_mask.unsqueeze(-1)

    return boxes
```

**IoU (Intersection over Union)** measures how much two boxes overlap:

```
IoU = Area of Overlap / Area of Union

IoU = 0.0 -> no overlap
IoU = 1.0 -> identical boxes
IoU > 0.45 -> "same object", remove the lower-confidence one
```

## YOLOv8 Variants

Different sizes trade speed for accuracy:

```python
# (depth_mult, width_mult, ratio_mult)
variants = {
    'n': (0.33, 0.25, 2.0),  # nano - fastest
    's': (0.33, 0.50, 2.0),  # small
    'm': (0.67, 0.75, 1.5),  # medium
    'l': (1.0,  1.0,  1.0),  # large
    'x': (1.0,  1.25, 1.0),  # extra large - most accurate
}
```

The multipliers control how many layers (depth) and channels (width) each variant has.

## Running YOLOv8

```bash
# Detect objects in an image
python examples/yolov8.py "https://example.com/photo.jpg" n

# With a larger variant for better accuracy
python examples/yolov8.py "path/to/image.jpg" l
```

Output:
```
Objects detected:
- person: 3
- car: 2
- dog: 1
saved detections at ./outputs_yolov8/image_output.png
```

The results are drawn on the image with bounding boxes and labels.

## COCO Classes

YOLOv8 is trained on COCO (Common Objects in Context) with 80 classes:

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird,
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair,
couch, potted plant, bed, dining table, toilet, tv, laptop, mouse,
remote, keyboard, cell phone, microwave, oven, toaster, sink,
refrigerator, book, clock, vase, scissors, teddy bear, hair drier,
toothbrush
```

## Exercises

1. **Run detection**: Detect objects in a photo of your choice with `python examples/yolov8.py "image_url" n`.

2. **Understand multi-scale**: The backbone outputs features at 80x80, 40x40, and 20x20. If the input is 640x640, what is the stride at each scale? What size objects does each scale detect?

3. **IoU by hand**: Two boxes: A = (0, 0, 10, 10) and B = (5, 5, 15, 15). Compute the IoU. Would NMS with `iou_threshold=0.45` keep both?

4. **Compare variants**: Run detection with variant 'n' and 'l'. Compare speed and detection quality.

5. **Read the head**: In `DetectionHead.__call__`, trace how the raw predictions become `(x1, y1, x2, y2, confidence, class)` tuples.

## Source Code Map

| File | What to read |
|------|-------------|
| `examples/yolov8.py` | Full YOLOv8 pipeline (model, preprocessing, postprocessing, visualization) |
| `examples/yolov3.py` | YOLOv3 (older architecture, anchor-based) |
| `extra/models/retinanet.py` | RetinaNet (another one-stage detector) |
| `extra/models/mask_rcnn.py` | Mask R-CNN (two-stage detector with instance segmentation) |
