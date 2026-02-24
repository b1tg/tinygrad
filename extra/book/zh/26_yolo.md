# 第26章：YOLO — 目标检测

YOLO（You Only Look Once）用于检测图像中的目标——它能告诉你图像中*有什么*以及*在哪里*。本章使用 tinygrad 的 YOLOv8 实现来讲解目标检测的工作原理。

## 分类 vs 检测

| 任务 | 输出 | 示例 |
|------|------|------|
| 分类 | 每张图像一个标签 | "这是一只狗" |
| 检测 | 多个边界框 + 标签 | "狗在 (100,50,300,400)，猫在 (400,200,550,350)" |

检测输出一组 `(x1, y1, x2, y2, confidence, class)` 元组。

## YOLOv8 架构

YOLOv8 由三部分组成：

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

### 主干网络：Darknet

主干网络在三个尺度上提取特征：

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

**为什么需要多尺度？** 小目标由高分辨率特征（80x80）检测，大目标由低分辨率特征（20x20）检测。

### C2f 模块

C2f（带两个卷积的跨阶段部分连接）是一种在速度和精度之间取得平衡的瓶颈模块：

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

### SPPF（快速空间金字塔池化）

SPPF 通过级联最大池化在多个尺度上捕获上下文信息：

```python
class SPPF:
    def __call__(self, x):
        x = self.cv1(x)
        x2 = self.maxpool(x)     # 5x5 receptive field
        x3 = self.maxpool(x2)    # 10x10 receptive field
        x4 = self.maxpool(x3)    # 15x15 receptive field
        return self.cv2(x.cat(x2, x3, x4, dim=1))  # concatenate all
```

### 颈部网络：特征金字塔网络（FPN）

FPN 通过自顶向下和自底向上的路径融合不同尺度的特征：

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

这产生了三个检测头，分别在 80x80、40x40 和 20x20 分辨率上工作。

### 检测头

检测头预测边界框和类别概率：

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

### 无锚框检测

YOLOv8 是**无锚框**的——它不使用预定义的框形状。相反，每个网格单元直接预测从单元中心到四条框边的距离：

```python
def dist2bbox(distance, anchor_points, xywh=True):
    # distance: left, top, right, bottom from anchor point
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt    # top-left corner
    x2y2 = anchor_points + rb    # bottom-right corner
    return center_xy.cat(wh, dim=1)
```

**DFL（Distribution Focal Loss）** 将每条边的距离预测为 16 个可能值上的概率分布，然后取加权平均。这比直接回归更精确。

## 非极大值抑制（NMS）

模型会预测数百个重叠的框。NMS 用于去除重复：

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

**IoU（交并比）** 衡量两个框的重叠程度：

```
IoU = Area of Overlap / Area of Union

IoU = 0.0 -> no overlap
IoU = 1.0 -> identical boxes
IoU > 0.45 -> "same object", remove the lower-confidence one
```

## YOLOv8 变体

不同大小的变体在速度和精度之间权衡：

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

这些乘数控制每个变体的层数（深度）和通道数（宽度）。

## 运行 YOLOv8

```bash
# Detect objects in an image
python examples/yolov8.py "https://example.com/photo.jpg" n

# With a larger variant for better accuracy
python examples/yolov8.py "path/to/image.jpg" l
```

输出：
```
Objects detected:
- person: 3
- car: 2
- dog: 1
saved detections at ./outputs_yolov8/image_output.png
```

检测结果会绘制在图像上，包含边界框和标签。

## COCO 类别

YOLOv8 在 COCO（Common Objects in Context）数据集上训练，包含 80 个类别：

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

## 练习

1. **运行检测**：使用 `python examples/yolov8.py "image_url" n` 对你选择的照片进行目标检测。

2. **理解多尺度**：主干网络输出 80x80、40x40 和 20x20 的特征。如果输入为 640x640，每个尺度的步幅是多少？每个尺度检测什么大小的目标？

3. **手算 IoU**：两个框：A = (0, 0, 10, 10) 和 B = (5, 5, 15, 15)。计算 IoU。使用 `iou_threshold=0.45` 的 NMS 会保留两个框吗？

4. **比较变体**：分别使用变体 'n' 和 'l' 运行检测。比较速度和检测质量。

5. **阅读检测头**：在 `DetectionHead.__call__` 中，追踪原始预测如何变为 `(x1, y1, x2, y2, confidence, class)` 元组。

## 源代码索引

| 文件 | 阅读内容 |
|------|----------|
| `examples/yolov8.py` | 完整的 YOLOv8 流程（模型、预处理、后处理、可视化） |
| `examples/yolov3.py` | YOLOv3（较早的架构，基于锚框） |
| `extra/models/retinanet.py` | RetinaNet（另一种单阶段检测器） |
| `extra/models/mask_rcnn.py` | Mask R-CNN（带实例分割的两阶段检测器） |
