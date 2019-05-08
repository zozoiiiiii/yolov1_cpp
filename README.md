# purpose
  learn the yolo v1 algorithm  

You only look once (YOLO) is a system for detecting objects on the Pascal VOC 2012 dataset. It can detect the 20 Pascal object classes:

    person
    bird, cat, cow, dog, horse, sheep
    aeroplane, bicycle, boat, bus, car, motorbike, train
    bottle, chair, dining table, potted plant, sofa, tv/monitor

# how it works
	input: image resize into 448*448*3
	blackbox: 
		1. consider image into 7*7 grid
		2. every grid has information: 30=20(20 objects classify) + 2(confidence) + 8(2 bounding box position:xywh)
	output: 7*7*30
	


# reference
* https://pjreddie.com/darknet/yolov1/
* AlexeyAB/yolo2_light
* https://zhuanlan.zhihu.com/p/46691043