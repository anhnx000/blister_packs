# python ../yolov5/train.py --img 416 --batch 30 --epochs 150 --data  ../DATN_data/robo_sorted/v0_v10_yolo_crop/data_crop.yaml  --weights yolov5s.pt --device 0,1
# python ../yolov5/train.py --img 416 --batch 30 --epochs 200 --data  ../DATN_data/robo_sorted/v0_v10_yolo_crop/data_crop.yaml  --weights yolov5s.pt --device 0,1
# python ../yolov5/train.py --img 416 --batch 30 --epochs 300 --data  ../DATN_data/robo_sorted/v0_v10_yolo_crop/data_crop.yaml  --weights yolov5s.pt --device 0,1
python ../yolov5/train.py --img 416 --batch 30 --epochs 100 --data  ../DATN_data/roboflow/yolo_crop_v3/data.yaml  --weights yolov5m.pt --device 0,1
python ../yolov5/train.py --img 416 --batch 30 --epochs 150 --data  ../DATN_data/roboflow/yolo_crop_v3/data.yaml  --weights yolov5m.pt --device 0,1
python ../yolov5/train.py --img 416 --batch 30 --epochs 200 --data  ../DATN_data/roboflow/yolo_crop_v3/data.yaml  --weights yolov5m.pt --device 0,1
python ../yolov5/train.py --img 416 --batch 30 --epochs 250 --data  ../DATN_data/roboflow/yolo_crop_v3/data.yaml  --weights yolov5m.pt --device 0,1
python ../yolov5/train.py --img 416 --batch 30 --epochs 300 --data  ../DATN_data/roboflow/yolo_crop_v3/data.yaml  --weights yolov5m.pt --device 0,1
