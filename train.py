import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\UPAD-YOLO\ultralytics\cfg\models\11\yolov11-UDMConv-PPA_ASFF.yaml')

    model.train(
        # 路径配置
        data=r'D:\JZX-DATA\data.yaml',
        project='runs/train',
        name='Container defects-yolo',

        # 训练参数
        epochs=300,
        patience=100,  # 早停耐心值
        batch=8,
        imgsz=640,
        device='0',
        workers=4,

        # 优化器参数
        lr0=0.001,  # 初始学习率
        lrf=0.01,  # 最终学习率
        weight_decay=0.05,  # 权重衰减系数
        warmup_epochs=3.0,
        cos_lr=True,  # 余弦退火调度

        # 正则化与增强
        dropout=0.0,  # 无Dropout
        label_smoothing=0.1,
        mixup=0.1,  # 降低Mixup强度
        copy_paste=0.2,
        auto_augment='randaugment',

        # 系统设置
        amp=True,  # 自动混合精度
        cache='disk',
        close_mosaic=50,  # 最后50epoch关闭马赛克增强
        pretrained=False,  # 不使用预训练权重
        exist_ok=True,  # 允许覆盖现有目录
        val=True,  # 开启验证
        save_period=-1  # 仅保存最终模型
    )