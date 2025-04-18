import os
import argparse
from ultralytics import YOLO


class Trainer():
    def __init__(self, args):
        self.root = os.getcwd()
        ### 配置文件 ###
        self.name_yaml = os.path.join(self.root, "crack-seg.yaml")
        ### 训练起始点 ###
        self.name_pretrain = os.path.join(self.root, "yolo11n-seg.pt")
        ### 初始训练模型保存路径 ###
        self.path_train = os.path.join(self.root, "runs/segment/train")
        ### 约束训练模型起始路径与保存路径 ###
        self.name_train = os.path.join(self.root, "runs/segment/train/weights/best.pt")
        self.path_constraint_train = os.path.join(self.root, "runs/segment/constraint")
        ### 剪枝前模型路径与剪枝后保存路径 ###
        self.name_prune_before = os.path.join(self.root, "runs/segment/constraint/weights/best.pt")
        self.name_prune_after = os.path.join(self.root, "runs/segment/constraint/weights/prune.pt")
        ### 剪枝后训练保存路径 ###
        self.path_finetune = os.path.join(self.root, "runs/segment/finetune")

        ### 剪枝率 ###
        self.prune_ratio = args.prune_ratio
        ### 批次大小 ###
        self.batch_sizes = args.batch_sizes # [batch_size 1, batch_size 2, None, batch_size 4]
        ### 训练迭代数 ###
        self.epochs = args.epochs # [Stage 1, Stage 2, None, Stage 4]

    def train(self):
        r"""
        Stage 1: Train the original pretrained model yolo11n-seg.pt
        """
        stage = 1
        model = YOLO(self.name_pretrain)
        model.train(data=self.name_yaml, device="0", imgsz=720, epochs=self.epochs[0], batch=self.batch_sizes[0],
                    workers=0, name=self.path_train)
        # self.val(stage)

    def constraint_train(self):
        r"""
        constraint training, disable automatic mixed precision
        """
        stage = 2
        model = YOLO(self.name_train)
        model.train(data=self.name_yaml, device="0", imgsz=640, epochs=self.epochs[1], batch=self.batch_sizes[0], 
                    amp=False, workers=0, name=self.path_constraint_train)
        # self.val(stage)

    def prune(self):
        r"""
        prune the model for memory compression
        """
        stage = 3
        from utils.yolo.seg_pruning import do_pruning
        do_pruning(self.name_prune_before, self.name_prune_after, self.name_yaml, self.prune_ratio)
        # self.val(stage)

    def finetune(self):
        r"""
        finetune the model to retrieve accuracy
        """
        stage = 4
        model = YOLO(self.name_prune_after)
        for param in model.parameters():
            param.requires_grad = True
        model.train(data=self.name_yaml, device="0", imgsz=640, epochs=self.epochs[3], batch=self.batch_sizes[3], 
                    workers=0, name=self.path_finetune, prune=True)
        # self.val(stage)

    def train_pipeline(self):
        self.train()
        self.constraint_train()
        self.prune()
        self.finetune()

    def val(self, stage):
        if stage == 1:
            model = YOLO(self.name_train)
        elif stage == 2:
            model = YOLO(self.name_prune_before)
        elif stage == 3:
            model = YOLO(self.name_prune_after)
        else:
            model = YOLO(os.path.join(self.path_finetune, 'weights/best.pt'))

        total_parameters = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_parameters}")

        # Validate the model
        metrics = model.val(data=self.name_yaml, device="0", batch=1, workers=0)


def get_args():
    parser = argparse.ArgumentParser(description='argument for training&pruning yolov11')
    parser.add_argument('--prune_ratio', type=float, default=0.5)
    parser.add_argument('--epochs', type=lambda x: [int(i) if i.isdigit() else None for i in x.split(',')], default=[50, 50, None, 100])
    parser.add_argument('--batch_sizes', type=lambda x: [int(i) if i.isdigit() else None for i in x.split(',')], default=[64, 64, None, 64])
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    trainer = Trainer(args)
    trainer.train_pipeline()


if __name__ == '__main__':
    main()