"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)
from pointcept.datasets import build_dataset, collate_fn, point_collate_fn

TESTERS = Registry("testers")

class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,  # Changed to batch size 2
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,  # Using point_collate_fn
        )
        return test_loader

    def test(self):
        raise NotImplementedError

@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json
            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(os.path.join(save_path, "submit", "test", "submission.json"), "w") as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        
        record = {}
        for idx, data_dict_batch in enumerate(self.test_loader):
            end = time.time()
            for b_idx, data_dict in enumerate(data_dict_batch):
                fragment_list = data_dict.pop("fragment_list")
                segment = data_dict.pop("segment")
                data_name = data_dict.pop("name")
                pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
                if os.path.isfile(pred_save_path):
                    logger.info(
                        "{}/{}: {}, loaded pred and label.".format(
                            idx + 1, len(self.test_loader), data_name
                        )
                    )
                    continue
                else:
                    pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                    for i in range(len(fragment_list)):
                        fragment_batch_size = 1
                        s_i, e_i = i * fragment_batch_size, min(
                            (i + 1) * fragment_batch_size, len(fragment_list)
                        )
                        input_dict = collate_fn(fragment_list[s_i:e_i])
                        for key in input_dict.keys():
                            if isinstance(input_dict[key], torch.Tensor):
                                input_dict[key] = input_dict[key].cuda(non_blocking=True)
                        with torch.no_grad():
                            pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                            pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                        logger.info(
                            "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                                idx + 1,
                                len(self.test_loader),
                                data_name=data_name,
                                batch_idx=i,
                                batch_num=len(fragment_list),
                            )
                        )

                    probs = pred.max(1)[0].data.cpu().numpy()
                    pred = pred.max(1)[1].data.cpu().numpy()
                    np.save(pred_save_path, pred)

                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    probs = probs[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
                intersection, union, target = intersection_and_union(
                    pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
                )
                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)
                record[data_name] = dict(
                    intersection=intersection, union=union, target=target
                )

                mask = union != 0
                iou_class = intersection / (union + 1e-10)
                iou = np.mean(iou_class[mask])
                acc = sum(intersection) / (sum(target) + 1e-10)

                m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
                m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

                batch_time.update(time.time() - end)
                logger.info(
                    "Test: {} [{}/{}]-{} "
                    "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    "Accuracy {acc:.4f} ({m_acc:.4f}) "
                    "mIoU {iou:.4f} ({m_iou:.4f})".format(
                        data_name,
                        idx + 1,
                        len(self.test_loader),
                        segment.size,
                        batch_time=batch_time,
                        acc=acc,
                        m_acc=m_acc,
                        iou=iou,
                        m_iou=m_iou,
                    )
                )
                if (
                    self.cfg.data.test.type == "ScanNetDataset"
                    or self.cfg.data.test.type == "ScanNet200Dataset"
                ):
                    np.savetxt(
                        os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                        self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                        fmt="%d",
                    )
                elif self.cfg.data.test.type == "SemanticKITTIDataset":
                    sequence_name, frame_name = data_name.split("_")
                    os.makedirs(
                        os.path.join(
                            save_path, "submit", "sequences", sequence_name, "predictions"
                        ),
                        exist_ok=True,
                    )
                    pred = pred.astype(np.uint32)
                    pred = np.vectorize(
                        self.test_loader.dataset.learning_map_inv.__getitem__
                    )(pred).astype(np.uint32)
                    pred.tofile(
                        os.path.join(
                            save_path,
                            "submit",
                            "sequences",
                            sequence_name,
                            "predictions",
                            f"{frame_name}.label",
                        )
                    )
                    os.makedirs(
                        os.path.join(
                            save_path, "submit", "sequences", sequence_name, "probability"
                        ),
                        exist_ok=True,
                    )
                    prob_save_path = os.path.join(save_path, "submit", "sequences", sequence_name, "probability", "{}_prob.npy".format(data_name))
                    np.save(prob_save_path, probs)

                elif self.cfg.data.test.type == "NuScenesDataset":
                    np.array(pred + 1).astype(np.uint8).tofile(
                        os.path.join(
                            save_path,
                            "submit",
                            "lidarseg",
                            "test",
                            "{}_lidarseg.bin".format(data_name),
                        )
                    )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return point_collate_fn(batch)