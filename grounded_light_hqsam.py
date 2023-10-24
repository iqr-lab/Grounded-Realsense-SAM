import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from segment_anything.modeling import (
    MaskDecoderHQ,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)

from grounded_segment_anything.EfficientSAM.LightHQSAM.tiny_vit_sam import TinyViT


def setup_model():
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
        image_encoder=TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=160,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    return mobile_sam


class GroundedLightHQSAM:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "grounded_segment_anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = (
        "grounded_segment_anything/groundingdino_swint_ogc.pth"
    )

    # Building GroundingDINO inference model
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
    )

    # Building MobileSAM predictor
    HQSAM_CHECKPOINT_PATH = "grounded_segment_anything/EfficientSAM/sam_hq_vit_tiny.pth"
    checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
    light_hqsam = setup_model()

    def __init__(self):
        self.light_hqsam.load_state_dict(self.checkpoint, strict=True)
        self.light_hqsam.to(device=self.DEVICE)

        self.sam_predictor = SamPredictor(self.light_hqsam)

    # Predict classes and hyper-param for GroundingDINO
    def predict(
        self, image, caption, box_threshold=0.25, nms_threshold=0.8, visualization=True
    ):
        detections, labels = self.grounding_dino_model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=box_threshold,
        )

        # NMS post process
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                nms_threshold,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        labels = [labels[i] for i in nms_idx]

        # print(f"After NMS: {len(detections.xyxy)} boxes")

        # annotate image with detections
        if visualization:
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(
                scene=image.copy(), detections=detections, labels=labels
            )

        # Prompting SAM with detected boxes
        def segment(
            sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
        ) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, _ = sam_predictor.predict(
                    box=box,
                    multimask_output=False,
                    hq_token_only=True,
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        # annotate image with masks
        if visualization:
            mask_annotator = sv.MaskAnnotator()
            annotated_image = mask_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        return annotated_image if visualization else detections
