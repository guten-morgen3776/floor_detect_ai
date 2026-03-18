"""
YOLO セグメンテーション推論スクリプト（カスタムTTA + WBF版）
- 建築図面からの部屋インスタンスセグメンテーションを想定
- 4方向回転のTTA → 逆変換 → WBFでボックス統合 → マスク加重平均で統合
- 依存: ultralytics, opencv-python, numpy, ensemble-boxes
  pip install ensemble-boxes
"""

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# WBF用（座標正規化・統合）
try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    raise ImportError("ensemble-boxes が必要です: pip install ensemble-boxes")

# =============================================================================
# CONFIG: パス・パラメータ
# =============================================================================
CONFIG = {
    "weights": "runs/segment/train/weights/best.pt",
    "source": "path/to/your/image_or_dir",
    "project": "runs/segment-predict",
    "name": "predict_tta_wbf",
    "kaggle_working": "/kaggle/working",
    # 推論
    "conf": 0.25,
    "imgsz": 640,
    "device": None,
    # WBF
    "iou_thr": 0.5,
    "skip_box_thr": 0.0001,
    # マスク統合
    "mask_fusion_thr": 0.5,   # 加重平均マスクの二値化閾値
    "mask_iou_thr": 0.3,      # WBFボックスに「紐付く」元予測のIoU閾値
    "use_weighted_mask": True,  # True=スコアで加重平均, False=単純平均
    # 可視化
    "draw_boxes": True,  # 最終画像にバウンディングボックスを描画するか（False=マスクのみ）
}


# -----------------------------------------------------------------------------
# 1. カスタムTTA: 画像変換（回転のみ、Flipは使わない）
# -----------------------------------------------------------------------------

# 回転の種類: 0=そのまま, 1=90°CW, 2=180°, 3=270°CW
ROTATE_0 = 0
ROTATE_90_CW = 1
ROTATE_180 = 2
ROTATE_270_CW = 3


def build_tta_images(img: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    """
    入力画像に対してTTA用の4パターン（0°, 90°, 180°, 270°）の画像リストを作成する。
    OpenCVのcv2.rotateを用いて正確に回転する（図面タスクのため回転を重視）。

    Args:
        img: BGR画像 (H, W, C)

    Returns:
        [(image, rotation_id), ...]  length=4
        rotation_id: 0, 1, 2, 3
    """
    out = []
    # 0度（オリジナル）
    out.append((img.copy(), ROTATE_0))
    # 90度時計回り
    out.append((cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), ROTATE_90_CW))
    # 180度
    out.append((cv2.rotate(img, cv2.ROTATE_180), ROTATE_180))
    # 270度時計回り（= 90度反時計回り）
    out.append((cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), ROTATE_270_CW))
    return out


def get_rotated_shape(orig_h: int, orig_w: int, rotation_id: int) -> Tuple[int, int]:
    """回転後の画像サイズ (height, width) を返す。90/270では縦横が入れ替わる。"""
    if rotation_id in (ROTATE_0, ROTATE_180):
        return (orig_h, orig_w)
    return (orig_w, orig_h)


# -----------------------------------------------------------------------------
# 2. 逆変換: 回転画像上の座標・マスクを元画像の座標系に戻す
# -----------------------------------------------------------------------------

def inverse_transform_box(
    x1: float, y1: float, x2: float, y2: float,
    rotation_id: int,
    orig_h: int, orig_w: int,
    rotated_h: int, rotated_w: int,
) -> Tuple[float, float, float, float]:
    """
    回転した画像上でのボックス (x1,y1,x2,y2) を、
    元画像の座標系におけるボックスに変換する。
    4頂点を逆回転してから、そのAABB（軸並行バウンディングボックス）を返す。

    Args:
        x1,y1,x2,y2: 回転画像上のxyxy（絶対座標）
        rotation_id: 0,1,2,3
        orig_h, orig_w: 元画像の高さ・幅
        rotated_h, rotated_w: 回転画像の高さ・幅

    Returns:
        (x1_o, y1_o, x2_o, y2_o) 元画像座標でのxyxy
    """
    # 4頂点を逆回転して元座標に
    points = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ], dtype=np.float64)

    if rotation_id == ROTATE_0:
        x1_o = float(np.clip(points[:, 0].min(), 0, orig_w))
        y1_o = float(np.clip(points[:, 1].min(), 0, orig_h))
        x2_o = float(np.clip(points[:, 0].max(), 0, orig_w))
        y2_o = float(np.clip(points[:, 1].max(), 0, orig_h))
        return (x1_o, y1_o, x2_o, y2_o)

    # 回転画像上の点 (x_r, y_r) -> 元画像上の点 (x_o, y_o)
    # 90°CW: 元 (x_o,y_o) -> 回転後 (orig_h-1-y_o, x_o)。逆: (x_r,y_r)->(y_r, orig_h-1-x_r)
    # 270°CW(90°CCW): 元 -> 回転後 (orig_w-1-y_o, x_o) の逆: (x_r,y_r)->(y_r, orig_w-1-x_r)
    def rot_inv(x_r: float, y_r: float) -> Tuple[float, float]:
        if rotation_id == ROTATE_90_CW:
            x_o = y_r
            y_o = orig_h - 1 - x_r
        elif rotation_id == ROTATE_180:
            x_o = rotated_w - 1 - x_r
            y_o = rotated_h - 1 - y_r
        elif rotation_id == ROTATE_270_CW:
            x_o = y_r
            y_o = orig_w - 1 - x_r
        else:
            x_o, y_o = x_r, y_r
        return (x_o, y_o)

    transformed = np.array([rot_inv(p[0], p[1]) for p in points])
    x1_o = float(np.clip(transformed[:, 0].min(), 0, orig_w))
    y1_o = float(np.clip(transformed[:, 1].min(), 0, orig_h))
    x2_o = float(np.clip(transformed[:, 0].max(), 0, orig_w))
    y2_o = float(np.clip(transformed[:, 1].max(), 0, orig_h))
    return (x1_o, y1_o, x2_o, y2_o)


def inverse_transform_mask(
    mask: np.ndarray,
    rotation_id: int,
) -> np.ndarray:
    """
    回転した画像上で得られたマスク（2D配列）を、
    元の向きに逆回転して返す。cv2.rotateで逆回転のみ行う。

    Args:
        mask: (H_rot, W_rot) のマスク
        rotation_id: 適用した回転のID

    Returns:
        元画像と同じ向きのマスク（サイズは rotation_id により元と異なる場合あり）。
        90/270のとき縦横が入れ替わるので、戻すと元の (orig_h, orig_w) になる。
    """
    if rotation_id == ROTATE_0:
        return mask.copy()
    if rotation_id == ROTATE_90_CW:
        # 逆 = 90°CCW
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation_id == ROTATE_180:
        return cv2.rotate(mask, cv2.ROTATE_180)
    if rotation_id == ROTATE_270_CW:
        # 逆 = 90°CW
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    return mask.copy()


def inverse_transform_masks_to_original_shape(
    masks: List[np.ndarray],
    rotation_ids: List[int],
    orig_h: int,
    orig_w: int,
) -> List[np.ndarray]:
    """
    複数マスクを逆回転し、さらに元画像サイズ (orig_h, orig_w) にリサイズして
    そろえて返す。逆回転後のサイズが orig と一致するので、90/270の場合は
    逆回転するだけで (orig_h, orig_w) になる。
    """
    out = []
    for m, rid in zip(masks, rotation_ids):
        inv = inverse_transform_mask(m, rid)
        if inv.shape[0] != orig_h or inv.shape[1] != orig_w:
            inv = cv2.resize(
                inv.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
        out.append(inv)
    return out


# -----------------------------------------------------------------------------
# 3. 推論実行と結果の取得・逆変換まで一括
# -----------------------------------------------------------------------------

def run_inference_per_image(
    model: YOLO,
    img: np.ndarray,
    conf: float,
    imgsz: int,
    device: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """
    1枚の画像に対してYOLO推論を行い、augment=Falseで結果を取得する。
    返す座標・マスクは「この画像の座標系」のまま（逆変換は呼び出し側で行う）。

    Returns:
        boxes_xyxy: (N, 4) この画像上の絶対座標 xyxy
        scores: (N,)
        cls: (N,) クラスID
        masks: list of (H, W) 各検出のマスク（二値または確率）
    """
    results = model.predict(
        img,
        conf=conf,
        imgsz=imgsz,
        device=device,
        augment=False,
        verbose=False,
    )
    if not results or len(results) == 0:
        return None, None, None, None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None, None, None

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)

    masks = None
    if r.masks is not None and r.masks.data is not None:
        # (N, H, W) のテンソル。H,Wはこの画像のサイズに合わせてある
        masks = [
            r.masks.data[i].cpu().numpy()
            for i in range(len(r.boxes))
        ]
    return boxes_xyxy, scores, cls, masks


def collect_tta_predictions(
    model: YOLO,
    tta_list: List[Tuple[np.ndarray, int]],
    orig_h: int,
    orig_w: int,
    conf: float,
    imgsz: int,
    device: Optional[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[List[np.ndarray]]]:
    """
    各TTA画像で推論し、ボックス・スコア・クラス・マスクを取得したうえで、
    ボックスとマスクをすべて「元画像の座標系・サイズ」に逆変換して返す。
    返り値は「全TTAの全検出」をフラットにしたリストではなく、
    TTAごとのリスト（boxes_list[i] = i番目のTTAの (N_i, 4) など）とする。
    WBFの入力形式に合わせる。

    Returns:
        boxes_list: 各要素 (N_i, 4) xyxy 元画像座標
        scores_list: 各要素 (N_i,)
        labels_list: 各要素 (N_i,)
        masks_list: 各要素 list of (orig_h, orig_w) のマスク（逆変換済み）
    """
    boxes_list = []
    scores_list = []
    labels_list = []
    masks_list = []

    for img_rot, rot_id in tta_list:
        rh, rw = get_rotated_shape(orig_h, orig_w, rot_id)
        boxes_xyxy, scores, cls, masks = run_inference_per_image(
            model, img_rot, conf, imgsz, device
        )
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            boxes_list.append(np.zeros((0, 4), dtype=np.float64))
            scores_list.append(np.array([], dtype=np.float64))
            labels_list.append(np.array([], dtype=np.int32))
            masks_list.append([])
            continue

        # ボックスを元座標に逆変換
        inv_boxes = []
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            inv_xyxy = inverse_transform_box(
                x1, y1, x2, y2, rot_id, orig_h, orig_w, rh, rw
            )
            inv_boxes.append(inv_xyxy)
        inv_boxes = np.array(inv_boxes, dtype=np.float64)
        boxes_list.append(inv_boxes)
        scores_list.append(scores.astype(np.float64))
        labels_list.append(cls.astype(np.int32))

        # マスクを逆変換し、元画像サイズにそろえる
        if masks:
            # 各マスクは回転画像サイズ (rh, rw)。逆回転して (orig_h, orig_w) に
            inv_masks = inverse_transform_masks_to_original_shape(
                masks, [rot_id] * len(masks), orig_h, orig_w
            )
            masks_list.append(inv_masks)
        else:
            masks_list.append([])

    return boxes_list, scores_list, labels_list, masks_list


# -----------------------------------------------------------------------------
# 4. WBF（Weighted Boxes Fusion）でボックス統合
# -----------------------------------------------------------------------------

def normalize_boxes_01(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """xyxy 絶対座標を [0,1] に正規化。WBFは [0,1] を要求する。"""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy.copy()
    b = boxes_xyxy.copy()
    b[:, [0, 2]] /= width
    b[:, [1, 3]] /= height
    return np.clip(b, 0.0, 1.0)


def denormalize_boxes(boxes_01: np.ndarray, width: int, height: int) -> np.ndarray:
    """[0,1] のボックスを絶対座標に戻す。"""
    if len(boxes_01) == 0:
        return boxes_01.copy()
    b = boxes_01.copy()
    b[:, [0, 2]] *= width
    b[:, [1, 3]] *= height
    return b


def run_wbf(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    width: int,
    height: int,
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.0001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Boxes Fusion により、複数TTAのボックス・スコア・ラベルを統合する。
    座標はいったん [0,1] に正規化してからWBFに渡し、結果を絶対座標に戻す。

    Returns:
        fused_boxes: (M, 4) xyxy 絶対座標
        fused_scores: (M,)
        fused_labels: (M,)
    """
    # 正規化 [0,1]。WBFはリストの各要素が [[x1,y1,x2,y2], ...] の形式を要求
    norm_boxes_list = [
        normalize_boxes_01(b, width, height) for b in boxes_list
    ]
    boxes_input = [b.tolist() for b in norm_boxes_list]
    scores_input = [s.tolist() for s in scores_list]
    labels_input = [l.tolist() for l in labels_list]

    boxes_01, scores, labels = weighted_boxes_fusion(
        boxes_input,
        scores_input,
        labels_input,
        weights=None,  # 均等重み（必要なら [1,1,1,1] など指定可）
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    boxes_01 = np.array(boxes_01, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.int32)
    fused_boxes = denormalize_boxes(boxes_01, width, height)
    return fused_boxes, scores, labels


# -----------------------------------------------------------------------------
# 5. マスクの加重平均と統合（WBFボックスに紐付くマスクを融合）
# -----------------------------------------------------------------------------

def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """2つのボックス (x1,y1,x2,y2) のIoUを返す。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def find_contributing_predictions(
    fused_box: np.ndarray,
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    iou_thr: float,
) -> List[Tuple[int, int, float]]:
    """
    WBFで得られた1つの融合ボックス fused_box について、
    どの「元の予測」(tta_id, det_id) がこの融合を構成したか判定する。
    IoU >= iou_thr のものを返す。

    Returns:
        [(tta_id, det_id, score), ...]
    """
    contributors = []
    for tta_id, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for det_id in range(len(boxes)):
            iou = box_iou(fused_box, boxes[det_id])
            if iou >= iou_thr:
                contributors.append((tta_id, det_id, float(scores[det_id])))
    return contributors


def fuse_masks_for_box(
    contributors: List[Tuple[int, int, float]],
    masks_list: List[List[np.ndarray]],
    orig_h: int,
    orig_w: int,
    use_weighted: bool,
    thr: float,
) -> Optional[np.ndarray]:
    """
    1つのWBFボックスに紐付く複数マスクを、ピクセルごとに加重平均（または単純平均）し、
    閾値 thr で二値化したマスクを返す。
    マスクはすでに元画像サイズ (orig_h, orig_w) にそろえてある。
    """
    if not contributors:
        return None
    # 同じサイズのマスクを集める
    acc = np.zeros((orig_h, orig_w), dtype=np.float64)
    w_sum = 0.0
    for tta_id, det_id, score in contributors:
        if tta_id >= len(masks_list) or det_id >= len(masks_list[tta_id]):
            continue
        m = masks_list[tta_id][det_id]
        if m.shape[0] != orig_h or m.shape[1] != orig_w:
            m = cv2.resize(
                m.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
        m = m.astype(np.float64)
        # 確率テンソルの場合はそのまま、二値の場合は 0/1
        if use_weighted:
            acc += m * score
            w_sum += score
        else:
            acc += m
            w_sum += 1.0
    if w_sum <= 0:
        return None
    acc /= w_sum
    binary = (acc >= thr).astype(np.uint8)
    return binary


def fuse_all_masks(
    fused_boxes: np.ndarray,
    fused_scores: np.ndarray,
    fused_labels: np.ndarray,
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    masks_list: List[List[np.ndarray]],
    orig_h: int,
    orig_w: int,
    mask_iou_thr: float,
    mask_fusion_thr: float,
    use_weighted_mask: bool,
) -> List[Optional[np.ndarray]]:
    """
    ［最重要］各WBF融合ボックスについて、紐付く元予測のマスクを加重平均（または単純平均）し、
    閾値で二値化したマスクのリストを返す。
    """
    out_masks = []
    for i in range(len(fused_boxes)):
        contrib = find_contributing_predictions(
            fused_boxes[i],
            boxes_list,
            scores_list,
            labels_list,
            mask_iou_thr,
        )
        mask = fuse_masks_for_box(
            contrib,
            masks_list,
            orig_h,
            orig_w,
            use_weighted=use_weighted_mask,
            thr=mask_fusion_thr,
        )
        out_masks.append(mask)
    return out_masks


# -----------------------------------------------------------------------------
# 6. 可視化と保存
# -----------------------------------------------------------------------------

def draw_segmentation_on_image(
    img: np.ndarray,
    fused_boxes: np.ndarray,
    fused_scores: np.ndarray,
    fused_labels: np.ndarray,
    fused_masks: List[Optional[np.ndarray]],
    class_names: Optional[dict] = None,
    draw_boxes: bool = True,
) -> np.ndarray:
    """
    元画像に、統合されたボックス・マスクを重畳描画する。
    マスクは半透明の色で塗る。draw_boxes=True のときのみ矩形とラベルを描画する。
    """
    out = img.copy()
    h, w = out.shape[:2]
    np.random.seed(42)
    for i in range(len(fused_boxes)):
        x1, y1, x2, y2 = fused_boxes[i].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        color = tuple(int(c) for c in np.random.randint(50, 255, 3))
        if fused_masks[i] is not None:
            overlay = out.copy()
            mask = fused_masks[i]
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            overlay[mask > 0] = color
            cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
        if draw_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = class_names.get(int(fused_labels[i]), f"cls_{fused_labels[i]}") if class_names else str(int(fused_labels[i]))
            cv2.putText(
                out,
                f"{label} {fused_scores[i]:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
    return out


def process_one_image(
    model: YOLO,
    img: np.ndarray,
    save_path: Optional[Path],
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Optional[np.ndarray]]]:
    """
    1枚の画像に対して TTA → 推論 → 逆変換 → WBF → マスク融合 まで行い、
    可視化画像を保存する。統合された boxes, scores, labels, masks も返す。
    """
    orig_h, orig_w = img.shape[:2]

    # 1. TTA画像リスト
    tta_list = build_tta_images(img)

    # 2. 各画像で推論し、逆変換まで実施
    boxes_list, scores_list, labels_list, masks_list = collect_tta_predictions(
        model,
        tta_list,
        orig_h,
        orig_w,
        conf=config["conf"],
        imgsz=config["imgsz"],
        device=config["device"],
    )

    # 3. WBFでボックス統合
    fused_boxes, fused_scores, fused_labels = run_wbf(
        boxes_list,
        scores_list,
        labels_list,
        width=orig_w,
        height=orig_h,
        iou_thr=config["iou_thr"],
        skip_box_thr=config["skip_box_thr"],
    )

    # 4. マスクの加重平均と統合
    fused_masks = fuse_all_masks(
        fused_boxes,
        fused_scores,
        fused_labels,
        boxes_list,
        scores_list,
        labels_list,
        masks_list,
        orig_h,
        orig_w,
        mask_iou_thr=config["mask_iou_thr"],
        mask_fusion_thr=config["mask_fusion_thr"],
        use_weighted_mask=config["use_weighted_mask"],
    )

    # 5. 可視化して保存
    vis = draw_segmentation_on_image(
        img,
        fused_boxes,
        fused_scores,
        fused_labels,
        fused_masks,
        class_names=getattr(model, "names", None),
        draw_boxes=config.get("draw_boxes", True),
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis)

    return fused_boxes, fused_scores, fused_labels, fused_masks


# -----------------------------------------------------------------------------
# パス解決（元スクリプトと同様）
# -----------------------------------------------------------------------------

def is_kaggle_env() -> bool:
    return Path(CONFIG["kaggle_working"]).is_dir()


def resolve_project_dir() -> str:
    project = Path(CONFIG["project"])
    if is_kaggle_env():
        base = Path(CONFIG["kaggle_working"])
        project = base / project
    project.mkdir(parents=True, exist_ok=True)
    return str(project)


def resolve_weights_path() -> str:
    w = Path(CONFIG["weights"])
    if not w.is_absolute():
        script_dir = Path(__file__).resolve().parent
        w = script_dir.parent / w
    if not w.exists():
        raise FileNotFoundError(f"重みファイルが見つかりません: {w}")
    return str(w)


def main():
    config = CONFIG.copy()
    weights_path = resolve_weights_path()
    project_dir = resolve_project_dir()
    source = config["source"]

    if not source:
        raise ValueError("CONFIG['source'] を指定してください。")

    print(f"[INFO] 使用重み: {weights_path}")
    print(f"[INFO] 推論対象: {source}")
    print(f"[INFO] TTA: 0/90/180/270° 回転, WBF iou_thr={config['iou_thr']}, skip_box_thr={config['skip_box_thr']}")

    model = YOLO(weights_path)
    out_dir = Path(project_dir) / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 入力がファイルかディレクトリか
    src_path = Path(source)
    if src_path.is_file():
        image_paths = [src_path]
    elif src_path.is_dir():
        image_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            image_paths.extend(src_path.glob(ext))
        image_paths.sort()
    else:
        raise FileNotFoundError(f"ソースが見つかりません: {source}")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 読み込み失敗: {img_path}")
            continue
        save_path = out_dir / img_path.name
        print(f"[INFO] 処理中: {img_path.name}")
        process_one_image(model, img, save_path, config)

    print(f"[INFO] 推論完了。結果は {out_dir} に保存されています。")


if __name__ == "__main__":
    main()
