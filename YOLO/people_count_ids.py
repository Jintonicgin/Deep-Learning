import os
import argparse
from typing import List, Tuple, Set
import numpy as np
import imageio
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


# ================================
# 폰트 : 렌더링 ID 텍스트 사용(PIL)
# ================================
def ensure_font():
    for p in [
        r"/Users/mmymacymac/Downloads/나눔 글꼴/나눔고딕/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf",  # 영문
    ]:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, 16)
            except Exception:
                pass
    return ImageFont.load_default()


# ================================
# 모델 경로
# ================================
def resolve_model_source(model_arg: str) -> str:
    # 로컬 파일이면 그대로, 아니면 별칭을 간주하여 YOLO가 자동 다운로드
    if os.path.isfile(model_arg):
        return model_arg
    return model_arg


# ================================
# 분류한 클래스 이름
# ================================
def get_class_names(model) -> List[str]:
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


# ================================
# IoU 계산
# ================================
def iou_xxyy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# ================================
# 간단 IoU 기반 트래커
# ================================
class Track:  # 개별 객체 단위 상태
    __slots__ = ("tid", "bbox", "hits", "missed")

    def __init__(self, tid: int, bbox: np.ndarray):
        self.tid = tid                # 사람마다 부여되는 고유 ID
        self.bbox = bbox.astype(float)  # bounding box [x1,y1,x2,y2] : 현재 프레임 안에서 사람이 위치한 영역
        self.hits = 1                 # 특정 ID로 프레임마다 계속 검출된 횟수
        self.missed = 0               # tid가 살아 있지만 이번 프레임에서는 해당 사람을 못찾은 것, 연속해서 매칭 실패한 프레임 수


class PersonTracker:  # 여러 트랙의 관리자, 즉, 유니크한 인원수 집계
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30, min_hits: int = 3):
        self.iou_thresh = float(iou_thresh)   # 매칭 기준 IoU 임계값, 이 값 이상이면 같은 사람으로 매칭
        self.max_age = int(max_age)           # 연속 missed 허용 한도, 이를 넘기면 트랙을 만료(삭제)
        self.min_hits = int(min_hits)         # 유니크 확정 최소 관측 횟수
        self.next_id = 1                      # 새 트랙에 부여할 다음 ID 시작값은 1
        self.tracks: List[Track] = []         # 현재 유지되고 있는 사람들 track 객체
        self.unique_confirmed: Set[int] = set()  # 유니크로 확정된 tid들의 집합

    """
    1) 입력 dets(검출 바운딩 박스들)와 기존 self.tracks(추적 중인 트랙들) 사이의 IoU를 기준으로 1:1 greedy 매칭을 수행함.
       각 검출(detection)은 최대 하나의 트랙과만 연결될 수 있음. 각 트랙 역시 하나의 검출과 연결됨
       (greedy = 순차적으로 현재 가장 좋은(최대 IoU) 매칭을 선택하는 방법).
    2) 각 검출을 한 번씩 순회하면서, dets 배열을 (현재 사용되지 않은) 트랙들 중 IoU가 가장 큰 트랙을 찾음.
    3) 최대 IoU가 임계값 self.iou_thresh 이상인 경우에만 매칭하고, 매칭 실패한 검출/트랙은 각각 unmatched_dets, unmatched_tracks로 반환함.
    4) 매칭에 실패한 검출 인덱스와 트랙 인덱스는 이후 갱신 단계에서
       - unmatched_dets: 새로운 트랙 생성
       - unmatched_tracks: 오래된 트랙 유지/삭제 판단(지속 누락 시 삭제)
       반대로 몇 프레임 동안 누락되었다가 다시 검출되면 재매칭될 수 있음.
    """

    def _match(self, dets: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        # dets(detections) : 현재 프레임에서 검출된 사람의 바운딩 박스 좌표 모음(배열)
        matches: List[Tuple[int, int]] = []
        if len(self.tracks) == 0 or len(dets) == 0:
            return matches, list(range(len(dets))), list(range(len(self.tracks)))

        used_tracks = set()
        used_dets = set()

        for di, db in enumerate(dets):
            best_iou, best_ti = 0.0, -1
            for ti, trk in enumerate(self.tracks):
                if ti in used_tracks:
                    continue
                iou = iou_xxyy(db, trk.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_ti = ti
            if best_ti >= 0 and best_iou > self.iou_thresh:
                matches.append((di, best_ti))
                used_dets.add(di)
                used_tracks.add(best_ti)

        unmatched_dets = [i for i in range(len(dets)) if i not in used_dets]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        return matches, unmatched_dets, unmatched_tracks

    def update(self, dets: np.ndarray) -> List[Track]:
        # 한 프레임에 대해 트래커의 핵심 갱신
        matches, unmatched_dets, unmatched_tracks = self._match(dets)

        # 매칭된 트랙 갱신
        for di, ti in matches:
            trk = self.tracks[ti]
            trk.bbox = dets[di].astype(float)
            trk.hits += 1
            trk.missed = 0
            if trk.hits >= self.min_hits:
                self.unique_confirmed.add(trk.tid)

        # 매칭 실패한 기존 트랙은 missed 증가
        for ti in unmatched_tracks:
            trk = self.tracks[ti]
            trk.missed += 1

        # 오래된 트랙 제거
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

        # 매칭 실패한 검출은 새 트랙 생성
        for di in unmatched_dets:
            t = Track(self.next_id, dets[di])
            self.tracks.append(t)
            self.next_id += 1

        return self.tracks

    def unique_count(self) -> int:  # 유니크로 확정된 사람 수
        return len(self.unique_confirmed)


# ================================
# 렌더링 : 한 프레임 위에 박스와 ID 라벨을 그려서 numpy 배열로 주석 프레임을 반환
# ================================
def draw_boxes_with_ids(frame_rgb: np.ndarray,
                        tracks: List[Track],
                        overlay_text: str) -> np.ndarray:
    img = Image.fromarray(frame_rgb)
    drw = ImageDraw.Draw(img)
    font = ensure_font()

    for trk in tracks:
        x1, y1, x2, y2 = [float(v) for v in trk.bbox]
        drw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        tag = f"ID {trk.tid}"
        l, t, r, b = drw.textbbox((0, 0), tag, font=font)
        tw, th = (r - l), (b - t)
        top = max(0.0, y1 - th)
        drw.rectangle([x1, top, x1 + tw + 8, y1], fill=(0, 255, 0))
        drw.text((x1 + 4, top + 2), tag, fill=(0, 0, 0), font=font)

    if overlay_text:
        l, t, r, b = drw.textbbox((0, 0), overlay_text, font=font)
        drw.rectangle([5, 5, 5 + (r - l) + 14, 5 + (b - t) + 14], fill=(0, 0, 0))
        drw.text((12, 12), overlay_text, fill=(255, 255, 255), font=font)

    return np.asarray(img)


# ================================
# 메인
# ================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="YOLO 모델 경로 또는 별칭(예: yolov8s.pt)")
    ap.add_argument("--video-in", required=True, help="입력 영상 경로(mp4 등)")
    ap.add_argument("--video-out", required=True, help="출력 mp4 경로(예: .\\output.mp4)")
    ap.add_argument("--conf", type=float, default=0.30, help="신뢰도 임계값")
    ap.add_argument("--iou", type=float, default=0.50, help="NMS IoU 임계값")
    ap.add_argument("--device", type=str, default=None, help="예: 'cpu' 또는 '0'(GPU)")
    ap.add_argument("--macro-block-size", type=int, default=1, help="FFmpeg macro block size")

    # 트래커 하이퍼파라미터
    ap.add_argument("--track-iou-thresh", type=float, default=0.3, help="트랙-검출 매칭 IoU 임계값")
    ap.add_argument("--max-age", type=int, default=30, help="미관측 허용 프레임 수")
    ap.add_argument("--unique-min-hits", type=int, default=3, help="유니크 확정 최소 관측 수")

    args = ap.parse_args()

    # 모델과 클래스 이름 읽어오기
    model_path = resolve_model_source(args.model)
    model = YOLO(model_path)  # yolov8s.pt가 없으면 최초 1회 자동 다운로드
    class_names = get_class_names(model)
    person_idx = class_names.index("person")

    # 비디오 IO
    reader = imageio.get_reader(args.video_in)
    meta = reader.get_meta_data()  # fps, size(폭, 높이), nframes(총 프레임수), duration
    fps = float(meta.get("fps", 30.0))  # fps 정보가 없으면 기본값으로 30.0
    writer = imageio.get_writer(
        args.video_out, fps=fps, codec="libx264", quality=8, macro_block_size=args.macro_block_size  # H.264, quality(0~10)
    )

    # 트래커/통계
    tracker = PersonTracker(
        iou_thresh=args.track_iou_thresh, max_age=args.max_age, min_hits=args.unique_min_hits
    )
    frames_processed = 0
    total_person_boxes = 0
    failed_reads = 0

    try:
        for idx, frame in enumerate(reader):
            frames_processed += 1

            """
            pred = YOLO 모델이 한 프레임에 대해 반환한 Results 객체임!
            pred.boxes는 해당 프레임에서 검출된 모든 객체의 바운딩 박스 모음임.
            dets에 들어갈 값은 pred.boxes에서 추출한 박스 좌표 배열임.
            [0]은 1장의 이미지가 입력되었기 때문임. [Results]
            """
            pred = model.predict(
                source=frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False
            )[0]

            if pred.boxes is None or len(pred.boxes) == 0:
                tracker.update(np.zeros((0, 4), dtype=float))
                overlay = f"unique: {tracker.unique_count()} active: {len(tracker.tracks)} frame: {idx}"
                annotated = draw_boxes_with_ids(frame, tracker.tracks, overlay)
                writer.append_data(annotated)
                print(f"Frame {idx}: 0 person boxes, unique_total={tracker.unique_count()}, active_tracks={len(tracker.tracks)}")
                continue

            # YOLO 검출 결과에서 N×4 좌표배열/신뢰도/클래스ID를 Numpy로 추출
            boxes = pred.boxes.xyxy.cpu().numpy()     # 바운딩 박스 좌표 (x1,y1,x2,y2)
            confs = pred.boxes.conf.cpu().numpy()     # confidence(신뢰도)
            clses = pred.boxes.cls.cpu().numpy().astype(int)  # 클래스 ID (0=person)

            # 사람(person) 클래스이면서 신뢰도가 conf 이상인 것만 채택
            keep = (clses == person_idx) & (confs >= args.conf)

            if not np.any(keep):
                tracker.update(np.zeros((0, 4), dtype=float))
                overlay = f"unique: {tracker.unique_count()} active: {len(tracker.tracks)} frame: {idx}"
                annotated = draw_boxes_with_ids(frame, tracker.tracks, overlay)
                writer.append_data(annotated)
                print(f"Frame {idx}: 0 person boxes, unique_total={tracker.unique_count()}, active_tracks={len(tracker.tracks)}")
                continue

            person_boxes = boxes[keep]                           # 박스가 0개인데 형식은 [x1,y1,x2,y2] 좌표구조임.
            total_person_boxes += person_boxes.shape[0]

            active_tracks = tracker.update(person_boxes)
            overlay = f"unique: {tracker.unique_count()} active: {len(active_tracks)} frame: {idx}"
            annotated = draw_boxes_with_ids(frame, active_tracks, overlay)
            writer.append_data(annotated)

            print(
                f"Frame {idx}: {person_boxes.shape[0]} person boxes, "
                f"unique_total={tracker.unique_count()}, active_tracks={len(active_tracks)}"
            )

    finally:
        try:
            writer.close()
        except Exception:
            pass
        try:
            reader.close()
        except Exception:
            pass

    print(f"[OK] saved: {args.video_out}")
    print(
        "[SUMMARY] "
        f"frames_processed={frames_processed}, total_person_boxes={total_person_boxes}, "
        f"unique_persons={tracker.unique_count()}"
    )

    import json, os
    summary = {
        "frames_processed": int(frames_processed),
        "total_person_boxes": int(total_person_boxes),
        "unique_persons": int(tracker.unique_count())
    }
    # 원자적(atomic) 저장: 임시 파일에 쓴 후 교체
    tmpfile = "summary.json.tmp"
    dstfile = "summary.json"

    with open(tmpfile, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    os.replace(tmpfile, dstfile)  # summary.json 으로 교체
    print(f"[SUMMARY] saved to {dstfile}")


if __name__ == "__main__":
    main()