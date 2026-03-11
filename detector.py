"""
detector.py — YOLO + lógica de detecção de interação com produtos
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Instale: pip install ultralytics")

from config import (
    YOLO_MODEL, YOLO_CONF, YOLO_IOU, YOLO_DEVICE, YOLO_IMGSZ,
    CLASS_PERSON, CLASS_PRODUCTS,
    INTERACTION_DIST_PX, INTERACTION_MIN_FRAMES, INTERACTION_COOLDOWN_S,
    SHELF_ZONE,
)


@dataclass
class Detection:
    label: str
    conf: float
    box: Tuple[int, int, int, int]   # x1,y1,x2,y2
    is_person: bool = False

    @property
    def cx(self): return (self.box[0] + self.box[2]) // 2
    @property
    def cy(self): return (self.box[1] + self.box[3]) // 2
    @property
    def area(self): return (self.box[2]-self.box[0]) * (self.box[3]-self.box[1])


@dataclass
class InteractionEvent:
    timestamp: float
    person_box: Tuple
    product_label: str
    product_box: Tuple
    frame_count: int = 0

    @property
    def time_str(self):
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


def _box_dist(b1, b2) -> float:
    """Distância mínima entre dois bboxes (0 se sobrepostos)."""
    ax1,ay1,ax2,ay2 = b1
    bx1,by1,bx2,by2 = b2
    dx = max(0, max(bx1-ax2, ax1-bx2))
    dy = max(0, max(by1-ay2, ay1-by2))
    return float(np.hypot(dx, dy))


def _in_shelf_zone(box) -> bool:
    if SHELF_ZONE is None:
        return True
    sx1,sy1,sx2,sy2 = SHELF_ZONE
    x1,y1,x2,y2 = box
    cx,cy = (x1+x2)//2, (y1+y2)//2
    return sx1<=cx<=sx2 and sy1<=cy<=sy2


class ShelfDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.model.to(YOLO_DEVICE)
        self._names = self.model.names          # id→label
        self._frame_counters = defaultdict(int) # label→frames consecutivos
        self._last_alert: dict = {}             # label→timestamp último alerta
        self.active_interactions: List[InteractionEvent] = []
        self.event_history: List[InteractionEvent] = []
        print(f"[ShelfDetector] modelo={YOLO_MODEL} device={YOLO_DEVICE}")

    # ── Inferência ────────────────────────────────────────────────

    def run(self, frame: np.ndarray) -> Tuple[np.ndarray, List[InteractionEvent]]:
        """
        Roda YOLO no frame, detecta interações e retorna:
          - frame anotado
          - lista de eventos ativos neste frame
        """
        results = self.model(frame,
                             conf=YOLO_CONF,
                             iou=YOLO_IOU,
                             imgsz=YOLO_IMGSZ,
                             verbose=False)[0]

        detections = self._parse(results)
        persons  = [d for d in detections if d.is_person]
        products = [d for d in detections if not d.is_person and _in_shelf_zone(d.box)]

        new_events = self._check_interactions(persons, products)
        annotated  = self._draw(frame.copy(), detections, new_events)
        return annotated, new_events

    # ── Parsing ───────────────────────────────────────────────────

    def _parse(self, results) -> List[Detection]:
        dets = []
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            label = self._names[int(box.cls[0])]

            if CLASS_PRODUCTS and label != CLASS_PERSON and label not in CLASS_PRODUCTS:
                continue

            dets.append(Detection(
                label=label, conf=conf,
                box=(x1,y1,x2,y2),
                is_person=(label == CLASS_PERSON)
            ))
        return dets

    # ── Lógica de Interação ───────────────────────────────────────

    def _check_interactions(self, persons, products) -> List[InteractionEvent]:
        triggered = []
        seen_labels = set()

        for person in persons:
            for product in products:
                dist = _box_dist(person.box, product.box)
                if dist <= INTERACTION_DIST_PX:
                    key = product.label
                    seen_labels.add(key)
                    self._frame_counters[key] += 1

                    if self._frame_counters[key] >= INTERACTION_MIN_FRAMES:
                        now = time.time()
                        last = self._last_alert.get(key, 0)
                        if now - last >= INTERACTION_COOLDOWN_S:
                            ev = InteractionEvent(
                                timestamp=now,
                                person_box=person.box,
                                product_label=key,
                                product_box=product.box,
                                frame_count=self._frame_counters[key],
                            )
                            self._last_alert[key] = now
                            self._frame_counters[key] = 0
                            triggered.append(ev)
                            self.event_history.append(ev)
                            if len(self.event_history) > 200:
                                self.event_history.pop(0)

        # Zerar contadores de produtos não vistos neste frame
        for key in list(self._frame_counters):
            if key not in seen_labels:
                self._frame_counters[key] = 0

        self.active_interactions = triggered
        return triggered

    # ── Anotação visual ───────────────────────────────────────────

    def _draw(self, frame, detections, events) -> np.ndarray:
        alerted_products = {ev.product_box for ev in events}

        for d in detections:
            x1,y1,x2,y2 = d.box
            if d.is_person:
                color = (0, 220, 255)
                cv_label = f"pessoa {d.conf:.0%}"
            else:
                color = (255, 180, 0) if d.box not in alerted_products else (0, 60, 255)
                cv_label = f"{d.label} {d.conf:.0%}"

            import cv2
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.rectangle(frame, (x1, y1-22), (x1+len(cv_label)*9, y1), color, -1)
            cv2.putText(frame, cv_label, (x1+3, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)

        # Linha de conexão pessoa ↔ produto em alerta
        import cv2
        for ev in events:
            px = (ev.person_box[0]+ev.person_box[2])//2
            py = (ev.person_box[1]+ev.person_box[3])//2
            qx = (ev.product_box[0]+ev.product_box[2])//2
            qy = (ev.product_box[1]+ev.product_box[3])//2
            cv2.line(frame, (px,py), (qx,qy), (0,60,255), 2)
            cv2.putText(frame, "INTERACAO!", (qx-40, qy-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,60,255), 2)

        # SHELF_ZONE overlay
        if SHELF_ZONE:
            import cv2
            sx1,sy1,sx2,sy2 = SHELF_ZONE
            cv2.rectangle(frame, (sx1,sy1), (sx2,sy2), (100,255,100), 1)

        return frame
