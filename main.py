"""
main.py — ShelfMonitor: loop principal
Uso:
    python main.py
    python main.py --source video.mp4       # testar com vídeo
    python main.py --source imagem.jpg      # testar com imagem estática
    python main.py --no-stream              # sem servidor web
"""

import cv2
import time
import threading
import argparse
import sys

from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
from fisheye  import FisheyeCorrector
from detector import ShelfDetector
import server


# ── Argparse ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ShelfMonitor")
parser.add_argument("--source",    default=None,
                    help="Câmera (0,1..), RTSP url ou arquivo de vídeo/imagem")
parser.add_argument("--no-stream", action="store_true",
                    help="Desabilita servidor web (exibe janela local)")
parser.add_argument("--no-fisheye", action="store_true",
                    help="Pula correção fisheye (câmera já corrigida)")
args = parser.parse_args()


# ── Abertura da fonte de vídeo ────────────────────────────────────
def open_source(src):
    if src is None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    elif src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        sys.exit(f"[ERRO] Não foi possível abrir: {src or CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    return cap


# ── Loop principal ────────────────────────────────────────────────
def main():
    corrector = None if args.no_fisheye else FisheyeCorrector()
    detector  = ShelfDetector()

    # Iniciar servidor web numa thread separada
    if not args.no_stream:
        t = threading.Thread(target=server.start, daemon=True)
        t.start()
        print(f"[ShelfMonitor] Dashboard → http://localhost:{server.STREAM_PORT}")

    cap = open_source(args.source)

    fps_counter, fps_t0, fps_val = 0, time.time(), 0.0

    print("[ShelfMonitor] Iniciando captura... (Ctrl+C para parar)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Fim de arquivo: reinicia
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break

            # 1. Correção fisheye
            if corrector:
                frame = corrector.correct(frame)

            # 2. Inferência YOLO + detecção de interação
            annotated, events = detector.run(frame)

            # 3. Overlay de FPS
            cv2.putText(annotated, f"{fps_val:.1f} FPS",
                        (8, annotated.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1)

            # 4. Publicar frame + eventos no servidor web
            if not args.no_stream:
                server.push_frame(annotated)
                for ev in events:
                    server.push_event(ev)
                server.update_fps(fps_val)

            # 5. Janela local (opcional)
            if args.no_stream:
                cv2.imshow("ShelfMonitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 6. Calcular FPS real
            fps_counter += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 2.0:
                fps_val    = fps_counter / elapsed
                fps_counter = 0
                fps_t0     = time.time()

            # 7. Log de eventos no terminal
            for ev in events:
                print(f"  ⚡ [{ev.time_str}] INTERAÇÃO → {ev.product_label}")

    except KeyboardInterrupt:
        print("\n[ShelfMonitor] Encerrado pelo usuário.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
