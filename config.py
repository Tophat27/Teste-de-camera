# ─────────────────────────────────────────────────────────────────
#  ShelfMonitor — Configuração Central
# ─────────────────────────────────────────────────────────────────

# ── Câmera ────────────────────────────────────────────────────────
CAMERA_INDEX   = 0          # índice V4L2 ou RTSP url  ex: "rtsp://..."
CAMERA_WIDTH   = 640
CAMERA_HEIGHT  = 480
CAMERA_FPS     = 15

# ── Correção Fisheye (OV5647 130°) ───────────────────────────────
FISHEYE = dict(
    fx   = 220.0,
    fy   = 220.0,
    cx   = 320.0,
    cy   = 230.0,
    k1   = 0.0,
    k2   = 0.0,
    k3   = 0.0,
    k4   = 0.0,
    rotate_180 = True,       # câmera montada invertida
)

# ── YOLO ──────────────────────────────────────────────────────────
YOLO_MODEL    = "yolov8n.pt"   # trocar pelo modelo treinado
YOLO_CONF     = 0.45
YOLO_IOU      = 0.45
YOLO_DEVICE   = "cpu"          # "cuda" se tiver GPU
YOLO_IMGSZ    = 640

# Classes de interesse (nomes no modelo treinado)
CLASS_PERSON   = "person"
CLASS_PRODUCTS = []            # ex: ["detergente","shampoo"] — vazio = tudo exceto person

# ── Detecção de Interação ─────────────────────────────────────────
# Distância mínima (px) entre bbox de pessoa e bbox de produto
INTERACTION_DIST_PX   = 60
# Frames consecutivos para confirmar interação (evitar falso positivo)
INTERACTION_MIN_FRAMES = 8
# Segundos de cooldown após disparar alerta do mesmo produto
INTERACTION_COOLDOWN_S = 5.0

# ── Zona de Prateleira (opcional) ────────────────────────────────
# Define a ROI onde os produtos ficam. None = imagem inteira.
# Formato: (x1, y1, x2, y2) em pixels após correção fisheye
SHELF_ZONE = None  # ex: (80, 30, 560, 200)

# ── Streaming Web ─────────────────────────────────────────────────
STREAM_HOST  = "0.0.0.0"
STREAM_PORT  = 5000
STREAM_JPEG_QUALITY = 80     # 1-100
MAX_EVENTS   = 100           # histórico de eventos no dashboard
