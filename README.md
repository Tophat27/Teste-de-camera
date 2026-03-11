# ShelfMonitor 🛒

Sistema de monitoramento ao vivo de prateleiras usando câmera OV5647 (130° FOV)  
com correção de olho de peixe e detecção de interação via YOLO.

---

## Estrutura do Projeto

```
shelf_monitor/
├── main.py         ← Ponto de entrada
├── config.py       ← Todos os parâmetros (edite aqui)
├── fisheye.py      ← Correção OV5647 (parâmetros calibrados)
├── detector.py     ← YOLO + lógica de interação
├── server.py       ← Streaming MJPEG + API REST (Flask)
└── templates/
    └── dashboard.html ← Interface web ao vivo
```

---

## Instalação

```bash
pip install ultralytics flask opencv-python numpy
```

---

## Uso

### Câmera ao vivo
```bash
python main.py
```

### Testar com vídeo gravado
```bash
python main.py --source video.mp4
```

### Apenas janela local (sem servidor web)
```bash
python main.py --no-stream
```

### Câmera já corrigida (pular fisheye)
```bash
python main.py --no-fisheye
```

Depois abra: **http://localhost:5000**

---

## Parâmetros Calibrados (OV5647 130°)

```python
fx = fy = 220.0   # focal length (px)
cx = 320.0        # centro óptico X
cy = 230.0        # centro óptico Y (levemente deslocado)
k1 = k2 = k3 = k4 = 0.0   # coeficientes distorção
rotate_180 = True           # câmera montada invertida
```

---

## Como treinar o YOLO para seus produtos

### 1. Coletar imagens
Use frames já corrigidos pelo fisheye:
```bash
python main.py --no-stream --no-fisheye=False
# Pressione 'S' no terminal para salvar frames de exemplo
```

### 2. Anotar com Roboflow ou Label Studio
- Crie classes: `pessoa`, `detergente`, `shampoo`, etc.
- Anote bounding boxes nas imagens corrigidas

### 3. Treinar
```bash
yolo train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### 4. Usar o modelo treinado
Em `config.py`:
```python
YOLO_MODEL = "runs/detect/train/weights/best.pt"
CLASS_PRODUCTS = ["detergente", "shampoo", "sabao"]
```

---

## Lógica de Detecção de Interação

```
Pessoa detectada  +  Produto detectado
         ↓
   Distância entre bboxes ≤ INTERACTION_DIST_PX (60px)
         ↓
   Conta frames consecutivos
         ↓
   ≥ INTERACTION_MIN_FRAMES (8 frames) → ALERTA
         ↓
   Cooldown de INTERACTION_COOLDOWN_S (5s) por produto
```

---

## Ajuste Fino

| Parâmetro | Arquivo | Quando ajustar |
|---|---|---|
| `INTERACTION_DIST_PX` | config.py | Falsos positivos (muito distante) |
| `INTERACTION_MIN_FRAMES` | config.py | Alertas muito rápidos/lentos |
| `SHELF_ZONE` | config.py | Limitar área monitorada |
| `YOLO_CONF` | config.py | Muitas detecções erradas |
| `YOLO_DEVICE` | config.py | Usar `"cuda"` se tiver GPU |
