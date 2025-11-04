import cv2
import torch
from torch import load
from model import DETR
import albumentations as A
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler
import sys
import time 
import numpy as np

# Initialize logger and handlers
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")

# Transformações - CORRIGIDO
transforms = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(
        min_height=224, 
        min_width=224, 
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2()
])

# Model setup
model = DETR(num_classes=3)
model.eval()

# ✅ CORREÇÃO: Carregar modelo corretamente
try:
    checkpoint = torch.load('checkpoints/490_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint)
    logger.success("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
logger.realtime(f'Using device: {device}')

CLASSES = get_classes() 
COLORS = get_colors()
logger.info(f"Classes: {CLASSES}")

# Camera setup

logger.realtime("Starting camera capture...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    logger.error("❌ Cannot open camera")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
logger.info(f"Camera resolution: {frame_width}x{frame_height}")

# Initialize performance tracking
frame_count = 0
fps_start_time = time.time()

def draw_detection(frame, bbox, class_name, confidence, color):
    """Draw detection with better visibility"""
    x1, y1, x2, y2 = bbox
    
    # Garantir que as coordenadas estão dentro do frame
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_width, int(x2))
    y2 = min(frame_height, int(y2))
    
    # Verificar se a bbox é válida
    if x2 <= x1 or y2 <= y1:
        return frame
    
    # ✅ CORREÇÃO SIMPLES: Usar cor fixa temporariamente
    bgr_color = (0, 255, 0)  # Verde para todas as classes
    
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 4)
    
    # Background para texto
    label = f"{class_name}: {confidence:.2f}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
    
    text_y = max(y1 - 10, 20)
    
    # Desenhar fundo do texto
    cv2.rectangle(frame, 
                 (x1, text_y - label_size[1] - 10),
                 (x1 + label_size[0] + 10, text_y + 5),
                 bgr_color, -1)
    
    # Desenhar texto
    cv2.putText(frame, label, 
               (x1 + 5, text_y - 5), 
               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

logger.realtime("🎯 Starting detection loop... Press Q to quit")

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read frame from camera")
        break
    
    # Time the inference
    inference_start = time.time()
    
    try:
        # ✅ CORREÇÃO: Pré-processamento correto
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = transforms(image=rgb_frame)
        input_tensor = transformed['image'].unsqueeze(0)
        
        # ✅ CORREÇÃO IMPORTANTE: Mover tensor para o mesmo dispositivo do modelo
        input_tensor = input_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
        
        inference_time = (time.time() - inference_start) * 1000
        
        # ✅ CORREÇÃO: Processar outputs no dispositivo correto
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # [num_queries, num_classes]
        keep = probas.max(-1).values > 0.5
        
        # ✅ CORREÇÃO: Mover boxes para CPU antes de processar
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), (frame_width, frame_height))
        
        # Get predictions
        scores, classes = probas[keep].max(-1)
        
        # ✅ CORREÇÃO: Mover scores e classes para CPU
        scores_cpu = scores.cpu()
        classes_cpu = classes.cpu()
        
        # Prepare detections
        detections = []
        for idx, (score, class_idx, bbox) in enumerate(zip(scores_cpu, classes_cpu, bboxes_scaled)):
            bbox_np = bbox.numpy()
            
            detections.append({
                'class': CLASSES[class_idx.item()],
                'confidence': score.item(),
                'bbox': bbox_np.tolist()
            })
            
            # Desenhar detecções
            frame = draw_detection(
                frame, 
                bbox_np, 
                CLASSES[class_idx.item()], 
                score.item(), 
                COLORS[class_idx.item()]
            )
        
        # Log detections periodically
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed_time = time.time() - fps_start_time
            fps = 30 / elapsed_time if elapsed_time > 0 else 0
            
            if detections:
                detection_handler.log_detections(detections, frame_id=frame_count)
            
            detection_handler.log_inference_time(inference_time, fps)
            logger.info(f"📊 Frame {frame_count} | Detections: {len(detections)} | FPS: {fps:.1f}")
            
            fps_start_time = time.time()
            
    except Exception as e:
        logger.error(f"Inference error: {e}")
        continue

    # Mostrar frame
    display_frame = cv2.resize(frame, (1280, 720))
    
    # Adicionar informações na tela
    info_text = f"Detections: {len(detections)} | Press Q to quit"
    cv2.putText(display_frame, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Sign Language Detection - Press Q to quit', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        logger.realtime("Stopping real-time detection...")
        break

cap.release() 
cv2.destroyAllWindows()
logger.success("Real-time detection stopped")