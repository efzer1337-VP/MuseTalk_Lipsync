import os
import sys
import queue
import threading
import cv2
import time
import torch

# Добавляем корневую папку проекта в пути поиска Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel
from musetalk.utils.face_parsing import FaceParsing

# Импортируем сам модуль, чтобы иметь возможность инициализировать в нем глобальные переменные
import scripts.realtime_inference as realtime_inference
from scripts.realtime_inference import Avatar

def main():
    # 1. Эмуляция аргументов (аналог того, что собирает parser в realtime_inference.py)
    class Args:
        version = "v15"
        ffmpeg_path = "./ffmpeg-4.4-amd64-static/"
        gpu_id = 0
        vae_type = "sd-vae"
        unet_config = "./models/musetalk/musetalk.json" 
        unet_model_path = "./models/musetalkV15/unet.pth" # Для v15 путь такой
        whisper_dir = "./models/whisper"
        bbox_shift = 0
        extra_margin = 10
        fps = 25
        audio_padding_length_left = 2
        audio_padding_length_right = 2
        batch_size = 20
        parsing_mode = "jaw"
        left_cheek_width = 90
        right_cheek_width = 90
        skip_save_images = True # ОБЯЗАТЕЛЬНО ТАК, чтобы не писать на диск
        
    args = Args()
    
    # Прокидываем args в модуль
    realtime_inference.args = args

    # 2. Инициализация моделей (для RTX 4080 займет пару секунд)
    print("Loading models (this takes a few seconds)...")
    device = torch.device(f"cuda:{args.gpu_id}")
    
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    timesteps = torch.tensor([0], device=device)
    weight_dtype = unet.model.dtype

    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    if args.version == "v15":
        fp = FaceParsing(left_cheek_width=args.left_cheek_width, right_cheek_width=args.right_cheek_width)
    else:
        fp = FaceParsing()

    # Прокидываем загруженные модели в глобальное пространство модуля realtime_inference,
    # потому что класс Avatar берет их оттуда без передачи через конструктор.
    realtime_inference.device = device
    realtime_inference.vae = vae
    realtime_inference.unet = unet
    realtime_inference.pe = pe
    realtime_inference.timesteps = timesteps
    realtime_inference.weight_dtype = weight_dtype
    realtime_inference.audio_processor = audio_processor
    realtime_inference.whisper = whisper
    realtime_inference.fp = fp

    # 3. Подготовка Аватара
    avatar_id = "sun_avatar_test"
    video_path = "./data/video/sun.mp4" # Тестовое видео
    audio_path = "./data/audio/sun.wav" # Тестовое аудио
    
    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        print(f"Файл {video_path} или {audio_path} не найден!")
        return

    print("Preparing Avatar... (If it asks to re-create: type 'n' to use cached frames, or 'y' to process the video from scratch)")
    avatar = Avatar(
        avatar_id=avatar_id,
        video_path=video_path,
        bbox_shift=args.bbox_shift,
        batch_size=args.batch_size,
        preparation=True # Нарежет видео и подготовит маски. 
    )

    # 4. Запуск Реалтайм инференса
    out_frame_queue = queue.Queue()

    def run_musetalk_inference():
        print(f"Starting generation for audio: {audio_path}")
        avatar.inference(
            audio_path=audio_path,
            out_vid_name=None,
            fps=args.fps,
            skip_save_images=args.skip_save_images,
            out_frame_queue=out_frame_queue
        )
        # Отправляем None, чтобы мы знали, что видео закончилось
        out_frame_queue.put(None)

    # Запускаем генерацию в фоне
    inference_thread = threading.Thread(target=run_musetalk_inference)
    inference_thread.start()

    # 5. Получение и показ кадров "на лету"
    print("Playing realtime frames...")
    window_name = "MuseTalk Realtime (RTX 4080)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Расчет задержки для 25 FPS
    frame_delay = 1.0 / args.fps
    
    while True:
        try:
            start_time = time.time()
            
            # Ждем кадр
            frame = out_frame_queue.get(timeout=10)
            
            if frame is None:
                print("End of stream.")
                break
            
            # Показываем кадр в окне
            cv2.imshow(window_name, frame)
            
            # Считаем, сколько нужно подождать, чтобы проигрывать именно в 25 FPS
            elapsed = time.time() - start_time
            sleep_time = max(1, int((frame_delay - elapsed) * 1000))
            
            # Нажмите 'q', чтобы закрыть
            if cv2.waitKey(sleep_time) & 0xFF == ord('q'):
                break
                
        except queue.Empty:
            print("Timeout: no frame received for 10 seconds.")
            break

    cv2.destroyAllWindows()
    inference_thread.join()
    print("Done!")

if __name__ == "__main__":
    main()
