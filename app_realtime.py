import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import sys
import queue
import threading
import cv2
import time
import torch
import glob
import builtins
import gradio as gr
import codecs

# Fix UnicodeEncodeError for Windows console when printing special characters like '「'
if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# Solve asynchronous IO issues on Windows for Gradio
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Добавляем корневую папку проекта в пути поиска Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel
from musetalk.utils.face_parsing import FaceParsing
import scripts.realtime_inference as realtime_inference
from scripts.realtime_inference import Avatar

# Глобальный флаг, чтобы не грузить модели каждый раз
MODELS_LOADED = False
GLOBAL_AVATARS = {}

def load_models_if_needed():
    global MODELS_LOADED
    if MODELS_LOADED:
        return
        
    print("Loading models for Realtime WebUI...")
    class Args:
        version = "v15"
        ffmpeg_path = "./ffmpeg-4.4-amd64-static/"
        gpu_id = 0
        vae_type = "sd-vae"
        unet_config = "./models/musetalkV15/musetalk.json" 
        unet_model_path = "./models/musetalkV15/unet.pth"
        whisper_dir = "./models/whisper"
        bbox_shift = 0
        extra_margin = 10
        fps = 25
        audio_padding_length_left = 2
        audio_padding_length_right = 2
        batch_size = 4
        parsing_mode = "jaw"
        left_cheek_width = 90
        right_cheek_width = 90
        skip_save_images = True
        
    args = Args()
    realtime_inference.args = args

    device = torch.device(f"cuda:{args.gpu_id}")
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device, memory_format=torch.channels_last)
    
    # Ускорение через torch.compile (доступно в PyTorch 2.0+)
    try:
        print("Compiling UNet for faster inference...")
        unet.model = torch.compile(unet.model, mode="reduce-overhead")
    except Exception as e:
        print(f"torch.compile failed: {e}. Continuing without compilation.")
        
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

    # Прокидываем загруженные модели в глобальное пространство модуля
    realtime_inference.device = device
    realtime_inference.vae = vae
    realtime_inference.unet = unet
    realtime_inference.pe = pe
    realtime_inference.timesteps = timesteps
    realtime_inference.weight_dtype = weight_dtype
    realtime_inference.audio_processor = audio_processor
    realtime_inference.whisper = whisper
    realtime_inference.fp = fp
    
    MODELS_LOADED = True
    print("Models loaded successfully!")

def get_available_videos():
    return sorted(glob.glob("./data/video/*.mp4") + glob.glob("./data/video/*.avi"))

def get_available_audios():
    return sorted(glob.glob("./data/audio/*.wav") + glob.glob("./data/audio/*.mp3"))

def start_realtime_inference(video_path, audio_path, force_recreate):
    try:
        if not video_path or not audio_path:
            raise gr.Error("Please select both video and audio files.")
            
        yield gr.skip(), gr.skip(), "### ⏳ Статус: Загрузка ИИ-моделей в память (займет пару секунд)...", gr.skip()
        
        load_models_if_needed()
        
        # Извлекаем имя для кэша
        avatar_id = os.path.basename(video_path).split('.')[0]
        
        # Проверяем, не сломан ли кэш от предыдущего падения
        avatar_path = f"./results/v15/avatars/{avatar_id}"
        latents_path = f"{avatar_path}/latents.pt"
        
        should_recreate = force_recreate
        if os.path.exists(avatar_path) and not os.path.exists(latents_path):
            should_recreate = True
            print("Detected incomplete cache. Forcing re-creation...")
        
        if should_recreate or not os.path.exists(avatar_path):
            yield gr.skip(), gr.skip(), "### ⏳ Статус: Подготовка кэша видео (нарезка кадров, поиск лиц). Это долгий процесс, займет около минуты...", gr.skip()
        else:
            yield gr.skip(), gr.skip(), "### ⚡ Статус: Найден готовый кэш! Запуск генерации...", gr.skip()
            
        # Переопределяем функцию input, чтобы Avatar не ждал ввода в консоли
        if should_recreate:
            builtins.input = lambda prompt="": "y"
        else:
            builtins.input = lambda prompt="": "n"
            
        # Уведомляем пользователя о начале кэширования через UI (отправляем пустой кадр и аудио None)
        yield None, gr.skip(), gr.skip(), gr.update(value=None)
        
        global GLOBAL_AVATARS
        if avatar_id in GLOBAL_AVATARS and not should_recreate:
            avatar = GLOBAL_AVATARS[avatar_id]
        else:
            avatar = Avatar(
                avatar_id=avatar_id,
                video_path=video_path,
                bbox_shift=0,
                batch_size=20,
                preparation=True 
            )
            GLOBAL_AVATARS[avatar_id] = avatar

        out_frame_queue = queue.Queue()

        def run_musetalk():
            try:
                avatar.inference(
                    audio_path=audio_path,
                    out_vid_name=None,
                    fps=25,
                    skip_save_images=True,
                    out_frame_queue=out_frame_queue
                )
            except Exception as e:
                import traceback
                print("Inference error:\n", traceback.format_exc())
            out_frame_queue.put(None)

        thread = threading.Thread(target=run_musetalk)
        thread.start()

        frame_delay = 1.0 / 25.0
        first_frame = True
        
        out_video_path = f"{avatar_path}/temp_preview.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = None
        
        while True:
            start_t = time.time()
            try:
                frame = out_frame_queue.get(timeout=10)
                if frame is None:
                    break
                
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(out_video_path, fourcc, 25.0, (w, h))
                
                writer.write(frame)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if first_frame:
                    # Запускаем аудио вместе с первым кадром
                    yield rgb_frame, gr.Audio(value=audio_path, autoplay=True, visible=False), "### 🟢 Статус: Идет трансляция (Live)..."
                    first_frame = False
                else:
                    yield rgb_frame, gr.skip(), gr.skip()
                
                elapsed = time.time() - start_t
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except queue.Empty:
                raise gr.Error("Timeout: Generator didn't produce frames for 10 seconds. Check console.")
                break
                
        thread.join()
        yield gr.skip(), gr.skip(), f"### ✅ Статус: Готово! Секвенция сохранена в папку аватара."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Backend error: {str(e)}")

# Стилизация
custom_css = """
#main-container { max-width: 1100px; margin: 0 auto; }
.output-image { border-radius: 12px; overflow: hidden; box-shadow: 0 8px 20px rgba(0,0,0,0.4); border: 1px solid #333; }
.start-btn { background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%) !important; border: none !important; color: white !important; font-size: 18px !important; transition: transform 0.2s; }
.start-btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(75, 108, 183, 0.5); }
"""

# Функция для вочера
def watcher_loop(video_path, watch_dir, save_sequence):
    if not video_path or not watch_dir:
        return "Ошибка: выберите аватар и папку для мониторинга."
    
    os.makedirs(watch_dir, exist_ok=True)
    print(f"Watcher started on: {watch_dir}")
    processed_files = set(glob.glob(os.path.join(watch_dir, "*.wav")))
    
    load_models_if_needed()
    avatar_id = os.path.basename(video_path).split('.')[0]
    
    if avatar_id not in GLOBAL_AVATARS:
        avatar = Avatar(avatar_id=avatar_id, video_path=video_path, bbox_shift=0, batch_size=4, preparation=True)
        GLOBAL_AVATARS[avatar_id] = avatar
    else:
        avatar = GLOBAL_AVATARS[avatar_id]

    while True:
        current_files = set(glob.glob(os.path.join(watch_dir, "*.wav")))
        new_files = current_files - processed_files
        
        for audio_path in new_files:
            print(f"New audio detected: {audio_path}")
            # Запускаем инференс
            avatar.inference(
                audio_path=audio_path,
                out_vid_name=None,
                fps=25,
                skip_save_images=not save_sequence
            )
            processed_files.add(audio_path)
            print(f"Processing finished for: {audio_path}")
            
        time.sleep(0.5)

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as ui:
    with gr.Column(elem_id="main-container"):
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px 0;">
                <h1 style="font-size: 2.5em; margin-bottom: 0;">🎙️ MuseTalk Real-Time Studio</h1>
                <p style="font-size: 1.2em; color: #666;">⚡ <i>Optimized for T2S Pipeline</i> — Automate sequence generation from wav files.</p>
            </div>
            """
        )
        
        with gr.Tabs():
            with gr.Tab("Manual Control"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Control Panel")
                        video_input = gr.Dropdown(choices=get_available_videos(), label="🎬 Reference Video", allow_custom_value=True)
                        audio_input = gr.Dropdown(choices=get_available_audios(), label="🎵 Driving Audio", allow_custom_value=True)
                        
                        save_sequence = gr.Checkbox(label="💾 Save PNG Sequence", value=True)
                        force_recreate = gr.Checkbox(label="🔄 Force Re-Process Video", value=False)
                        
                        start_btn = gr.Button("🚀 START INFERENCE", elem_classes="start-btn", size="lg")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 📺 Output")
                        status_text = gr.Markdown("### 🕒 Статус: Ожидание запуска...")
                        video_output = gr.Image(label="Live Stream", interactive=False, elem_classes="output-image")
                        audio_output = gr.Audio(label="Audio Stream", visible=False)
                
                start_btn.click(
                    fn=start_realtime_inference,
                    inputs=[video_input, audio_input, force_recreate],
                    outputs=[video_output, audio_output, status_text]
                )

            with gr.Tab("Watcher Mode (Auto)"):
                gr.Markdown("Мониторинг папки: как только T2S положит туда WAV, мы начнем генерацию.")
                with gr.Row():
                    watch_video_input = gr.Dropdown(choices=get_available_videos(), label="🎬 Выберите Аватар")
                    watch_dir_input = gr.Textbox(value="./data/audio/input_stream", label="📂 Папка для мониторинга")
                    watch_save_png = gr.Checkbox(label="💾 Сохранять PNG", value=True)
                
                watch_btn = gr.Button("🔍 START WATCHER", variant="primary")
                watch_status = gr.Markdown("Watcher status: Stopped")
                
                watch_btn.click(
                    fn=watcher_loop,
                    inputs=[watch_video_input, watch_dir_input, watch_save_png],
                    outputs=[watch_status]
                )

if __name__ == "__main__":
    ui.queue().launch(server_port=7861, inbrowser=True)
