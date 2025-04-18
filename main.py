#Gamdai v2
import os
import random

import PU

"""
This is main area where execution takes place
"""


def font_pic(pos):
    if pos == "PROPN":
        return ("fonts/dancing/static/DancingScript-Medium.ttf", 3 ,"black")
    elif pos == "PRON":
        return ("fonts/monoton/Monoton-Regular.ttf", 3, "black")
    else:
        return ("fonts/luckiest guy/LuckiestGuy-Regular.ttf", 7, "black")


with open("scripts/hooks file.txt", "r", encoding="utf-8") as hook_file:
    hook_list = hook_file.readlines()
    random_hook = random.choice(hook_list)
    hook = random_hook


with open("scripts/scripts file.txt", "r", encoding="utf-8") as script_file:
    all_lines = script_file.readlines()
    current_line = all_lines[0]




PU.system().in_folder_deletion(folder="processing/", shred=True)
# PU.system().in_folder_deletion(folder="videos/", shred=True)

audio = PU.audio_handler()
video_maker = PU.movie_maker()
text_processor = PU.text_processing()
downloader = PU.downloader()


def generate_background_video():
    gta_folder = os.listdir("bgv/downloaded vids/gta")
    minecraft_folder = os.listdir("bgv/downloaded vids/minecraft")
    gta_random = random.choice(gta_folder)
    minecraft_random = random.choice(minecraft_folder)
    # print(random_file_1, random_file_2)
    video_maker.overlap(video_maker.overlap_definer(filename=f"bgv/downloaded vids/gta/{gta_random}", resize=(1080, 990)), video_maker.overlap_definer(filename=f"bgv/downloaded vids/minecraft/{minecraft_random}", resize=(1080, 990), y_pos=960), bg_video=None, output_file="processing/dual_gen_bg.mp4")

generate_background_video()
# exit(0)


audio_file = "processing/audio.mp3"

audio.generate_audio(hook, text_processor.anti_symbols(text=current_line), output_file="processing/audio")
transcribed_dict = audio.transcribe(audio_path=audio_file)
video_maker.keep_duration(duration=audio.audio_time(audio_path=audio_file), filename="processing/dual_gen_bg.mp4", output_file="processing/audio_edited.mp4")
filtered = text_processor.anti_contractions(text=transcribed_dict["text"])
filtered = text_processor.anti_symbols(text=filtered)
pos_tagged = text_processor.pos_tag(filtered)

query = text_processor.query_generate(text=transcribed_dict["text"])

timings = []
for idx, pos in enumerate(pos_tagged):
    if pos in ["PROPN", "NOUN"]:
        if pos_tagged[idx-1] not in ["PROPN", "NOUN"]:
            timings.append(idx)
timings.append(len(pos_tagged)-1)


gif_database = downloader.smart_download(query_list=query)


drawtexts = []
for idx in range(len(transcribed_dict["sep"])):
    font_ = font_pic(pos_tagged[idx])
    drawtexts.append(video_maker.drawtext_definer(text=transcribed_dict["sep"][idx], start_time=transcribed_dict["s"][idx], end_time=transcribed_dict["e"][idx], x_pos="(W-text_w)/2", y_pos="((H-text_h)/2) + 200", font=font_[0], border=font_[1], border_color=font_[2], size="120+10*sin(1.6*PI*t/2)"))
video_maker.draw_text(*drawtexts, video_file="processing/audio_edited.mp4", output_file="processing/drawtext.mp4")

idx_main = 0
x= []

for idx, pos in enumerate(pos_tagged):
    if pos in ["PROPN", "NOUN"]:
        if pos_tagged[idx-1] not in ["PROPN", "NOUN"]:
            try:
                gif_dim = video_maker.video_dimensions(gif_database[idx_main])
                x.append(video_maker.overlap_definer(filename=gif_database[idx_main], start_time=transcribed_dict["s"][timings[idx_main]],
                                            end_time=transcribed_dict["s"][timings[idx_main+1]], x_pos="(W-500)/2+50*sin(2*PI*t/5)", y_pos="(H-500*3.5)/2+25*sin(2*PI*t/5)+200", resize=float(500/gif_dim[0]), round_corner=True))
                idx_main += 1
            except:
                pass


video_maker.overlap(*x, bg_video="processing/drawtext.mp4", output_file="processing/final_vid_without_audio.mp4")

audio.video_audio_merge(audio="music/idea 9.m4a", video="processing/final_vid_without_audio.mp4", output="processing/music_fit.mp4", volume=0.25)

audio.video_audio_merge(audio=audio_file, video="processing/music_fit.mp4", output=f"videos/final_output{random.randint(1000, 9000)}.mp4", volume=0.9)


try:
    open("scripts/scripts file.txt", "w", encoding="utf-8").close()
finally:
    with open("scripts/scripts file.txt", "w", encoding="utf-8") as script_file:
        script_file.writelines(all_lines[1:])
