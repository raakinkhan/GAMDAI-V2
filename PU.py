"""
Processing Unit
"""
from spacy.util import filter_spans
from PIL import Image, ImageDraw
import random
from duckduckgo_search import DDGS
import requests
import os
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import whisper
import spacy
import contractions
import string
from spacy.matcher import Matcher
import ffmpeg
from mutagen.mp3 import MP3
from pydub import AudioSegment
from io import BytesIO
import subprocess
import json
import yt_dlp
from keybert import KeyBERT

# print(decorator.__version__)
class system:
    def __init__(self):
        pass

    def write(self, data, file, type_="w"):
        with open(file, type_) as file_W:
            file_W.write(data)

    def shred(self, path, passes=10):
        """
        :param path: file path to shred
        :param passes: number of shreds
        :return: deletes the file
        """
        for pass_ in range(passes):
            with open(path, "rb") as shreding_file:
                data = shreding_file.read()
                random_data = os.urandom(len(data))
            with open(path, "wb") as shreding_file:
                shreding_file.write(random_data)
        os.remove(path)



    def in_folder_deletion(self, folder: str, shred=False):
        """
        :param folder: specific folder to delete
        :param shred: Bool, if you want shreding or not
        :return: deletes the file ultimately
        """
        files = os.listdir(folder)
        if len(files) != 0:
            for file in files:
                if shred is False:
                    os.remove(folder+file)
                else:
                    self.shred(path=folder+file, passes=30)






class text_processing:
    def __init__(self):
        self.spacy_nlp = spacy.load("en_core_web_md")

    def anti_contractions(self, text):
        """
        :return: returns a string without contractions
        """
        contract_fix = contractions.fix(text)
        return contract_fix


    def anti_symbols(self, text):
        """
        :return: returns a string without punctuations
        """
        text_ = list(filter(lambda x: False if x in string.punctuation else True, text))
        text_ = "".join(text_)
        return text_


    def pos_tag(self, text):
        """
        :return: returns parts of speech of each word in a given txt snippet
        """
        pos_list = []
        for token in self.spacy_nlp(text):
            pos_list.append(token.pos_)
        return pos_list


    def query_generate(self, text):
        """
        :return: returns a list of detailed search queries
        """



        # def detect_word_and_give_pos_tag(word):
        #     pos_tagged_ = self.pos_tag((text))
        #     text_tokenized_ = text.split()
        #     idx = text_tokenized_.index(word)
        #     return pos_tagged_[idx]
        # # print(pos_tagged_, text_tokenized_)

        matcher = Matcher(self.spacy_nlp.vocab)

        # Define the patterns
        pattern_adj_noun = [
            {"POS": "ADJ", "OP": "+"},  # one or more adjectives
            {"POS": "NOUN"}  # followed by a noun
        ]

        pattern_adj_propn = [
            {"POS": "ADJ", "OP": "+"},
            {"POS": "PROPN"}
        ]

        noun_to_noun = [
            {"POS": "NOUN", "OP": "+"}
        ]

        propn_to_propn = [
            {"POS": "PROPN", "OP": "+"}
        ]



        # interjections = [
        #     {"POS": "INTJ", "OP":"+"}
        # ]
        #
        # verb = [
        #     {"POS": "VERB"}
        # ]

        # Add patterns to the matcher
        matcher.add("PATTERN_ADJ_NOUN", [pattern_adj_noun])
        matcher.add("PATTERN_NOUN_TO_NOUN", [noun_to_noun])
        matcher.add("PATTERN_PROPN_TO_PROPN", [propn_to_propn])
        # matcher.add("PATTERN_INTERJECTION", [interjections])
        # matcher.add("PATTERN_ADJ_PROPN", [pattern_adj_propn])
        # matcher.add("PATTERN_VERB", [verb])

        # Process the text
        doc = self.spacy_nlp(text)

        # Collect all matched spans
        spans = []
        for match_id, start, end in matcher(doc):

            span = doc[start:end]
            spans.append(span)


        # Filter overlapping spans: this function sorts and returns the longest spans while discarding overlaps
        filtered_spans = filter_spans(spans)

        # Create the final query list
        query = [span.text for span in filtered_spans]

        # pos_tags = list(map(detect_word_and_give_pos_tag, query))
        #
        # print(po)



        return query

    def keywords_detect(self, text, k=1):
        kw_model = KeyBERT()
        main_keyword = kw_model.extract_keywords(text)
        query = []
        for i in range(k):
            query.append(main_keyword[i][0])
        return query

class audio_handler:
    def __init__(self):
        self.client = ElevenLabs(api_key="sk_a7321c729abb932d171003dd0755f9d9e3e4d0dcd7fb0f7e")
        self.model = whisper.load_model("tiny")

    def generate_audio(self, *text, output_file):
        """
        :param text: send this to get converted to speech
        :param output_file: save this as filename
        :return: saves audio as specified file
        """
        if len(text) == 2:
            text_ = text_processing().anti_contractions(text=text[0])  # removes contractions from text
            audio = self.client.text_to_speech.convert(
                text=str(text_),
                voice_id="pNInz6obpgDQGcFmaJgB",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                voice_settings={
                    "stability":0.3,
                    "similarity_boost":0.4,
                    "style":0.3,
                    "use_speaker_boost":True,
                    "speed":1.2
                }
            )
            save(audio, filename=output_file+"1.mp3")
        if len(text) == 2:
            text2 = text_processing().anti_contractions(text=text[1])  # removes contractions from text
        else:
            text2 = text_processing().anti_contractions(text=text[0])
        audio = self.client.text_to_speech.convert(
            text=str(text2),
            voice_id="pNInz6obpgDQGcFmaJgB",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.65,
                "similarity_boost": 0.6,
                "style": 1.0,
                "use_speaker_boost": False,
                "speed": 1.0
            }
        )
        save(audio, filename=output_file+"2.mp3")
        if len(text) == 2:
            self.concatenate(f"{output_file}.mp3", f"{output_file}1.mp3", f"{output_file}2.mp3")
        #     return (self.audio_time(f"{output_file}1.mp3"), self.audio_time(f"{output_file}2.mp3"))
        # else:
        #     return (self.audio_time(f"{output_file}1.mp3"))


    def transcribe(self, audio_path: str):
        """
        :param audio_path: filename
        :return: text, words, start time of each words , end time of each words
        """
        start_time = []
        end_time = []
        result = self.model.transcribe(audio_path, word_timestamps=True)
        text = result["text"].strip()
        text = text_processing().anti_symbols(text=text)  # removes all symbols
        for segment in result["segments"]:
            for word in segment["words"]:
                start_time.append(float(word["start"]))
                end_time.append(float(word["end"]))
        word_sep_text = text.split(" ")

        return {"text": text, "sep": word_sep_text, "s": start_time, "e": end_time}


    def audio_time(self, audio_path):
        """
        :param audio_path: filename
        :return: audio length in seconds
        """
        audio = MP3(audio_path)
        duration = audio.info.length
        return duration


    def concatenate(self, output_path, *files):
        """
        :param output_path: a specific path for output
        :param files: concatenating n number of audio files
        :return:
        """
        combined = AudioSegment.empty()

        for file in files:
            audio = AudioSegment.from_file(file)
            combined += audio
            combined.export(output_path)

    def video_audio_merge(self, audio, video, output, volume=1.0):
        """
        Merge a video file with an external audio file. If the video already contains an audio track,
        this function mixes the video's audio with the external audio (scaling the external audio's volume
        with the provided 'volume' argument). If the video does not have an audio track, the external audio
        is simply added.

        Parameters:
          audio   - Path to the external audio file.
          video   - Path to the video file.
          output  - Path for the output file.
          volume  - A scaling factor for the external audio (default is 1.0).

        This function requires FFmpeg and FFprobe.
        """

        def video_has_audio(video_path):
            """
            Use FFprobe to check if the video has an audio stream.
            Returns True if an audio stream is present, otherwise False.
            """
            try:
                cmd = [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'a',
                    '-show_entries', 'stream=codec_type',
                    '-of', 'json',
                    video_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout)
                # If any stream is returned, assume the video has audio.
                return len(data.get("streams", [])) > 0
            except Exception as e:
                # In case of error, assume no audio available
                return False

        has_audio = video_has_audio(video)

        if has_audio:
            # Prepare a filter_complex to:
            # 1. Ensure both audio streams have consistent format (sample rate, channel layout, etc.)
            # 2. Apply volume to the external audio input.
            # 3. Mix the video's original audio ([0:a]) with the external audio ([1:a]).
            #
            # Explanation of filter segments:
            # [0:a] → from the video file, formatted and with its volume unchanged (set as 1.0).
            # [1:a] → from the external audio file, with its volume scaled by the 'volume' parameter.
            # amix   → mixer filter that blends the two streams.
            filter_complex = (
                "[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,volume=1.0[a0];"
                "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,volume={}[a1];"
                "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[a]"
            ).format(volume)

            # Construct the FFmpeg command.
            cmd = [
                "ffmpeg", "-y",
                "-i", video,
                "-i", audio,
                "-filter_complex", filter_complex,
                "-map", "0:v",  # take video stream from the original video file
                "-map", "[a]",  # use the mixed audio stream from the filter
                "-c:v", "copy",  # copy the video stream without re-encoding
                "-c:a", "aac",  # encode the audio with AAC (or choose another as needed)
                output
            ]
        else:
            # If the video has no audio, simply merge the video and the external audio.
            # The '-shortest' flag ensures that the output stops when the shortest stream ends.
            cmd = [
                "ffmpeg", "-y",
                "-i", video,
                "-i", audio,
                "-c:v", "copy",  # keep the original video encoding
                "-c:a", "aac",  # encode the audio as AAC
                "-shortest",  # cut the output to the length of the shortest stream
                output
            ]

        # Execute the command.
        try:
            subprocess.run(cmd, check=True)
            print("Merging complete. Output saved to:", output)
        except subprocess.CalledProcessError as e:
            print("An error occurred while merging:", e)

    def audio_beats_detect(self, audio, ):
        pass

class movie_maker:
    def __init__(self):
        self.chaotic_add_on = {"w":"+ 100*sin(0.8*t) + 60*sin(3*t + 5) + 40*cos(1.7*t + 1.2)", "h":"+ 80*cos(0.5*t) + 50*sin(2.5*t + 3) - 30*cos(4*t + 2)"}
        self.fade_expr = (
            "if(lt(t,1), 0, if(lt(t,2), t-1, if(lt(t,4), 1, if(lt(t,5), 5-t, 0))))"
        )

    def concatenate_videos(self, output, *files):
        """
        :param output: enter output file
        :param files: video you wanna concatenate
        :return: a non audio concatenated video
        """
        if len(files) < 2:
            raise ValueError("You need at least two videos to concatenate.")

        # Read video-only inputs
        inputs = [ffmpeg.input(f) for f in files]

        # Collect video streams only
        streams = [inp.video for inp in inputs]

        # Concatenate without audio (a=0)
        joined = ffmpeg.concat(*streams, v=1, a=0).node
        video_out = joined[0]

        # Output final video
        ffmpeg.output(video_out, output).overwrite_output().run()


    def keep_duration(self, duration, filename, output_file):
        ffmpeg.input(filename, t=duration).output(output_file).run()

    def video_duration(self, filename: str):
        """

        :param filename: path name
        :return:
        """
        probe = ffmpeg.probe(filename=filename)
        duration = float(probe["format"]["duration"])
        return duration

    def video_dimensions(self, filename: str):
        probe = ffmpeg.probe(filename=filename)
        video_stream = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        if not video_stream:
            return None
        else:
            width = int(video_stream[0]['width'])
            height = int(video_stream[0]['height'])
            return (width, height)

    def overlap(self, *files: tuple, bg_video=None, output_file, limit_time=60):
        """

        :param files: a tuple with ( file, (x, y), (timings), (scale_factor))
        :return:
        """

        def create_rounded_mask(width, height, radius=50):
            img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=255)
            file_name = f"processing/mask{random.randint(0, 99999)}.png"
            img.save(file_name)
            return file_name


        if len(files) < 1:
            raise ValueError("You need at least two videos or 1 video and 1 image to overlap.")
        if bg_video != None:
            background_input = ffmpeg.input(filename=bg_video, t=limit_time)
        else:
            ffmpeg.input('color=c=black:s=1080*1920:d=60', f='lavfi').output('processing/black_bg.mp4', vcodec="libx264").run()
            background_input = ffmpeg.input(filename="processing/black_bg.mp4", t=limit_time)
        inputs = []
        # inputs.append({"input":background_input, "type":"bg"})
        for file , *rest in files:
            position = rest[0] if len(rest) > 0 else (0, 0)
            timings = rest[1] if len(rest) > 1 and rest[1][1] is not None else (0.0, self.video_duration(filename=bg_video) if bg_video != None else self.video_duration(filename="processing/black_bg.mp4"))
            scale_factor = rest[2] if len(rest) > 2 else 1
            corner = rest[3] if len(rest) > 3 else False
            video_dimensions = self.video_dimensions(filename=file)
            scale_x = float(scale_factor[0]) if isinstance(scale_factor, tuple) else float(scale_factor)
            scale_y = float(scale_factor[1]) if isinstance(scale_factor, tuple) else float(scale_factor)
            if isinstance(scale_factor, float) or isinstance(scale_factor, int):
                resized_input = ffmpeg.input(file, ss=0, t=limit_time).filter('scale', int(video_dimensions[0] * scale_x), int(video_dimensions[1] * scale_y)).filter('setpts', 'PTS-STARTPTS')
                # print(scale_x, scale_y, int(video_dimensions[0] * scale_x), int(video_dimensions[1] * scale_y))
            else:
                resized_input = ffmpeg.input(file, ss=0, t=limit_time).filter('scale', scale_x, scale_y).filter('setpts', 'PTS-STARTPTS')

            if corner:
                # resized_input.filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2').output(file_, vcodec="libx264", pix_fmt="yuv420p").run()
                mask = create_rounded_mask(width=int(video_dimensions[0] * scale_x), height=int(video_dimensions[1] * scale_y))
                # print(int(video_dimensions[0] * scale_x), int(video_dimensions[1] * scale_y))
                mask_input = ffmpeg.input(mask)
                resized_input = ffmpeg.filter([resized_input, mask_input], 'alphamerge')



            inputs.append({"input":resized_input, "pos":position, "time":timings, "type":"o"})
            #  format of inputs ---> [video, position_tuple, timings_tuple]

        #  main overlaying process
        input_0 = inputs[0]
        overlay_1 = ffmpeg.filter([background_input, input_0["input"].filter('setpts', f'PTS+{input_0["time"][0]}/TB')], "overlay", x=input_0["pos"][0], y=input_0["pos"][1], enable=f'between(t, {input_0["time"][0]},{input_0["time"][1]})')
        overlay = overlay_1
        for idx in range(1, len(inputs)):
            overlay = ffmpeg.filter([overlay, inputs[idx]["input"].filter('setpts', f'PTS+{inputs[idx]["time"][0]}/TB')], "overlay", x=inputs[idx]["pos"][0], y=inputs[idx]["pos"][1], enable=f'between(t, {inputs[idx]["time"][0]},{inputs[idx]["time"][1]})')
        ffmpeg.output(overlay, output_file).overwrite_output().run()

    def overlap_definer(self, filename, x_pos=0, y_pos=0, start_time=0, end_time=None, resize=1.0, round_corner=False):
        # return (filename, (x_pos if x_pos != None else 0, y_pos if y_pos != None else 0), (start_time if start_time != None else 0, end_time if end_time != None else None), resize if resize != None else 1)
        return (filename, (x_pos, y_pos), (start_time, end_time), resize, round_corner)
    def drawtext_definer(self, text, x_pos=0, y_pos=0, start_time=0, end_time=None, size=100, font="fonts/BubblerOne/BubblerOne-Regular.ttf", font_color="white", border=3, border_color="white"):
        return (text, (x_pos, y_pos), (start_time, end_time), {"size":size, "font":font, "color":font_color, "thickness": border, "bc":border_color})

    def draw_text(self, *text: tuple, video_file, output_file):
        """
        :param text: (sentence, (position), (timings), {font}, fade)
        :return:
        """
        #  {"size": 100, "font": "fonts/BubblerOne/BubblerOne-Regular.ttf", "color": "white", "thickness": 3, "bc": "white"}
        main_vid_input = ffmpeg.input(video_file)
        for sentence, *rest in text:
            position = rest[0] if len(rest) > 0 else (0, 0)
            timings = rest[1] if len(rest) > 1 and rest[1][1] is not None else (0.0, self.video_duration(filename=video_file))
            font = rest[2] if len(rest) > 2 else {"size":100, "font":"fonts/BubblerOne/BubblerOne-Regular.ttf", "color":"white", "thickness": 3, "bc":"white"}
            # fade = rest[3] if len(rest) > 3 else False
            # fade = (f"if(lt(t,{timings[0]}), 0, if(lt(t,{timings[0]+1}), t-1, if(lt(t,{timings[0]+1}), 1, if(lt(t,{timings[1]}), 5-t, 0))))") if fade is True else ""
            main_vid_input = main_vid_input.filter('drawtext', text=str(sentence), fontfile=font["font"], fontcolor=font["color"], fontsize=font["size"], borderw=font["thickness"], bordercolor=font["bc"], x=position[0], y=position[1], enable=f'between(t, {timings[0]}, {timings[1]})')
        ffmpeg.output(main_vid_input, output_file).run()
        #  clear text


class downloader:
    def __init__(self):
        self.gif_api = "pNRWCCCquU9Y0TErzJh9R8DDhZebwNw9"

    def download_gif(self, query):
        """
        :param query:  well explained gif query to be provided
        :return: gif filename list
        """
        LIMIT = 1

        url = f"https://api.giphy.com/v1/gifs/search?api_key={self.gif_api}&q={query}&limit={LIMIT}&rating=pg-13"

        response = requests.get(url)
        gif_url = response.json()["data"][random.randint(0, LIMIT - 1)]["images"]["original"]["url"]
        gif_response = requests.get(gif_url)
        file_ = f"processing/gif{random.randint(0,9999)}.gif"
        system().write(data=gif_response.content, file=file_, type_="wb")

        return file_

    def image_download(self, query):
        with DDGS() as ddgs:
            results = ddgs.images(keywords=query)
            image_results = results
            for _find in image_results:
                if _find["image"][-3:] in ["jpg", "png"]:
                    image_results = _find["image"]
                    break
            if image_results:
                img_url = image_results
                if img_url.lower().endswith(".jpg") or img_url.lower().endswith(".png"):
                    response = requests.get(img_url)
                    file_ = f"processing/image{random.randint(0, 9999)}"
                    # system().write(data=response.content, file=file_, type_="wb")
                    image = Image.open(BytesIO(response.content))
                    image.save(file_+".gif", format="GIF", save_all=True, loop=0, duration=1000)
                    input_gif = ffmpeg.input(file_+".gif", r=1)
                    input_gif = input_gif.filter("pad", "ceil(iw/2)*2", "ceil(ih/2)*2")
                    main_file = file_+f"{random.randint(0, 9999)}.mp4"
                    input_gif.output(main_file, vcodec="libx264", pix_fmt="yuv420p").run()
                    return main_file




    def smart_download(self, query_list, context_pos_tagged=None):

        visual_database = []
        for query in query_list:
            try:
                visual_database.append(self.image_download(query))
            except:
                try:
                    visual_database.append(self.download_gif(query))
                except:
                    pass

        return visual_database



class youtube_integrator:
    def generate_high_quality_tags(self, type_):
        """
        :param type_: type = title, discription, tags
        :return:
        """
        def hashtag_convert(tag):
            return "#"+str(tag)
        visibility_tags = ["shorts", "viral", "trending", "fyp"]
        amplification_tags = ["viralshorts", "subscribe", "reels"]
        entertainment_tags = ["shortscomedy", "shortsfunny", "Shortsjokes", "Shortshumor"]
        high_engaging_niche_tags = ["Gaming", "shortsGTA5", "shortsfortnite"]
        challenge_tags = ["shortschallenge", "shortscompetition"]
        context_tags = ["facts", "fact", "didyouknow", "truefacts", "amazingfacts", "interestingfacts", "factsdaily", "realfacts", "generalknowledge", "sciencefacts", "funfacts"]

        final_tag = []
        if type_ == "title":
            for generic in visibility_tags:
                final_tag.append(hashtag_convert(generic))
            final_tag.append(hashtag_convert(amplification_tags[1]))
            final_tag.append(hashtag_convert(high_engaging_niche_tags[0]))
            final_tag.append(hashtag_convert(context_tags[0]))
            final = " ".join(final_tag)
        elif type_ == "discription":
            all_tags = [*visibility_tags, *amplification_tags, *entertainment_tags, *high_engaging_niche_tags, *challenge_tags, *context_tags]
            x = []
            for tag in all_tags:
                x.append(hashtag_convert(tag))
            final = " ".join(x)
        else:
            all_tags = [*visibility_tags, *amplification_tags, *entertainment_tags, *high_engaging_niche_tags, *challenge_tags, *context_tags]
            final = ", ".join(all_tags)
        return final

    def download_video_section(self, url, start_time, end_time, output_path="bgv/downloaded vids/"+"%(title)s.%(ext)s"):
        ydl_opts = {
            'format': 'bestvideo[height<=720]',
            'outtmpl': output_path,
            'merge_output_format': 'mp4',
            'download_sections': [f'*{start_time}-{end_time}'],
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])






# print(text_processing().query_generate("wow my name is raakin khan and guess what shrimps can see colors we can not even imagine yet they still eat garbage off the ocean floor")
# )
# print(youtube_integrator().generate_high_quality_tags(type_="discription"))

#ffmpeg.input("processing/image8483.gif").output("processing/text12.mp4", vcodec="libx264", pix_fmt="yuv420p").run()
# downloader().image_download(query="iphone 13")
# m = movie_maker()
# m.overlap(m.overlap_definer(filename="processing/image29261159.mp4", round_corner=True, resize=0.3),  bg_video="bgv/gtaminecraft.mp4", output_file="processing/test.mp4",limit_time=5)
# youtube_integrator().download_video_section(url="https://www.youtube.com/watch?v=KFv-mbPgaG4", start_time='00:00:02', end_time="00:01:02")
# audio_handler().video_audio_merge(audio="processing/audio.mp3", video="videos/final_output3525.mp4", output="processing/music_fit1.mp4", volume=0.9)