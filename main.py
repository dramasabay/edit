import streamlit as st
import os, subprocess, math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from moviepy.editor import VideoFileClip, AudioFileClip, vfx
from moviepy.video.fx.all import resize
from moviepy.audio.fx.all import audio_normalize
from io import BytesIO
import base64

# ================== Small utils ==================
def run_cmd(args):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def detect_nvenc():
    out = run_cmd(["ffmpeg","-hide_banner","-encoders"]).stdout
    return " h264_nvenc " in (" " + out + " ")

def ffprobe_val(args):
    res = run_cmd(args)
    return res.stdout.strip() if res.returncode == 0 else ""

def src_props(path):
    # width,height,fps,duration,video_kbps
    w = ffprobe_val(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width","-of","csv=p=0",path]) or "0"
    h = ffprobe_val(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=height","-of","csv=p=0",path]) or "0"
    r = ffprobe_val(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=r_frame_rate","-of","csv=p=0",path]) or "30/1"
    dur = ffprobe_val(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",path]) or "0"
    br  = ffprobe_val(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=bit_rate","-of","csv=p=0",path]) or "0"
    try:
        num, den = r.split("/")
        fps = float(num)/float(den) if float(den)!=0 else float(num)
    except:
        fps = 30.0
    try:
        kbps = max(300, int(int(br)/1000)) if br.isdigit() else 0
    except:
        kbps = 0
    return int(w), int(h), float(fps), float(dur), int(kbps)

def make_even(x:int)->int:
    return x if x%2==0 else x-1

def map_nvenc_preset(ui:str)->str:
    # NVENC wants p1..p7 (p7 fastest). Map human presets.
    table={"ultrafast":"p7","superfast":"p6","veryfast":"p5","faster":"p4","fast":"p3","medium":"p2"}
    return table.get(str(ui).lower(),"p5")

def label_to_res(label, src_w, src_h, orientation):
    if label=="Auto (match source)":
        return src_w, src_h
    if orientation=="Portrait":
        mapping={"576p":(720,576),"720p":(720,1280),"FHD/1080p":(1080,1920),"2K":(1440,2560),"4K":(2160,3840)}
    else:
        mapping={"576p":(1024,576),"720p":(1280,720),"FHD/1080p":(1920,1080),"2K":(2560,1440),"4K":(3840,2160)}
    return mapping.get(label,(src_w,src_h))

# -------- math for cover/fit after rotation ----------
def cover_scale_for_rotation(src_w, src_h, out_w, out_h, angle_deg):
    """
    Return scale factor S applied BEFORE rotate so that after rotation the frame fully
    covers out_w x out_h (no borders). Uses exact rectangle-rotation bounds.
    """
    a = abs(angle_deg) * math.pi / 180.0
    cos, sin = abs(math.cos(a)), abs(math.sin(a))
    # rotated bounds from source WxH
    w_rot = src_w * cos + src_h * sin
    h_rot = src_w * sin + src_h * cos
    if w_rot == 0 or h_rot == 0:
        return 1.0
    s1 = out_w / w_rot
    s2 = out_h / h_rot
    return max(s1, s2)

def fit_scale_for_rotation(src_w, src_h, out_w, out_h, angle_deg):
    """
    'Fit' (letterbox) scaling to ensure all content visible after rotation.
    You may see small borders.
    """
    a = abs(angle_deg) * math.pi / 180.0
    cos, sin = abs(math.cos(a)), abs(math.sin(a))
    w_rot = src_w * cos + src_h * sin
    h_rot = src_w * sin + src_h * cos
    if w_rot == 0 or h_rot == 0:
        return 1.0
    s1 = out_w / w_rot
    s2 = out_h / h_rot
    return min(s1, s2)

# ================ FFmpeg (Pro) renderer ================
def build_ffmpeg_cmd(
    inp, outp, start, end, src_w, src_h, out_w, out_h, fps_out,
    mirror=False, rotate_deg=0.0, user_zoom=1.0, fill_mode="cover",
    use_nvenc=False, nvenc_preset="p5",
    quality_mode="Match source bitrate", source_kbps=5000,
    crf=18, threads=os.cpu_count() or 4, ui_preset="veryfast"
):
    # compute scale pre-rotate to ensure no borders (cover), or to fit
    if fill_mode == "cover":
        base_scale = cover_scale_for_rotation(src_w, src_h, out_w, out_h, rotate_deg)
    else:
        base_scale = fit_scale_for_rotation(src_w, src_h, out_w, out_h, rotate_deg)
    base_scale *= max(0.0001, user_zoom)  # include user's extra zoom

    # build filters: scale -> (optional mirror) -> rotate -> crop/scale to exact -> format
    sc_w = max(2, int(round(src_w * base_scale)))
    sc_h = max(2, int(round(src_h * base_scale)))

    filters = [f"scale={sc_w}:{sc_h}:flags=lanczos"]
    if mirror:
        filters.append("hflip")

    if abs(rotate_deg) > 1e-3:
        angle = rotate_deg * math.pi / 180.0
        filters.append(f"rotate={angle}:fillcolor=black@1:bilinear=1")

    # center-crop to out_w/out_h to guarantee no borders in 'cover' mode
    # for 'fit', we scale down to exact size (could add letterbox; here we still crop center)
    filters.append(f"crop={out_w}:{out_h}")

    # final format
    filters.append("format=yuv420p")
    vf = ",".join(filters)

    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error"]
    if start>0: cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", inp]
    if end>start: cmd += ["-to", f"{end:.3f}"]
    cmd += ["-vf", vf]

    if fps_out>0:
        cmd += ["-r", str(int(round(fps_out)))]

    if use_nvenc:
        cmd += ["-c:v","h264_nvenc","-rc","vbr_hq","-preset", nvenc_preset]
        if quality_mode=="Match source bitrate":
            target = max(600, int(source_kbps))
            maxrate, bufsize = int(target*1.3), int(target*2.0)
            cmd += ["-b:v", f"{target}k","-maxrate",f"{maxrate}k","-bufsize",f"{bufsize}k"]
        else:
            cmd += ["-cq", str(int(crf))]
    else:
        cmd += ["-c:v","libx264","-preset", ui_preset, "-profile:v","high"]
        if quality_mode=="Match source bitrate":
            target = max(600, int(source_kbps))
            maxrate, bufsize = int(target*1.3), int(target*2.0)
            cmd += ["-b:v", f"{target}k","-maxrate",f"{maxrate}k","-bufsize",f"{bufsize}k"]
        else:
            cmd += ["-crf", str(int(crf))]

    cmd += ["-c:a","aac","-b:a","192k","-movflags","+faststart","-threads", str(threads), outp]
    return cmd

# ================ MoviePy (Studio) helpers ================
def change_voice(audio_clip, effect, speed=1.0):
    from moviepy.audio.AudioClip import CompositeAudioClip
    if effect == "None":
        return audio_clip.fx(audio_normalize).fx(vfx.speedx, speed)
    elif effect == "Pitch Up (Chipmunk)":
        return audio_clip.fx(audio_normalize).fx(vfx.speedx, 1.2).fx(vfx.speedx, speed/1.2)
    elif effect == "Pitch Down (Darth Vader)":
        return audio_clip.fx(audio_normalize).fx(vfx.speedx, 0.8).fx(vfx.speedx, speed/0.8)
    elif effect == "Robot":
        a1 = audio_clip.fx(audio_normalize).fx(vfx.speedx, speed)
        a2 = a1.volumex(0.5).fx(vfx.speedx, 1.03)
        a3 = a1.volumex(0.5).fx(vfx.speedx, 0.97)
        return CompositeAudioClip([a1, a2, a3])
    else:
        return audio_clip

def apply_adjustments(img, temp,tint,saturation,exposure,contrast,shadow,highlight,whites,blacks,brilliance,sharpen,clarity,fade,vignette):
    im = img.convert("RGB")
    if temp != 0:
        r, g, b = im.split()
        if temp > 0:
            b = b.point(lambda x: x - temp if x - temp > 0 else 0)
        else:
            r = r.point(lambda x: x + abs(temp) if x + abs(temp) < 255 else 255)
        im = Image.merge('RGB', (r, g, b))
    if tint != 0:
        r, g, b = im.split()
        if tint > 0:
            g = g.point(lambda x: x - tint if x - tint > 0 else 0)
        else:
            r = r.point(lambda x: x + abs(tint) if x + abs(tint) < 255 else 255)
            b = b.point(lambda x: x + abs(tint) if x + abs(tint) < 255 else 255)
        im = Image.merge('RGB', (r, g, b))
    if saturation != 0:
        im = ImageEnhance.Color(im).enhance(1 + saturation / 50.0)
    if exposure != 0:
        im = ImageEnhance.Brightness(im).enhance(1 + exposure / 100.0)
    if contrast != 0:
        im = ImageEnhance.Contrast(im).enhance(1 + contrast / 100.0)
    np_img = np.array(im).astype(np.int16)
    if shadow != 0:
        mask = np_img < 128
        np_img[mask] = np.clip(np_img[mask] + shadow, 0, 255)
    if highlight != 0:
        mask = np_img > 128
        np_img[mask] = np.clip(np_img[mask] + highlight, 0, 255)
    if whites != 0:
        np_img = np.clip(np_img + (whites if whites > 0 else 0), 0, 255)
    if blacks != 0:
        np_img = np.clip(np_img - (abs(blacks) if blacks < 0 else 0), 0, 255)
    if brilliance != 0:
        np_img = np.clip(np_img + brilliance, 0, 255)
    im = Image.fromarray(np_img.astype(np.uint8))
    if sharpen > 0:
        im = im.filter(ImageFilter.UnsharpMask(percent=150, radius=2, threshold=3))
    if clarity > 0:
        im = ImageEnhance.Contrast(im).enhance(1 + clarity / 40.0)
    if fade > 0:
        fade_img = Image.new("RGB", im.size, (255, 255, 255))
        im = Image.blend(im, fade_img, fade / 10.0)
    # vignette omitted for speed
    return im

def adjust_frame_cover(frame, src_w, src_h, out_w, out_h, mirror, user_zoom, rotate_deg, **adj):
    # compute cover zoom before rotate
    s = cover_scale_for_rotation(src_w, src_h, out_w, out_h, rotate_deg) * max(0.0001, user_zoom)
    im = Image.fromarray(frame)
    if mirror: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    # pre-scale
    pre_w = max(2, int(round(src_w * s)))
    pre_h = max(2, int(round(src_h * s)))
    im = im.resize((pre_w, pre_h), resample=Image.LANCZOS)
    # rotate
    im = im.convert("RGBA").rotate(-rotate_deg, expand=True, resample=Image.BILINEAR, fillcolor=(0,0,0,255))
    # center-crop to out
    x = (im.width - out_w) // 2
    y = (im.height - out_h) // 2
    im = im.crop((max(0,x), max(0,y), max(0,x)+out_w, max(0,y)+out_h))
    # adjustments
    im = apply_adjustments(im, **adj)
    return np.array(im)

# ================== UI ==================
st.set_page_config(page_title="Realtime CapCut-Style Editor", layout="wide")
st.markdown("<h2 style='text-align:center;'>🎬 CapCut-Style Batch Editor (Per-File Trim • Smooth Rotate • No Borders)</h2>", unsafe_allow_html=True)

if "reset" not in st.session_state: st.session_state.reset=False
col_preview, col_controls = st.columns([1,2], gap="large")

with col_controls:
    if st.button("🧹 Clear All"): st.session_state.reset=True
    if st.session_state.reset:
        defaults = dict(
            orientation="Landscape", res="Auto (match source)", fps="Auto",
            mirror=False, scale_factor=1.0, rotate_deg=0.0, fill_mode="Cover (no borders)",
            speed_option=1.0, voice_changer="None", cut_mode="Per-file",
            encoder_preset="veryfast", quality_mode="Match source bitrate", quality_crf=18,
            audio_volume=80, temp=0, tint=0, saturation=0, exposure=0, contrast=0, highlight=0,
            shadow=0, whites=0, blacks=0, brilliance=0, sharpen=0, clarity=0, fade=0, vignette=0
        )
        for k,v in defaults.items(): st.session_state[k]=v
        st.session_state.reset=False

    folder = st.text_input("📁 Video folder", value="videos/")
    if not os.path.isdir(folder):
        st.warning("Enter a valid folder with videos (.mp4, .mov, .avi, .mkv)."); st.stop()

    files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".mp4",".mov",".avi",".mkv"))])
    if not files: st.warning("No videos found!"); st.stop()
    st.write(f"Found {len(files)} videos.")

    sel = st.selectbox("Select video for preview", files, index=0)
    sel_path = os.path.join(folder, sel)
    sw, sh, sfps, sdur, skbps = src_props(sel_path)

    st.markdown("---")
    orientation = st.radio("Export orientation (when NOT Auto resolution)", ["Landscape","Portrait"], horizontal=True, key="orientation")
    res_label = st.radio("Resolution", ["Auto (match source)","576p","720p","FHD/1080p","2K","4K"], horizontal=True, key="res")
    fps_choice = st.radio("Export Frame Rate (fps)", ["Auto",30,60], horizontal=True, key="fps")
    fill_mode = st.radio("Fill mode", ["Cover (no borders)","Fit (show full, may letterbox)"], horizontal=True, key="fill_mode")

    mirror = st.checkbox("Mirror (Flip left/right)", key="mirror")
    scale_factor = st.slider("Extra Zoom", 0.5, 5.0, st.session_state.get("scale_factor",1.0), 0.01, key="scale_factor")
    rotate_deg = st.number_input("Rotate (degrees)", -180.0, 180.0, st.session_state.get("rotate_deg",0.0), step=1.0, key="rotate_deg")
    speed_option = st.slider("Video speed", 0.25, 3.0, st.session_state.get("speed_option", 1.0), 0.05, key="speed_option")
    voice_changer = st.selectbox("Voice changer", ["None","Pitch Up (Chipmunk)","Pitch Down (Darth Vader)","Robot"], key="voice_changer")

    gpu_ok = detect_nvenc()
    dev = st.radio("Export using", ["Auto","CPU"] + (["GPU"] if gpu_ok else []), index=0)
    ui_preset = st.selectbox("Encoder preset", ["ultrafast","superfast","veryfast","faster","fast","medium"], index=2)
    quality_mode = st.selectbox("Quality mode", ["Match source bitrate","High quality CRF 18","Manual CRF"], index=0, key="quality_mode")
    quality_crf = st.slider("CRF/CQ (for CRF modes)", 14, 28, st.session_state.get("quality_crf",18), key="quality_crf")
    threads = st.slider("Threads", 1, os.cpu_count() or 4, os.cpu_count() or 4)

    # Trim UI
    cut_mode = st.radio("Trim mode", ["Global","Per-file"], horizontal=True, key="cut_mode")
    if cut_mode=="Global":
        cut = st.radio("Quick cut", ["Full","0:20","0:30","1:00","Custom"], horizontal=True)
        if cut=="Full": g_start, g_end = 0.0, sdur
        elif cut=="0:20": g_start, g_end = 0.0, min(20.0, sdur)
        elif cut=="0:30": g_start, g_end = 0.0, min(30.0, sdur)
        elif cut=="1:00": g_start, g_end = 0.0, min(60.0, sdur)
        else:
            g_start = st.number_input("Start (sec)", 0.0, float(sdur), 0.0)
            g_end   = st.number_input("End (sec)",   0.0, float(sdur), float(sdur))
        df = None
    else:
        rows=[]
        for f in files:
            p=os.path.join(folder,f); w,h,fps,dur,_=src_props(p)
            rows.append({"file":f,"start":0.0,"end":round(dur,3)})
        df = st.data_editor(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                            column_config={"file":st.column_config.TextColumn("File",disabled=True),
                                           "start":st.column_config.NumberColumn("Start (s)",min_value=0.0,step=0.1),
                                           "end":st.column_config.NumberColumn("End (s)",min_value=0.0,step=0.1)})
        g_start=g_end=None

    # Audio overlay (kept)
    audios = st.file_uploader("Audio (mp3/wav, optional)", type=["mp3","wav"], accept_multiple_files=True)
    audio_volume = st.slider("Audio overlay volume (%)", 0, 100, st.session_state.get("audio_volume",80), key="audio_volume")

    # Adjustments (kept)
    st.markdown("### Adjustments")
    temp = st.slider("Temp", -100, 100, st.session_state.get("temp",0), key="temp")
    tint = st.slider("Tint", -100, 100, st.session_state.get("tint",0), key="tint")
    saturation = st.slider("Saturation", -100, 100, st.session_state.get("saturation",0), key="saturation")
    exposure = st.slider("Exposure", -100, 100, st.session_state.get("exposure",0), key="exposure")
    contrast = st.slider("Contrast", -100, 100, st.session_state.get("contrast",0), key="contrast")
    highlight = st.slider("Highlight", -100, 100, st.session_state.get("highlight",0), key="highlight")
    shadow = st.slider("Shadow", -100, 100, st.session_state.get("shadow",0), key="shadow")
    whites = st.slider("Whites", -100, 100, st.session_state.get("whites",0), key="whites")
    blacks = st.slider("Blacks", -100, 100, st.session_state.get("blacks",0), key="blacks")
    brilliance = st.slider("Brilliance", -100, 100, st.session_state.get("brilliance",0), key="brilliance")
    sharpen = st.slider("Sharpen", 0, 10, st.session_state.get("sharpen",0), key="sharpen")
    clarity = st.slider("Clarity", 0, 40, st.session_state.get("clarity",0), key="clarity")
    fade = st.slider("Fade", 0, 10, st.session_state.get("fade",0), key="fade")
    vignette = st.slider("Vignette", 0, 10, st.session_state.get("vignette",0), key="vignette")

    output_prefix = st.text_input("Output filename prefix", "reel_edit_")

# ============== Preview (quick) ==============
with col_preview:
    if os.path.exists(sel_path):
        st.video(sel_path)  # show source for reference
        # also one processed frame (cover math)
        with VideoFileClip(sel_path) as sc:
            t = min(max(0.01, sc.duration/2.0), sc.duration-0.01)
            frame = sc.get_frame(t)
        out_w, out_h = label_to_res(st.session_state.get("res"), sw, sh, st.session_state.get("orientation"))
        out_w, out_h = make_even(out_w), make_even(out_h)
        processed = adjust_frame_cover(
            frame, sw, sh, out_w, out_h,
            st.session_state.get("mirror",False),
            st.session_state.get("scale_factor",1.0),
            st.session_state.get("rotate_deg",0.0),
            temp=temp,tint=tint,saturation=saturation,exposure=exposure,contrast=contrast,
            shadow=shadow,highlight=highlight,whites=whites,blacks=blacks,brilliance=brilliance,
            sharpen=sharpen,clarity=clarity,fade=fade,vignette=vignette
        )
        st.image(processed, caption="Preview (processed frame, Cover mode)")

# ================= Process =================
with col_controls:
    if st.button("🚀 Process (Trim Each Video)"):
        gpu_available = detect_nvenc()
        use_nvenc = True if (dev=="GPU" or (dev=="Auto" and gpu_available)) else False
        nv_preset = map_nvenc_preset(ui_preset)
        out_dir = os.path.join(folder,"output"); Path(out_dir).mkdir(parents=True, exist_ok=True)

        trims = ([(f, g_start, g_end) for f in files] if st.session_state.get("cut_mode")=="Global"
                 else [(row["file"], float(row["start"]), float(row["end"])) for _,row in df.iterrows()])
        total = len(trims)
        prog = st.progress(0.0, text="Starting…")

        for i,(fname, s, e) in enumerate(trims, start=1):
            in_path = os.path.join(folder, fname)
            sw, sh, sfps, sdur, skbps = src_props(in_path)
            if s>=e or s>=sdur:
                st.warning(f"Skip {fname}: invalid trim")
                prog.progress(i/total, text=f"Skipped {fname}")
                continue

            # target res & fps
            tw, th = label_to_res(res_label, sw, sh, st.session_state.get("orientation"))
            tw, th = make_even(tw), make_even(th)
            fps_out = sfps if st.session_state.get("fps")=="Auto" else int(st.session_state.get("fps"))
            fill = "cover" if st.session_state.get("fill_mode").startswith("Cover") else "fit"

            # decide renderer: if any extras are on, keep MoviePy path; else Pro FFmpeg
            studio_needed = (
                (abs(speed_option - 1.0) > 1e-3) or
                (voice_changer != "None") or
                (audios and len(audios) > 0) or
                any([temp,tint,saturation,exposure,contrast,shadow,highlight,whites,blacks,brilliance,sharpen,clarity,fade,vignette])
            )

            stem = Path(fname).stem
            out_name = f"{output_prefix}{stem}_trim_{int(s)}-{int(e)}_{tw}x{th}_{int(round(fps_out))}fps.mp4"
            out_path = os.path.join(out_dir, out_name)

            if not studio_needed:
                # -------- Pro (FFmpeg) – smooth + no borders ----------
                cmd = build_ffmpeg_cmd(
                    inp=in_path, outp=out_path, start=s, end=e,
                    src_w=sw, src_h=sh, out_w=tw, out_h=th, fps_out=fps_out,
                    mirror=mirror, rotate_deg=rotate_deg, user_zoom=scale_factor,
                    fill_mode=fill, use_nvenc=use_nvenc, nvenc_preset=nv_preset,
                    quality_mode=quality_mode, source_kbps=(skbps or 5000),
                    crf=int(quality_crf), threads=os.cpu_count() or 4, ui_preset=ui_preset
                )
                res = run_cmd(cmd)
                if res.returncode != 0 and use_nvenc:
                    # fallback to libx264
                    cmd = build_ffmpeg_cmd(
                        inp=in_path, outp=out_path, start=s, end=e,
                        src_w=sw, src_h=sh, out_w=tw, out_h=th, fps_out=fps_out,
                        mirror=mirror, rotate_deg=rotate_deg, user_zoom=scale_factor,
                        fill_mode=fill, use_nvenc=False, nvenc_preset="p5",
                        quality_mode=quality_mode, source_kbps=(skbps or 5000),
                        crf=int(quality_crf), threads=os.cpu_count() or 4, ui_preset=ui_preset
                    )
                    res = run_cmd(cmd)
                if res.returncode != 0:
                    st.error(f"FFmpeg failed for {fname}:\n{res.stderr}")
                else:
                    st.success(f"Saved (Pro): {out_name}")
            else:
                # -------- Studio (MoviePy) – old functions kept, with no-border math ----------
                with VideoFileClip(in_path) as clip:
                    c_end = min(e, clip.duration)
                    sub = clip.subclip(s, c_end)

                    # per-frame cover
                    sub = sub.fl_image(lambda f: adjust_frame_cover(
                        f, sw, sh, tw, th, mirror, scale_factor, rotate_deg,
                        temp=temp,tint=tint,saturation=saturation,exposure=exposure,contrast=contrast,
                        shadow=shadow,highlight=highlight,whites=whites,blacks=blacks,brilliance=brilliance,
                        sharpen=sharpen,clarity=clarity,fade=fade,vignette=vignette
                    ))
                    if abs(speed_option - 1.0) > 1e-3:
                        sub = sub.fx(vfx.speedx, speed_option)
                    sub = sub.fx(resize, newsize=(tw, th))

                    # audio – voice changer + overlays
                    if sub.audio is not None and voice_changer != "None":
                        sub = sub.set_audio(change_voice(sub.audio, voice_changer, speed_option))

                    if audios:
                        from moviepy.audio.AudioClip import CompositeAudioClip
                        audioclips = []
                        for audio_file in audios:
                            audio_path = f"temp_{audio_file.name}"
                            with open(audio_path, "wb") as af: af.write(audio_file.read())
                            aclip = AudioFileClip(audio_path).volumex(st.session_state.get("audio_volume",80)/100.0)
                            aclip = aclip.set_duration(sub.duration)
                            audioclips.append(aclip)
                        clips_to_mix = [sub.audio] if sub.audio else []
                        clips_to_mix += audioclips
                        if clips_to_mix:
                            audio_mix = CompositeAudioClip(clips_to_mix)
                            sub = sub.set_audio(audio_mix)

                    # write args – universal playback
                    common_ff = ["-pix_fmt","yuv420p","-movflags","+faststart"]
                    write_kwargs = dict(
                        audio_codec="aac",
                        audio_bitrate="192k",
                        verbose=False,
                        logger=None,
                        threads=os.cpu_count() or 4,
                        fps=int(round(fps_out))
                    )

                    if use_nvenc:
                        write_kwargs["codec"] = "h264_nvenc"
                        write_kwargs["ffmpeg_params"] = ["-rc","vbr_hq","-cq","18","-preset", map_nvenc_preset(ui_preset)] + common_ff
                    else:
                        write_kwargs["codec"] = "libx264"
                        write_kwargs["preset"] = ui_preset
                        write_kwargs["ffmpeg_params"] = ["-crf","18","-profile:v","high"] + common_ff

                    try:
                        sub.write_videofile(out_path, **write_kwargs)
                    except Exception as err:
                        if use_nvenc:
                            # fallback to libx264
                            write_kwargs["codec"] = "libx264"
                            write_kwargs["preset"] = ui_preset
                            write_kwargs["ffmpeg_params"] = ["-crf","18","-profile:v","high"] + common_ff
                            sub.write_videofile(out_path, **write_kwargs)
                        else:
                            raise
                    finally:
                        sub.close()
                st.success(f"Saved (Studio): {out_name}")

            prog.progress(i/total, text=f"Processed {i}/{total}: {fname}")

        prog.progress(1.0, text="Finished")
        st.balloons()
        st.success("All videos processed! No border when rotating (Cover mode).")
