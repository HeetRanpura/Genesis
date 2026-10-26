"""Render the 'Your Phone Is Stealing Your Life' YouTube Short (1080x1920, 30fps)."""
import math, os, random, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "short.mp4")
PREVIEW = os.environ.get("PREVIEW")  # comma list of seconds -> write PNGs only
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS, SR = 1080, 1920, 30, 44100
random.seed(7)

FONTS = "/System/Library/Fonts/Supplemental/"
IMPACT = FONTS + "Impact.ttf"
BLACK = FONTS + "Arial Black.ttf"
HELV = "/System/Library/Fonts/Helvetica.ttc"
EMOJI = "/System/Library/Fonts/Apple Color Emoji.ttc"
YELLOW, WHITE, RED, GREEN, BLUE = (255, 214, 10), (255, 255, 255), (255, 45, 85), (48, 209, 88), (10, 132, 255)

_fc = {}
def font(path, size):
    k = (path, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]

_ec = {}
def emoji(ch, size):
    k = (ch, size)
    if k not in _ec:
        base = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(base).text((100, 100), ch, font=font(EMOJI, 160), embedded_color=True, anchor="mm")
        base = base.crop(base.getbbox())
        s = size / max(base.size)
        _ec[k] = base.resize((max(1, int(base.width * s)), max(1, int(base.height * s))), Image.LANCZOS)
    return _ec[k]

def paste_c(dst, im, cx, cy, rot=0, alpha=1.0):
    if rot:
        im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 1:
        im = im.copy(); im.putalpha(im.getchannel("A").point(lambda a: int(a * alpha)))
    dst.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

def ease(t): t = min(max(t, 0), 1); return 1 - (1 - t) ** 3
def clamp(t): return min(max(t, 0), 1)

# ---------------------------------------------------------------- timeline
LINES = [l.strip() for l in open(os.path.join(HERE, "lines.txt")) if l.strip()]
SECTION_STARTS = {3, 6, 10, 14}  # 0-based phrase index that begins a new section
def wav_read(p):
    w = wave.open(p); a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return a
VO = [wav_read(os.path.join(HERE, "vo", f"{i+1:02d}.wav")) for i in range(len(LINES))]
starts, t = [], 0.55
for i, a in enumerate(VO):
    if i: t += 0.3 if i in SECTION_STARTS else 0.08
    starts.append(t); t += len(a) / SR
ends = [s + len(a) / SR for s, a in zip(starts, VO)]
DUR = ends[-1] + 0.55
NFR = int(DUR * FPS)
scene_t0 = [0.0] + starts[1:]
scene_t1 = starts[1:] + [DUR]
def section_of(i): return sum(1 for s in SECTION_STARTS if i >= s)
GLOW = [(110, 60, 255), (10, 132, 255), (255, 35, 70), (40, 200, 90), (255, 190, 0)]

# ---------------------------------------------------------------- captions
KEY = {"STOP", "SCROLLING", "STEALING", "YEARS", "LIFE", "REALIZE", "FIVE", "MINUTES", "TEN", "ENTIRE",
       "HOUR", "GONE", "SCARY", "NOTIFICATION", "BRAIN", "AGAIN", "HABIT", "TODAY", "OFF", "SCREEN", "TIME",
       "SHOCKED", "HONEST", "ONE", "VIDEO"}
def chunks_for(i):
    words = LINES[i].upper().replace("...", "…").split()
    groups, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) == 3 or len(" ".join(cur)) >= 13 or w[-1] in ",.!?:…":
            groups.append(cur); cur = []
    if cur: groups.append(cur)
    weights = [sum(len(w) for w in g) + 2 + (3 if g[-1][-1] in ",.!?:…" else 0) for g in groups]
    span = ends[i] - starts[i]; out, t = [], starts[i]
    for g, wt in zip(groups, weights):
        d = span * wt / sum(weights); out.append((t - 0.04, g)); t += d
    return out
CHUNKS = [c for i in range(len(LINES)) for c in chunks_for(i)]

_cap = {}
def caption_img(words):
    k = tuple(words)
    if k in _cap: return _cap[k]
    size = 132
    while True:
        f = font(IMPACT, size)
        widths = [f.getlength(w) for w in words]; sp = f.getlength(" ")
        if sum(widths) + sp * (len(words) - 1) < 940 or size < 70: break
        size -= 6
    tw = int(sum(widths) + sp * (len(words) - 1)) + 60
    im = Image.new("RGBA", (tw, size + 80), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    x = 30
    for w, ww in zip(words, widths):
        bare = "".join(c for c in w if c.isalpha())
        col = YELLOW if bare in KEY else WHITE
        d.text((x + 6, 48), w, font=f, fill=(0, 0, 0, 160), anchor="lm")  # drop shadow
        d.text((x, 40), w, font=f, fill=col, anchor="lm", stroke_width=11, stroke_fill=(0, 0, 0))
        x += ww + sp
    _cap[k] = im
    return im

def draw_caption(fr, t):
    cur = None
    for ct, g in CHUNKS:
        if ct <= t: cur = (ct, g)
    if cur is None or t > ends[-1] + 0.4: return
    ct, g = cur
    im = caption_img(g); p = clamp((t - ct) / 0.13)
    s = 1.22 - 0.22 * ease(p)
    if s != 1: im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
    paste_c(fr, im, W / 2, 1330)

# ---------------------------------------------------------------- backgrounds
def make_bg(col):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / W) ** 2 + ((y - 760) / (H * 0.8)) ** 2)
    g = np.clip(1 - r * 1.9, 0, 1) ** 2 * 0.42
    base = np.array([9, 9, 13], np.float32)
    arr = base + g[..., None] * np.array(col, np.float32)
    vig = np.clip(1 - (np.sqrt(((x - W / 2) / W) ** 2 + ((y - H / 2) / H) ** 2) - 0.35) * 1.3, 0.35, 1)
    arr = arr * vig[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)).convert("RGBA")
BGS = [make_bg(c) for c in GLOW]

# ---------------------------------------------------------------- props
def rrect(d, box, r, **kw): d.rounded_rectangle(box, r, **kw)

FEED_COLS = [(255, 94, 58), (90, 200, 250), (255, 204, 0), (175, 82, 222), (52, 199, 89), (255, 55, 95), (94, 92, 230)]
def feed(scroll, w=520, h=880):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 255)); d = ImageDraw.Draw(im)
    card = 330; first = int(scroll // card)
    for k in range(first - 1, first + 5):
        y = k * card - scroll
        c = FEED_COLS[k % len(FEED_COLS)]
        rrect(d, (24, y + 20, w - 24, y + 230), 22, fill=c)
        d.polygon([(w / 2 - 30, y + 95), (w / 2 - 30, y + 155), (w / 2 + 28, y + 125)], fill=(255, 255, 255, 230))
        d.ellipse((24, y + 248, 74, y + 298), fill=(80, 80, 88))
        rrect(d, (90, y + 252, w - 120, y + 270), 9, fill=(110, 110, 118))
        rrect(d, (90, y + 282, w - 220, y + 296), 7, fill=(70, 70, 78))
    return im

def phone(fr, cx, cy, screen, scale=1.0):
    sw, sh = screen.size; pad = 26
    body = Image.new("RGBA", (sw + pad * 2, sh + pad * 2), (0, 0, 0, 0)); d = ImageDraw.Draw(body)
    rrect(d, (0, 0, body.width - 1, body.height - 1), 78, fill=(28, 28, 32), outline=(90, 90, 100), width=5)
    mask = Image.new("L", screen.size, 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, sw - 1, sh - 1), 56, fill=255)
    body.paste(screen, (pad, pad), mask)
    rrect(d, (body.width / 2 - 70, pad + 18, body.width / 2 + 70, pad + 56), 19, fill=(0, 0, 0))
    if scale != 1: body = body.resize((int(body.width * scale), int(body.height * scale)), Image.BILINEAR)
    paste_c(fr, body, cx, cy)

def clock(fr, cx, cy, r, ang_min, ang_hr, ring=WHITE):
    d = ImageDraw.Draw(fr)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 20, 26), outline=ring, width=14)
    for k in range(12):
        a = k / 12 * 2 * math.pi; r0 = r * (0.78 if k % 3 == 0 else 0.84)
        d.line((cx + r0 * math.sin(a), cy - r0 * math.cos(a), cx + r * 0.9 * math.sin(a), cy - r * 0.9 * math.cos(a)), fill=ring, width=10 if k % 3 == 0 else 5)
    for a, L, wd, col in ((ang_hr, 0.5, 18, WHITE), (ang_min, 0.75, 11, RED)):
        d.line((cx, cy, cx + r * L * math.sin(a), cy - r * L * math.cos(a)), fill=col, width=wd)
    d.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=RED)

def big_text(fr, txt, cx, cy, size, col=WHITE, f=BLACK, stroke=0):
    ImageDraw.Draw(fr).text((cx, cy), txt, font=font(f, size), fill=col, anchor="mm", stroke_width=stroke, stroke_fill=(0, 0, 0))

def banner(fr, cx, cy, em, title, body, alpha=1.0):
    im = Image.new("RGBA", (900, 170), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    rrect(d, (0, 0, 899, 169), 40, fill=(44, 44, 50, 240), outline=(80, 80, 90), width=2)
    rrect(d, (28, 35, 128, 135), 24, fill=(70, 70, 80))
    paste_c(im, emoji(em, 64), 78, 85)
    d.text((155, 58), title, font=font(BLACK, 38), fill=WHITE, anchor="lm")
    d.text((155, 112), body, font=font(HELV, 36), fill=(200, 200, 210), anchor="lm")
    d.text((870, 58), "now", font=font(HELV, 30), fill=(150, 150, 160), anchor="rm")
    paste_c(fr, im, cx, cy, alpha=alpha)

def toggle(d, x, y, on_amt):
    col = tuple(int(GREEN[i] * on_amt + (72, 72, 80)[i] * (1 - on_amt)) for i in range(3))
    rrect(d, (x, y, x + 150, y + 88), 44, fill=col)
    kx = x + 6 + (150 - 88) * on_amt
    d.ellipse((kx, y + 6, kx + 76, y + 82), fill=WHITE)

# ---------------------------------------------------------------- scenes
CY = 700  # visual centre (captions live lower, Shorts UI lives at the bottom/right)
def scene(i, fr, lt, dur):
    p = lt / dur
    d = ImageDraw.Draw(fr)
    if i == 0:  # frantic scroll, then freeze
        speed = 2600 if lt < dur - 0.35 else 0
        sc = 2600 * min(lt, dur - 0.35)
        sh = (random.uniform(-14, 14) if speed == 0 else 0)
        phone(fr, W / 2 + sh, CY - 40, feed(sc))
        if speed == 0: paste_c(fr, emoji("✋", 300), W / 2, CY - 40)
    elif i == 1:  # time draining away
        clock(fr, W / 2, CY, 330, lt * 14, lt * 14 / 12)
        paste_c(fr, emoji("⏳", 230), W / 2, CY, rot=math.sin(lt * 9) * 12)
    elif i == 2:  # unaware: phone glows, slow push in, glitch
        s = 1.0 + 0.25 * p
        im = emoji("📱", int(560 * s))
        if int(lt * FPS) % 9 in (0, 1):
            r = im.copy(); paste_c(fr, r, W / 2 - 18, CY, alpha=0.5)
        paste_c(fr, im, W / 2, CY)
        paste_c(fr, emoji("🫣", 200), W / 2 + 230, CY + 230)
    elif i == 3:  # "just five minutes"
        rrect(d, (W / 2 - 230, CY - 330, W / 2 + 230, CY - 10), 90, fill=(255, 0, 51))
        d.polygon([(W / 2 - 60, CY - 250), (W / 2 - 60, CY - 90), (W / 2 + 85, CY - 170)], fill=WHITE)
        secs = int(300 * ease(p))
        big_text(fr, f"{secs // 60}:{secs % 60:02d}", W / 2, CY + 170, 210, WHITE)
        big_text(fr, "JUST 5 MINUTES…", W / 2, CY + 330, 64, (190, 190, 200))
    elif i == 4:  # one video becomes ten
        n = min(10, 1 + int(lt / (dur / 11)))
        for k in range(n):
            cx = W / 2 - 230 + (k % 2) * 460; cy = CY - 480 + (k // 2) * 175
            a = clamp((lt - k * dur / 11) / 0.12)
            box = Image.new("RGBA", (420, 175), (0, 0, 0, 0)); bd = ImageDraw.Draw(box)
            rrect(bd, (0, 0, 419, 174), 24, fill=FEED_COLS[k % 7])
            bd.polygon([(185, 55), (185, 120), (245, 87)], fill=WHITE)
            paste_c(fr, box.resize((int(420 * (0.6 + 0.4 * a)), int(175 * (0.6 + 0.4 * a)))), cx, cy, alpha=a)
        big_text(fr, f"{n} VIDEOS", W / 2, CY + 420, 96, YELLOW, f=IMPACT, stroke=8)
    elif i == 5:  # an hour is gone
        m = 5 + 55 * ease(p * 1.15)
        clock(fr, W / 2, CY - 120, 270, m / 60 * 2 * math.pi * 4, m / 60 * 2 * math.pi / 12, ring=RED if p > 0.8 else WHITE)
        hh, mm = divmod(int(m), 60)
        big_text(fr, f"{hh}:{mm:02d}:00", W / 2, CY + 300, 170, RED if p > 0.8 else WHITE, stroke=6)
    elif i == 6:  # reveal hit
        sh = (random.uniform(-26, 26), random.uniform(-26, 26)) if lt < 0.35 else (0, 0)
        pulse = 1 + 0.08 * math.sin(lt * 18) * max(0, 1 - lt)
        paste_c(fr, emoji("⚠️", int(470 * pulse)), W / 2 + sh[0], CY + sh[1])
    elif i == 7:  # notifications pile in
        items = [("💬", "Messages", "3 new messages"), ("🔥", "Trending", "You won't believe this…"),
                 ("❤️", "Social", "Someone liked your post"), ("🛒", "Shop", "Sale ends tonight!")]
        for k, (em, ti, bo) in enumerate(items):
            a = ease((lt - k * dur / 4.2) / 0.18)
            if a > 0: banner(fr, W / 2, CY - 360 + k * 200 - (1 - a) * 120, em, ti, bo, alpha=a)
    elif i == 8:  # brain <-> phone loop
        paste_c(fr, emoji("🧠", 260), W / 2 - 250, CY)
        paste_c(fr, emoji("📱", 260), W / 2 + 250, CY)
        for sgn in (-1, 1):
            d.arc((W / 2 - 200, CY - 260 * (sgn < 0) - 60, W / 2 + 200, CY + 260 * (sgn > 0) + 60), 200 if sgn < 0 else 20, 340 if sgn < 0 else 160, fill=YELLOW, width=12)
        a = lt * 4.5
        dx, dy = 200 * math.cos(a), 160 * math.sin(a)
        d.ellipse((W / 2 + dx - 26, CY + dy - 26, W / 2 + dx + 26, CY + dy + 26), fill=YELLOW)
    elif i == 9:  # habit loop ring
        R = 320; d.ellipse((W / 2 - R, CY - R, W / 2 + R, CY + R), outline=(255, 255, 255, 70), width=10)
        rot = lt * 1.2
        for k, lab in enumerate(("NOTIFY", "CHECK", "SCROLL")):
            a = rot + k * 2 * math.pi / 3
            x, y = W / 2 + R * math.cos(a), CY + R * math.sin(a)
            rrect(d, (x - 135, y - 48, x + 135, y + 48), 48, fill=(255, 45, 85))
            big_text(fr, lab, x, y, 44)
        big_text(fr, "HABIT", W / 2, CY, 120, YELLOW, f=IMPACT)
    elif i == 10:  # try this today
        s = 0.7 + 0.3 * ease(p * 2.5)
        paste_c(fr, emoji("✅", int(420 * s)), W / 2, CY)
    elif i == 11:  # notifications off
        apps = [("🎮", "Games"), ("🛍️", "Shopping"), ("📣", "Social"), ("📰", "News")]
        rrect(d, (90, CY - 450, W - 90, CY + 400), 50, fill=(28, 28, 32))
        big_text(fr, "NOTIFICATIONS", W / 2, CY - 365, 54, (160, 160, 170))
        for k, (em, nm) in enumerate(apps):
            y = CY - 260 + k * 165
            paste_c(fr, emoji(em, 90), 200, y + 44)
            d.text((280, y + 44), nm, font=font(BLACK, 52), fill=WHITE, anchor="lm")
            on = 1 - ease((lt - 0.25 - k * 0.38) / 0.18)
            toggle(d, W - 290, y, on)
    elif i == 12:  # screen time chart
        rrect(d, (90, CY - 450, W - 90, CY + 400), 50, fill=(28, 28, 32))
        big_text(fr, "SCREEN TIME", W / 2, CY - 360, 64, WHITE)
        vals = [0.62, 0.8, 0.7, 0.95, 0.88, 1.0, 0.92]
        for k, v in enumerate(vals):
            x = 190 + k * 115; h = 500 * v * ease((lt - k * 0.06) / 0.5)
            rrect(d, (x - 38, CY + 280 - h, x + 38, CY + 280), 16, fill=BLUE if k < 6 else RED)
            d.text((x, CY + 330), "MTWTFSS"[k], font=font(BLACK, 40), fill=(150, 150, 160), anchor="mm")
    elif i == 13:  # shocked
        s = 0.8 + 0.3 * ease(p * 1.5)
        sh = random.uniform(-10, 10) if p < 0.2 else 0
        big_text(fr, "7H 42M", W / 2, CY - 360, 150, RED, stroke=8)
        big_text(fr, "DAILY AVERAGE", W / 2, CY - 250, 52, (190, 190, 200))
        paste_c(fr, emoji("😳", int(420 * s)), W / 2 + sh, CY + 110)
    elif i == 14:  # now be honest
        paste_c(fr, emoji("👀", int(380 + 140 * p)), W / 2, CY)
    elif i == 15:  # loop back into scrolling
        phone(fr, W / 2, CY - 40, feed(900 * lt + 300), scale=1.0 + 0.04 * p)

# ---------------------------------------------------------------- video
def frame(n):
    t = n / FPS
    i = max(k for k in range(len(LINES)) if scene_t0[k] <= t)
    lt, dur = t - scene_t0[i], scene_t1[i] - scene_t0[i]
    fr = BGS[section_of(i)].copy()
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    scene(i, layer, lt, dur)
    punch = 1.08 - 0.08 * ease(lt / 0.2)  # zoom punch on every cut
    if punch > 1.001:
        layer = layer.resize((int(W * punch), int(H * punch)), Image.BILINEAR)
        layer = layer.crop(((layer.width - W) // 2, (layer.height - H) // 2, (layer.width - W) // 2 + W, (layer.height - H) // 2 + H))
    fr.alpha_composite(layer)
    draw_caption(fr, t)
    if i in SECTION_STARTS and lt < 0.1:  # flash on section changes
        col = (255, 40, 60) if i == 6 else (255, 255, 255)
        fr.alpha_composite(Image.new("RGBA", (W, H), col + (int(200 * (1 - lt / 0.1)),)))
    if t < 0.55:  # cold open: tiny notification before the first word
        banner(fr, W / 2, 200 + 0 * t, "🔔", "Reminder", "You've been scrolling 2h 14m", alpha=ease(t / 0.15))
    # progress bar keeps people watching
    ImageDraw.Draw(fr).rectangle((0, 0, int(W * t / DUR), 10), fill=YELLOW)
    return fr.convert("RGB")

# ---------------------------------------------------------------- audio
def tone(f, d, decay=6.0, harm=(1, 0.5, 0.25)):
    tt = np.arange(int(d * SR)) / SR
    return sum(a * np.sin(2 * np.pi * f * (k + 1) * tt) for k, a in enumerate(harm)) * np.exp(-decay * tt)
def ding():
    a = np.zeros(int(0.7 * SR)); a[:int(0.15 * SR)] += tone(1568, 0.15, 18); b = tone(2093, 0.55, 7)
    a[int(0.1 * SR):int(0.1 * SR) + len(b)] += b; return a * 0.35
def bass_hit():
    tt = np.arange(int(1.4 * SR)) / SR; f = 35 + 120 * np.exp(-tt * 7)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 2.6)
    n = np.random.default_rng(1).normal(0, 1, len(tt)); n = np.convolve(n, np.ones(60) / 60, "same") * np.exp(-tt * 14)
    return s * 1.0 + n * 1.2
def whoosh(d=0.3):
    n = np.random.default_rng(int(d * 1000)).normal(0, 1, int(d * SR))
    n = n - np.convolve(n, np.ones(25) / 25, "same"); env = np.sin(np.linspace(0, np.pi, len(n))) ** 2
    return n * env * 0.12
def click():
    tt = np.arange(int(0.03 * SR)) / SR; return np.sin(2 * np.pi * 2500 * tt) * np.exp(-tt * 250) * 0.25

def build_audio():
    N = int(DUR * SR) + SR; mix = np.zeros(N); voice = np.zeros(N)
    def add(buf, s, t):
        k = int(t * SR); buf[k:k + len(s)] += s[:N - k]
    for a, s in zip(VO, starts): add(voice, a / (np.abs(a).max() + 1e-9) * 0.9, s)
    tt = np.arange(N) / SR
    # suspense drone: dark minor bed that tightens toward the reveal, lifts on the payoff
    rev, pay, loop = starts[6], starts[10], starts[14]
    drone = (np.sin(2 * np.pi * 55 * tt) + 0.5 * np.sin(2 * np.pi * 65.4 * tt) + 0.35 * np.sin(2 * np.pi * 110 * tt + np.sin(tt * 0.7)))
    tension = 0.5 * np.sin(2 * np.pi * 58.3 * tt) * ((tt > rev) & (tt < pay))
    pad = (np.sin(2 * np.pi * 220 * tt) + np.sin(2 * np.pi * 277.2 * tt) + np.sin(2 * np.pi * 329.6 * tt)) * 0.18 * ((tt > pay) & (tt < loop))
    swell = np.clip(tt / rev, 0, 1) * 0.6 + 0.4
    music = (drone * swell * ((tt < pay) | (tt > loop)) + tension + pad) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.6 * tt) ** 2)
    fade = np.clip((DUR - tt) / 0.5, 0, 1) * np.clip(tt / 0.3, 0, 1)
    mix += music * 0.085 * fade
    add(mix, ding(), 0.02)
    for k in range(4): add(mix, ding() * 0.7, scene_t0[7] + k * (scene_t1[7] - scene_t0[7]) / 4.2)
    add(mix, bass_hit() * 0.75, rev - 0.03)
    for k in range(1, len(LINES)):
        if k != 6: add(mix, whoosh(), scene_t0[k] - 0.15)
    for sc in (1, 5):  # clock ticking
        t0 = scene_t0[sc]
        while t0 < scene_t1[sc]: add(mix, click() * 0.8, t0); t0 += 0.25
    for k in range(4): add(mix, click() * 1.6, scene_t0[11] + 0.25 + k * 0.38)
    # duck the bed under speech
    env = np.convolve(np.abs(voice), np.ones(4410) / 4410, "same")
    duck = 1 - 0.45 * np.clip(env * 12, 0, 1)
    out = np.tanh((voice + mix * duck) * 1.1) * 0.95
    out = out[:int(DUR * SR)]
    pcm = (out * 32767).astype(np.int16)
    path = os.path.join(HERE, "mix.wav")
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes()); w.close()
    return path

if PREVIEW:
    for s in PREVIEW.split(","):
        frame(int(float(s) * FPS)).save(os.path.join(HERE, f"prev_{float(s):05.2f}.png"))
    print("duration", round(DUR, 2)); sys.exit()

wav = build_audio()
cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for n in range(NFR):
    proc.stdin.write(frame(n).tobytes())
proc.stdin.close(); proc.wait()
print("wrote", OUT, "duration", round(DUR, 2), "frames", NFR)
