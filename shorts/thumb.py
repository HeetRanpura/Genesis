import sys, math
src = open("render.py").read().split("if PREVIEW:")[0]
sys.argv = ["render.py"]; g = {"__file__": "render.py", "__name__": "thumb"}; exec(src, g)
W, H = g["W"], g["H"]
fr = g["BGS"][2].copy()
g["clock"](fr, W/2, 760, 360, math.radians(300), math.radians(40), ring=g["RED"])
g["paste_c"](fr, g["emoji"]("📱", 300), W/2 + 300, 1020, rot=-14)
g["paste_c"](fr, g["emoji"]("😳", 220), W/2 - 320, 1040)
for k, (txt, col) in enumerate((("WHERE DID", g["WHITE"]), ("THE TIME", g["YELLOW"]), ("GO?", g["YELLOW"]))):
    g["big_text"](fr, txt, W/2, 1340 + k * 170, 190, col, f=g["IMPACT"], stroke=14)
fr.convert("RGB").save(sys.argv[1] if len(sys.argv) > 1 else "/Users/heetranpura/Genesis/shorts/thumbnail.png")
