from moviepy.editor import VideoFileClip

# input and output paths
input_path = "squirrel_paw_results/output_20260409_104513_poc_trial.mp4"
output_path = "squirrel_paw_results/output_20260409_104513_poc_trial.gif"

# load video
clip = VideoFileClip(input_path)

# optional: resize (reduces file size)
# clip = clip.resize(width=400)

# optional: set fps (lower = smaller GIF)
clip.write_gif(output_path, fps=10)

print("GIF saved to", output_path)