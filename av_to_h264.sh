conda install -c nvidia cuda-toolkit=12.1

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

for f in *.mp4; do
  ffmpeg -y -i "$f" -c:v libx264 -preset fast -crf 18 "h264_$f"
done
