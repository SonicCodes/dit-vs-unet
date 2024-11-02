wget https://huggingface.co/ramimmo/mini.imgnet.int8/resolve/main/inet.json
wget https://huggingface.co/ramimmo/mini.imgnet.int8/resolve/main/inet.npy
pip3 install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install -r ./requirements.txt
