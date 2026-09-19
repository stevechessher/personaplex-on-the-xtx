# NeMo-Speech.cpp streaming ASR server on an AMD GPU (Vulkan/RADV).
# The install (prebuilt linux-x86_64-vulkan archive via scripts/install.sh --backend vulkan) and the models
# live on the host in /opt/nemospeech, mounted at /w:
#   HOME=/w/home-vulkan sh install.sh --backend vulkan
#   mv /w/home-vulkan/.local/share/nemo-speech/lib/libstdc++.so.6{,.bundled}   # RADV needs the system one
#   mv /w/home-vulkan/.local/share/nemo-speech/lib/libgcc_s.so.1{,.bundled}
#   nemo-speech pull nemotron-en
# Run (bound to the docker bridge only, so it isn't on the LAN):
#   docker run -d --name nsserve --restart unless-stopped --device /dev/dri/renderD128 --group-add <render gid> \
#     -p 172.17.0.1:8997:8080 -v /opt/nemospeech:/w nsserve:local serve --asr-model <gguf> --device vulkan \
#     --host 0.0.0.0 --port 8080 --no-ui --endpointing --stop-history-eou-ms 500
FROM ubuntu:24.04
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      ca-certificates curl libvulkan1 mesa-vulkan-drivers libgomp1 && rm -rf /var/lib/apt/lists/*
ENV NEMO_SPEECH_MODEL_DIR=/w/models HOME=/w/home-vulkan
ENTRYPOINT ["/w/home-vulkan/.local/share/nemo-speech/bin/nemo-speech"]
