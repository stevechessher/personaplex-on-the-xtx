#!/bin/bash
# runs INSIDE a throwaway ubuntu:24.04 container: /w = /opt/personaplex/build, /rel = release bundle (ro)
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential cmake git libsdl2-dev wget xz-utils ca-certificates > /dev/null
cd /w
[ -d moshi.cpp ] || git clone -q https://github.com/Codes4Fun/moshi.cpp
cd moshi.cpp; git checkout -q -f v0.8.0-beta; git clean -qfd; git apply /w/inject.patch; git log -1 --format='moshi.cpp %H %ci'; cd ..
[ -d ggml ] || git clone -q --branch for_moshi --single-branch https://github.com/Codes4Fun/ggml
cd ggml; git checkout -q "$(git rev-list -1 --before='2026-02-14' for_moshi)"; git log -1 --format='ggml %H %ci'; cd ..
[ -d sentencepiece ] || git clone -q --branch v0.2.0 --depth 1 https://github.com/google/sentencepiece
if [ ! -d ffmpeg ]; then
  API=$(wget -qO- https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest)
  URL=$(echo "$API" | grep -o 'https://[^"]*n8\.[0-9][^"]*linux64-lgpl-shared[^"]*\.tar\.xz' | head -1)
  [ -n "$URL" ] || URL=$(echo "$API" | grep -o 'https://[^"]*master-latest-linux64-lgpl-shared\.tar\.xz' | head -1)
  echo "ffmpeg url: $URL"
  wget -q "$URL" -O ff.tar.xz
  mkdir ffmpeg; tar xf ff.tar.xz -C ffmpeg --strip-components=1; rm ff.tar.xz
fi
ls ffmpeg/lib/libavcodec.so.62 > /dev/null   # must match the bundle's soname
ls ffmpeg/lib | grep -E "libavcodec.so|libavformat.so" | head
mkdir -p libs; for l in ggml ggml-base sentencepiece; do ln -sf /rel/lib$l.so.0 libs/lib$l.so; done
rm -rf b; mkdir b; cd b
cmake ../moshi.cpp -DCMAKE_BUILD_TYPE=Release \
  -DGGML_INCLUDE_DIR=/w/ggml/include -DGGML_LIBRARY_DIR=/w/libs \
  -DSentencePiece_INCLUDE_DIR=/w/sentencepiece/src -DSentencePiece_LIBRARY_DIR=/w/libs \
  -DFFmpeg_DIR=/w/ffmpeg > /w/cmake.log
make -j6 moshi personaplex 2>&1 | tail -25
ls -la bin/
echo BUILD_OK
