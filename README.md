
# Image watermarking using 3-level Discrete Wavelet Transformation

## Local Usage

```
# install all the dependencies before running the code
$ pip install time PyWavelets numpy pillow opencv-python pygame

# For 2-Level DWT

# watermark image will be embedded in cover image
$ python3 watermark.py --opt embedding --level 2 --origin origin.png --watermark watermark.png --embedding embedding-2.jpg

# to extract the watermark from the watermarked image
$ python3 watermark.py --opt extracting --level 2 --origin origin.png --embedding embedding-2.jpg --extracting extracting-2.jpg

# For 3-Level DWT

# watermark image will be embedded in cover image
$ python3 watermark.py --opt embedding --level 3 --origin origin.png --watermark watermark.png --embedding embedding-3.jpg

# to extract the watermark from the watermarked image
$ python3 watermark.py --opt extracting --level 3 --origin origin.png --embedding embedding-3.jpg --extracting extracting-3.jpg

```


