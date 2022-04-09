#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import cv2
import time
import pywt
import argparse
import pygame
import numpy as np
import math
from io import BytesIO,StringIO
from PIL import Image
pygame.init()

ORIGIN_RATE = 0.98 # k  1.5 - 0.6
WATERMARK_RATE = 0.009 # q  

# 4 coefficients : Approximation(cA), Horizontal(cH), Vertical(cV), Diagonal(cD)

# we have used haar as a mother wavelet
def dwt2_for_single_image_3_level(img):

	# first level --> we taking the image as an input
	coeffs_1 = pywt.dwt2(img, 'haar', mode='reflect')
	
	# second level --> we taking the LL of first level as an input for second level
	coeffs_2 = pywt.dwt2(coeffs_1[0], 'haar', mode='reflect')
	
	# third level -->  we taking the LL of second level as an input for third level
	coeffs_3 = pywt.dwt2(coeffs_2[0], 'haar', mode='reflect')
	
	return (coeffs_1, coeffs_2, coeffs_3)

def dwt2_for_single_image_2_level(img):

	# first level --> we taking the image as an input
	coeffs_1 = pywt.dwt2(img, 'haar', mode='reflect')
	# coeffs = (LL, (LH, HL, HH))
	
	# second level --> we taking the LL of first level as an input for second level
	coeffs_2 = pywt.dwt2(coeffs_1[0], 'haar', mode='reflect')
	# coeffs = (LL, (LH, HL, HH))

	return (coeffs_1, coeffs_2)

def dwt2_for_3_level(original_image, watermark_image):

	# for original_image
	(coeffs1_1, coeffs1_2, coeffs1_3) = dwt2_for_single_image_3_level(original_image)

	# for watermark_image
	(coeffs2_1, coeffs2_2, coeffs2_3) = dwt2_for_single_image_3_level(watermark_image)
	
	return (coeffs1_1, coeffs1_2, coeffs1_3, coeffs2_3)

def dwt2_for_2_level(original_image, watermark_image):
	# coeffs = (LL, (LH, HL, HH))

	# for original image
	(coeffs1_1, coeffs1_2) = dwt2_for_single_image_2_level(original_image)
	
	# for watermark image
	(coeffs2_1, coeffs2_2) = dwt2_for_single_image_2_level(watermark_image)
	
	# 
	return (coeffs1_1, coeffs1_2, coeffs2_2)


def idwt2_for_3_level(img, coeffs1_1_h, coeffs1_2_h, coeffs1_3_h):
	# third level for 2-D inverse discrete wavelet transform 
	cf3 = (img, coeffs1_3_h)
	img = pywt.idwt2(cf3, 'haar', mode='reflect')

	# second level for 2-D inverse discrete wavelet transform
	cf2 = (img, coeffs1_2_h)
	img = pywt.idwt2(cf2, 'haar', mode='reflect')

	# first level for 2-D inverse discrete wavelet transform
	cf1 = (img, coeffs1_1_h)
	img = pywt.idwt2(cf1, 'haar', mode='reflect')
	
	return img

def idwt2_for_2_level(img, coeffs1_1_h, coeffs1_2_h):
	# second level for 2-D inverse discrete wavelet transform
	cf2 = (img, coeffs1_2_h)
	img = pywt.idwt2(cf2, 'haar', mode='reflect')

	# first level for 2-D inverse discrete wavelet transform
	cf1 = (img, coeffs1_1_h)
	img = pywt.idwt2(cf1, 'haar', mode='reflect')

	return img

def channel_embedding(original_image_chan, watermark_image_chan):

	if(dwtLevel == 2):
		# original image : LL1, LL2
		# watermark image : LL2 
		(coeffs1_1, coeffs1_2, coeffs2_2) = dwt2_for_2_level(original_image_chan, watermark_image_chan)

		# formula 
		# k = original_rate
		# q = watermark_rate 

		# WMI = k * (LL2) + q * (WM2)
		embedding_image = cv2.add(cv2.multiply(ORIGIN_RATE, coeffs1_2[0]), cv2.multiply(WATERMARK_RATE, coeffs2_2[0]))

		# 2-D inverse discrete wavelet transform
		embedding_image = idwt2_for_2_level(embedding_image, coeffs1_1[1], coeffs1_2[1])
		
		np.clip(embedding_image, 0, 255, out=embedding_image)
		embedding_image = embedding_image.astype('uint8')

		return embedding_image
	else:
		(coeffs1_1, coeffs1_2, coeffs1_3, coeffs2_3) = dwt2_for_3_level(original_image_chan, watermark_image_chan)

		# formula : 
		embedding_image = cv2.add(cv2.multiply(coeffs1_3[0],ORIGIN_RATE), cv2.multiply(coeffs2_3[0],WATERMARK_RATE))
		
		# 2-D Inverse Discrete Wavelet Transform.
		embedding_image = idwt2_for_3_level(embedding_image, coeffs1_1[1], coeffs1_2[1], coeffs1_3[1])

		np.clip(embedding_image, 0, 255, out=embedding_image)
		embedding_image = embedding_image.astype('uint8')
		return embedding_image


def img_segment_embedding(watermark_image, original_image):
	origin_size = original_image.shape[:2]

	watermark_image = cv2.resize(watermark_image, (origin_size[0], origin_size[1]))
	
	# split the original and watermarked image into its `R G B` layer

	(original_image_r, original_image_g, original_image_b) = cv2.split(original_image)
	(watermark_image_r, watermark_image_g, watermark_image_b) = cv2.split(watermark_image)

	embedding_image_r = channel_embedding(original_image_r, watermark_image_r)

	embedding_image_g = channel_embedding(original_image_g, watermark_image_g)

	embedding_image_b = channel_embedding(original_image_b, watermark_image_b)

	embedding_image = cv2.merge([embedding_image_r, embedding_image_g, embedding_image_b])
	return embedding_image



def split_img_segments(image, num):
	segments = []
	if num <= 1:
		segments.append(image)
		return segments
	ratio = 1.0 / float(num)
	height = image.shape[0]
	width = image.shape[1]
	pHeight = int(ratio * height)
	pHeightInterval = (height - pHeight) / (num - 1)
	pWidth = int(ratio * width)
	pWidthInterval = (width - pWidth) / (num - 1)

	for i in range(num):
		for j in range(num):
			x = int(pWidthInterval * i)
			y = int(pHeightInterval * j)
			segments.append(image[y:y + pHeight, x:x + pWidth, :])
	return segments



def merge_img_segments(segments, num, shape):
	if num <= 1:
		return segments[0]
	ratio = 1.0 / float(num)
	height = shape[0]
	width = shape[1]
	channel = shape[2]
	image = np.empty([height, width, channel], dtype=int)

	pHeight = int(ratio * height)
	pHeightInterval = (height - pHeight) / (num - 1)
	pWidth = int(ratio * width)
	pWidthInterval = (width - pWidth) / (num - 1)
	cnt = 0
	for i in range(num):
		for j in range(num):
			x = pWidthInterval * i
			y = pHeightInterval * j
			image[y:y + pHeight, x:x + pWidth, :] = segments[cnt]
			cnt += 1
	return image

# Calculates Peak signal to noise ratio
def psnr(original_image, other_image):
	# Mean square error
	mse = np.mean((original_image - other_image) ** 2)
	
	print("mse:", mse)

	if mse == 0:
			return 100
	else:
		PIXEL_MAX = 255.0
		return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# function for embedding watermark to the original image
def embedding(args, flag):

	num = args.image_segments_num
	# get the original image
	origin_image = cv2.imread(args.origin)
	
	# get the watermarked image
	watermark_img = cv2.imread(args.watermark)

	# make `num` segments of original image
	origin_img_segments = split_img_segments(origin_image, num)

	embedding_img_segments = []
	
	# for every image segment we will process 2-D, 3-level DWT operation and append that in an array
	for segment in origin_img_segments:
		embedding_img_segments.append(img_segment_embedding(watermark_img,segment))


	# merge all the embedded segments blocks
	embedding_image = merge_img_segments(embedding_img_segments, num, origin_image.shape)
	cv2.imwrite(args.embedding, embedding_image)

	res = psnr(origin_image, embedding_image)
	print("psnr:",res)


def channel_extracting(original_image_chan, embedding_image_chan):
	if(dwtLevel == 2):
		(coeffs1_1, coeffs1_2, coeffs2_2) = dwt2_for_2_level(original_image_chan, embedding_image_chan)
		
		recovered_image_layer = cv2.divide(cv2.subtract(coeffs2_2[0], cv2.multiply(ORIGIN_RATE, coeffs1_2[0])), WATERMARK_RATE)
		recovered_image_layer = idwt2_for_2_level(recovered_image_layer, (None, None, None), (None, None, None))

		return recovered_image_layer
	else:
		(coeffs1_1, coeffs1_2, coeffs1_3, coeffs2_3) = dwt2_for_3_level(original_image_chan, embedding_image_chan)
		
		recovered_image_layer = cv2.divide(cv2.subtract(coeffs2_3[0], cv2.multiply(ORIGIN_RATE, coeffs1_3[0])), WATERMARK_RATE)
		recovered_image_layer = idwt2_for_3_level(recovered_image_layer, (None, None, None), (None,None, None), (None, None, None))
		
		return recovered_image_layer



def img_segment_extracting(origin_image, embedding_image, num):
	# split the original and embedded image into its `R G B` layer

	(origin_image_r, origin_image_g, origin_image_b) = cv2.split(origin_image)
	(embedding_image_r, embedding_image_g, embedding_image_b) = cv2.split(embedding_image)

	extracting_img_r = channel_extracting(origin_image_r, embedding_image_r)
	extracting_img_g = channel_extracting(origin_image_g, embedding_image_g)
	extracting_img_b = channel_extracting(origin_image_b, embedding_image_b)
	
	# merge the extracting `R G B` layers
	
	extracting_img = cv2.merge([extracting_img_r, extracting_img_g, extracting_img_b])
	
	return extracting_img



def extracting(args):
	num = args.image_segments_num
	embedding_image = cv2.imread(args.embedding)
	original_image = cv2.imread(args.origin)
	origin_size = original_image.shape[:2]
	embedding_image = cv2.resize(embedding_image, (origin_size[1], origin_size[0]))

	# Several Blocks

	original_img_segments = split_img_segments(original_image, num)
	embedding_img_segments = split_img_segments(embedding_image, num)
	extracting_img_segments = []

	for i in range(0, num * num):
		extracting_img_segments.append(img_segment_extracting(original_img_segments[i], embedding_img_segments[i], i))


	extracting_img = merge_img_segments(extracting_img_segments, num, original_image.shape)
	cv2.imwrite(args.extracting, extracting_img)

	res = psnr(original_image, extracting_img)
	print("psnr:",res)


	



description = '\n'.join(['Compares encode algs using the SSIM metric.',
                        '  Example:',
                        '   python watermark.py  --opt embedding --origin origin.jpg --watermark watermark.jpg --embedding embedding.jpg'
                        ])

parser = argparse.ArgumentParser(prog='compare', formatter_class=argparse.RawTextHelpFormatter, description=description)
parser.add_argument('--opt', default='embedding', help='embedding or extracting')
parser.add_argument('--level', default='3', help='The level of DWT to be applied, options from 2 and 3')
parser.add_argument('--origin', default='./samples/test.jpg', help='origin image file, length and width must be a multiple of 8')
parser.add_argument('--watermark', default='./samples/watermark.jpg', help='watermark image file')
parser.add_argument('--watermark_word', default='lzh3', help='watermark words')
parser.add_argument('--embedding', default='./samples/watermarked.jpg', help='embedding image file')
parser.add_argument('--image_segments_num', default=1, type=int, help="The sqrt number of image's segments, may be 1,2,4")
parser.add_argument('--extracting', default='./samples/extract.jpg', help='extracting image file')


args = parser.parse_args()

dwtLevel = args.level

start = time.time()
if args.opt == 'embedding':
  embedding(args, 'image')
elif args.opt == 'extracting':
  extracting(args)

print(time.time() - start)
