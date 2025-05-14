# Imports
if __name__ == "__main__":
	#import os
	import random
	import requests
	import json
	import time
	import tarfile
	import numpy as np
	import multiprocessing
	from tensorflow import keras

headers = {
	"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:95.0) Gecko/20100101 Firefox/95.0",
}


# Parameters
random_download = False
n_images_per_class = 50
n_single_images_per_class = 5
n_random_images = 500
imagenet_classes = [
	"n02391049", # Zebra
 	#'n03000134'
	"n03530642", # Honeycomb
	"n09218315", # Honeycomb
	"n04542943", # Waffle iron
	"n06785654", # Crossword puzzle
	"n02110341", # Dalmatian
	"n01440764", # Tench
]


# Paths
from os import path, listdir, makedirs, remove
dir_path = path.dirname(path.realpath(__file__))
visual_tcav_dir_path = path.join(dir_path, "../VisualTCAV")

visual_tcav_cache_dir_path = path.join(visual_tcav_dir_path, "cache")
visual_tcav_cache_dtd_dir_path = path.join(visual_tcav_cache_dir_path, "dtd")

visual_tcav_concept_images_dir_path = path.join(visual_tcav_dir_path, "concept_images")
visual_tcav_concept_images_random_dir_path = path.join(visual_tcav_concept_images_dir_path, "random")

visual_tcav_models_dir_path = path.join(visual_tcav_dir_path, "models")
visual_tcav_models_inception_dir_path = path.join(visual_tcav_models_dir_path, "InceptionV3")
visual_tcav_models_resnet_dir_path = path.join(visual_tcav_models_dir_path, "ResNet50V2")
visual_tcav_models_vgg_dir_path = path.join(visual_tcav_models_dir_path, "VGG16")
visual_tcav_models_convnext_dir_path = path.join(visual_tcav_models_dir_path, "ConvNeXtTiny")

visual_tcav_test_images_dir_path = path.join(visual_tcav_dir_path, "test_images")

imagenet_class_info_file = path.join(dir_path, "imagenet_class_info.json")

if __name__ == "__main__":

	print("Creating folders...", end=' ')
	
	makedirs(visual_tcav_dir_path, exist_ok=True)

	makedirs(visual_tcav_cache_dir_path, exist_ok=True)
	makedirs(visual_tcav_cache_dtd_dir_path, exist_ok=True)

	makedirs(visual_tcav_concept_images_dir_path, exist_ok=True)
	makedirs(visual_tcav_concept_images_random_dir_path, exist_ok=True)

	makedirs(visual_tcav_models_dir_path, exist_ok=True)
	makedirs(visual_tcav_models_inception_dir_path, exist_ok=True)
	makedirs(visual_tcav_models_resnet_dir_path, exist_ok=True)
	makedirs(visual_tcav_models_vgg_dir_path, exist_ok=True)

	makedirs(visual_tcav_test_images_dir_path, exist_ok=True)

	print("Done!")


# Models
if __name__ == "__main__":

	print("Downloading models...", end=' ')

	inception = keras.applications.InceptionV3(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")
	inception.compile(loss='mse')
	inception.save(path.join(visual_tcav_models_inception_dir_path, "InceptionV3-architecture-and-weights-compiled.h5"))
	inception_classes_file = open(path.join(visual_tcav_models_inception_dir_path, "InceptionV3-imagenet-classes.txt"), "w")
	for cl in keras.applications.inception_v3.decode_predictions(np.array([[i for i in range(1000)]]), 1000)[0][::-1]:
		inception_classes_file.write(cl[1] + "\n")
	inception_classes_file.close()

	resnet = keras.applications.ResNet50V2(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")
	resnet.compile(loss='mse')
	resnet.save(path.join(visual_tcav_models_resnet_dir_path, "ResNet50V2-architecture-and-weights-compiled.h5"))
	resnet_classes_file = open(path.join(visual_tcav_models_resnet_dir_path, "ResNet50V2-imagenet-classes.txt"), "w")
	for cl in keras.applications.resnet_v2.decode_predictions(np.array([[i for i in range(1000)]]), 1000)[0][::-1]:
		resnet_classes_file.write(cl[1] + "\n")
	resnet_classes_file.close()

	vgg = keras.applications.VGG16(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")
	vgg.compile(loss='mse')
	vgg.save(path.join(visual_tcav_models_vgg_dir_path, "VGG16-architecture-and-weights-compiled.h5"))
	vgg_classes_file = open(path.join(visual_tcav_models_vgg_dir_path, "VGG16-imagenet-classes.txt"), "w")
	for cl in keras.applications.vgg16.decode_predictions(np.array([[i for i in range(1000)]]), 1000)[0][::-1]:
		vgg_classes_file.write(cl[1] + "\n")
	vgg_classes_file.close()

	convnext = keras.applications.ConvNeXtTiny(include_top=True, include_preprocessing=False, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")
	convnext.compile(loss='mse')
	convnext.save(path.join(visual_tcav_models_convnext_dir_path, "ConvNeXtTiny-architecture-and-weights-compiled"))
	convnext_classes_file = open(path.join(visual_tcav_models_convnext_dir_path, "ConvNeXtTiny-imagenet-classes.txt"), "w")
	for cl in keras.applications.convnext.decode_predictions(np.array([[i for i in range(1000)]]), 1000)[0][::-1]:
		convnext_classes_file.write(cl[1] + "\n")
	convnext_classes_file.close()

	print("Done!")


# Images download
url_to_scrape = lambda wnid: f'http://www.image-net.org/api/imagenet.synset.geturls?wnid={wnid}'

if __name__ == "__main__":
	class_info_dict = dict()
	with open(imagenet_class_info_file) as class_info_json_f:
		class_info_dict = json.load(class_info_json_f)

def get_image_name(url):
	name = url.split('/')[-1].split("?")[0]
	return name

def get_image(url):
	if not url.lower().endswith(".jpg") and not url.lower().endswith(".jpeg"):
		return False
	try:
		import requests
		img_resp = requests.get(url, timeout=(3, 3), headers=headers)
	except:
		return False
	if not 'content-type' in img_resp.headers:
		return False
	if not 'image' in img_resp.headers['content-type']:
		return False
	if (len(img_resp.content) < 1000):
		return False
	img_name = get_image_name(url)
	if (len(img_name) <= 1):
		return False
	return img_resp

def get_image_from_url(url, img_file_path, n):
	img_resp = get_image(url)
	if not img_resp:
		return False
	img_name = str(n) + ".jpg"
	from os import path
	with open(path.join(img_file_path, img_name), 'wb') as img_f:
		img_f.write(img_resp.content)
	return True

# Test images
if __name__ == "__main__":

	print("Downloading test images... this may take a while...", end=' ')

	for imagenet_class in imagenet_classes:
		
		response = requests.get(url_to_scrape(imagenet_class), headers=headers)
		try:
			urls_to_scrape = np.array([url.decode('utf-8') for url in response.content.splitlines()])
		except:
			break
		img_file_path = path.join(visual_tcav_test_images_dir_path, class_info_dict[imagenet_class]["class_name"])
		makedirs(img_file_path, exist_ok=True)
		if random_download:
			random.shuffle(urls_to_scrape)

		procs = dict()
		for i, url in enumerate(urls_to_scrape):
			
			rem = min(n_images_per_class, len(urls_to_scrape)) - len(listdir(img_file_path))
			if rem <= 0:
				break

			if len(procs) < min(rem, 100):
				procs[i] = multiprocessing.Process(target=get_image_from_url, args=(url, img_file_path, i,))
				procs[i].start()
			else:
				for proc in list(procs):
					procs[proc].join()
					procs.pop(proc)

			time.sleep(0.1)
		
		for proc in list(procs):
			procs[proc].join()
			procs.pop(proc)

		for i, file in enumerate(listdir(img_file_path)):
			if i >= n_images_per_class:
				remove(path.join(img_file_path, file))
	
		# Extract one image per class
		to_export = random.choice([dir for dir in listdir(img_file_path) if dir.endswith(".jpg")])
		import shutil
		shutil.copyfile(path.join(img_file_path, file), path.join(visual_tcav_test_images_dir_path, class_info_dict[imagenet_class]["class_name"] + ".jpg"))


	# DO NOT UNCOMMENT !
	#for imagenet_class in imagenet_classes:
	#	response = requests.get(url_to_scrape(imagenet_class), headers=headers)
	#	urls_to_scrape = np.array([url.decode('utf-8') for url in response.content.splitlines()])
	#	img_file_path = os.path.join(visual_tcav_test_images_dir_path)
	#	os.makedirs(img_file_path, exist_ok=True)
	#	if random_download:
	#		random.shuffle(urls_to_scrape)
	#	procs = dict()
	#	for i, url in enumerate(urls_to_scrape):
	#		
	#		rem = n_single_images_per_class - len([path for path in os.listdir(img_file_path) if path.startswith(class_info_dict[imagenet_class]["class_name"] + "_")])
	#		if rem <= 0:
	#			break
	#
	#		if len(procs) < min(rem, 100):
	#			procs[i] = multiprocessing.Process(target=get_image_from_url, args=(url, img_file_path, class_info_dict[imagenet_class]["class_name"] + "_" + str(i),))
	#			procs[i].start()
	#		else:
	#			for proc in list(procs):
	#				procs[proc].join()
	#				procs.pop(proc)
	#
	#		time.sleep(0.1)
	#
	#	for i, file in enumerate([path for path in os.listdir(img_file_path) if path.startswith(class_info_dict[imagenet_class]["class_name"] + "_")]):
	#		if i == 0:
	#			try:
	#				os.rename(os.path.join(img_file_path, file), os.path.join(img_file_path, class_info_dict[imagenet_class]["class_name"] + ".jpg"))
	#			except:
	#				os.remove(os.path.join(img_file_path, class_info_dict[imagenet_class]["class_name"] + ".jpg"))
	#				os.rename(os.path.join(img_file_path, file), os.path.join(img_file_path, class_info_dict[imagenet_class]["class_name"] + ".jpg"))
	# DO NOT UNCOMMENT !

	print("Done!")

def get_first_image_of_class(code):
	try:
		from requests import get
		response = get(url_to_scrape(code), timeout=(3, 3), headers=headers)
	except Exception as e:
		return False
	try:
		urls_to_scrape = [url.decode('utf-8') for url in response.content.splitlines()]
	except:
		return False
	#print(len(urls_to_scrape))
	from random import shuffle
	shuffle(urls_to_scrape)
	count = 0
	for url in urls_to_scrape:
		if count > 3:
			return False
		img_resp = get_image(url)
		if not img_resp:
			count+=1
			continue
		img_name = str(code) + ".jpg"
		from os import path
		with open(path.join(visual_tcav_concept_images_random_dir_path, img_name), 'wb') as img_f:
			img_f.write(img_resp.content)
		return True
	return False

# Random images
if __name__ == "__main__":

	print("Downloading random images... this may take a while...", end=' ')

	keys = list(class_info_dict.keys())
	random.shuffle(keys)

	procs = dict()
	for i, key in enumerate(keys):

		rem = n_random_images - len(listdir(visual_tcav_concept_images_random_dir_path))
		if rem <= 0:
			break
		
		if len(procs) < min(rem, 100):
			procs[key] = multiprocessing.Process(target=get_first_image_of_class, args=(key,))
			procs[key].start()
		else:
			for proc in list(procs):
				procs[proc].join()
				procs.pop(proc)

		time.sleep(0.1)

	for proc in list(procs):
		procs[proc].join()
		procs.pop(proc)

	for i, file in enumerate(listdir(visual_tcav_concept_images_random_dir_path)):
		if i >= n_random_images:
			remove(path.join(visual_tcav_concept_images_random_dir_path, file))

	print("Done!")


# Broden dataset (DTD) downloader
dtd_url = f'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
dtd_filename = "dtd-r1.0.1.tar.gz"

dtd_concepts_to_extract = ['striped', 'zigzagged', 'waffled', 'honeycombed', 'chequered', 'dotted']

def download_dtd_zip(dtd_url):
	try:
		response = requests.get(dtd_url, headers=headers)
	except:
		return False
	if not 'content-length' in response.headers:
		return False
	if response.headers['content-length'] != '625239812':
		return False
	if (len(response.content) < 1000):
		return False
	return response

if __name__ == "__main__":

	print("Downloading DTD images...", end=' ')

	response = download_dtd_zip(dtd_url)
	if response:
		with open(path.join(visual_tcav_cache_dtd_dir_path, dtd_filename), 'wb') as dtd_f:
			dtd_f.write(response.content)

		# Extract
		with tarfile.open(path.join(visual_tcav_cache_dtd_dir_path, dtd_filename)) as dtd_f:
			for dtd_concept in dtd_concepts_to_extract:
				visual_tcav_concept_images_dtd_dir_path = path.join(visual_tcav_concept_images_dir_path, dtd_concept)
				makedirs(visual_tcav_concept_images_dtd_dir_path, exist_ok=True)
				for tarinfo in dtd_f.getmembers():
					if tarinfo.name.startswith(f"dtd/images/{dtd_concept}/"):
						tarinfo.name = path.basename(tarinfo.name)
						dtd_f.extract(tarinfo, visual_tcav_concept_images_dtd_dir_path)

	print("Done!")
