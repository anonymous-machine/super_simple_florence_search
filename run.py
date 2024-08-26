import argparse
import hashlib
import mimetypes
import os

from pathlib import Path

import psycopg
import torch

import numpy as np
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


def init_database():
	connection = get_db_connection()
	cursor = connection.cursor()
	cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
	create_table_statement = """
	CREATE TABLE IF NOT EXISTS florence_files(
	file_path TEXT,
	sha3_hash CHAR(64),
	florence_caption TEXT,
	embedded_caption vector(768)
	);"""
	cursor.execute(create_table_statement)
	connection.commit()
	cursor.close()

def read_in_chunks(file_object, chunk_size=1024):
	while True:
		data = file_object.read(chunk_size)
		if not data:
			break
		yield data

def hash_file(path: Path, chunk_size: int = 65535) -> str:
	hash_fn = hashlib.sha3_256()
	with open(path, "rb") as f:
		for file_chunk in read_in_chunks(f, chunk_size=chunk_size):
			hash_fn.update(file_chunk)
	return str(hash_fn.hexdigest())

def guess_mime_prefix(path):
	try:
		prefix = mimetypes.guess_type(path)[0].split("/")[0]
	except Exception as e:
		prefix = ""
	return prefix

def get_db_connection():
	db_host = os.environ.get("POSTGRES_HOST", "localhost")
	db_user = os.environ.get("POSTGRES_USER", "user")
	db_name = os.environ.get("POSTGRES_NAME", "vectordb")
	db_port = os.environ.get("POSTGRES_PORT", "5432")
	if db_port[0] != ":":
		db_port = ":" + db_port
	db_password = os.environ.get("POSTGRES_PASSWORD", "password")
	db_url = f"postgresql://{db_user}:{db_password}@{db_host}{db_port}/{db_name}"
	connection = psycopg.connect(db_url)
	register_vector(connection)
	return connection


def load_sentence_transformer() -> SentenceTransformer:
	#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
	model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
	return model

def load_florence() -> (AutoModelForCausalLM, AutoProcessor):
	florence_model = "microsoft/Florence-2-large-ft"
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
	model = AutoModelForCausalLM.from_pretrained(florence_model, torch_dtype=torch_dtype,
																							 trust_remote_code=True).to(device)
	processor = AutoProcessor.from_pretrained(florence_model, trust_remote_code=True)
	return model, processor

def get_caption(model: AutoModelForCausalLM, processor: AutoProcessor, image_path: Path) -> str:
	task = "<MORE_DETAILED_CAPTION>"
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
	image = Image.open(image_path)
	inputs = processor(text=task, images=image, return_tensors="pt").to(device, torch_dtype)
	generated_ids = model.generate(
		input_ids=inputs["input_ids"],
		pixel_values=inputs["pixel_values"],
		max_new_tokens=1024,
		num_beams=3,
		do_sample=False
	)
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

	parsed_answer = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))

	return parsed_answer[task]

def ingest(root: Path):
	init_database()
	file_list = [f for f in root.rglob("*") if f.is_file() and guess_mime_prefix(f) == "image"]

	connection = get_db_connection()
	florence_model, florence_processor = load_florence()
	sentence_transformer = load_sentence_transformer()
	for file in tqdm(file_list):
		try:
			cursor = connection.cursor()
			file_hash = hash_file(file)
			select_statement = f"SELECT * FROM florence_files WHERE sha3_hash LIKE '{file_hash}'"
			cursor.execute(select_statement)
			results = cursor.fetchall()
			if len(results) > 0:
				tqdm.write(f"{file} already in database, continuing")
				continue
			caption = get_caption(model=florence_model, processor=florence_processor, image_path=file)
			tqdm.write(f"Caption for {file}: {caption}")
			embedded_caption = sentence_transformer.encode(caption)
			insert_statement = """INSERT INTO florence_files (file_path, sha3_hash, florence_caption, embedded_caption) VALUES (%s, %s, %s, %s);"""
			values = (str(file), file_hash, caption, embedded_caption)
			cursor.execute(insert_statement, values)
			connection.commit()
		except Exception as e:
			tqdm.write(f"Error on {file}: {e}")
			break

def run_query(query_string, search_depth:int = 25):
	connection = get_db_connection()
	sentence_transformer = load_sentence_transformer()
	embedded_query = sentence_transformer.encode(query_string)
	select_statement = f"SELECT (embedded_caption <#> %s) AS similarity, file_path, florence_caption FROM florence_files ORDER BY similarity LIMIT {search_depth}"
	cursor = connection.cursor()
	cursor.execute(select_statement, (embedded_query, ))
	results = cursor.fetchall()
	for i, r in enumerate(results):
		print(f"{i}: {r[1]} ({r[2]})")
	return results

def cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=Path)
	parser.add_argument("--search", type=str)
	parser.add_argument("--search_depth", type=int, default=25)

	args = parser.parse_args()

	if args.root is not None:
		print(f"Ingesting photos at {args.root}")
		ingest(args.root)

	if args.search is not None:
		run_query(query_string=args.search, search_depth=args.search_depth)

if __name__ == '__main__':
	load_dotenv()
	cli()
