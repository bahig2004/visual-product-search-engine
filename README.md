install this-> "faiss-cpu" el lib ely bast5dmha 
search query dah example msh asly 3alshan yas3d fel integration
--------------------------------------------------------------------------
Send this to the integration person:


---

How to use my code

This code only covers the FAISS indexing part and the similarity search part for the image retrieval system.

It does not do full system integration, UI, API wiring, or model training.
Your job is only to connect it to the previous stage and the next stage.


---

What my code is responsible for

My code handles these 2 things:

1. Build the FAISS index

takes the final image feature vectors

normalizes them for cosine similarity

stores them in a FAISS index

saves metadata needed to map search results back to images



2. Run similarity search

takes one query feature vector

normalizes it the same way

searches the FAISS index

returns the nearest matching image IDs / file paths / scores





---

What my code expects from the previous stage

Before my code can work, the previous stage must already give:

the dataset image embeddings / feature vectors

one feature vector per image

fixed feature size for all vectors

the mapping of each vector to its image identity

for example:

image id

filename

relative path

label if available




Important

The feature extraction stage must be finished first.

My code does not create image features by itself unless that was explicitly added somewhere else.

So the integration person must make sure the feature extractor passes ready vectors into the indexing code.


---

Main files and how they are used

1) builddatasetindex.py

Use this when you want to build the search index for the dataset.

It should be run:

after all dataset image features are prepared

when the dataset changes

when you want to rebuild the index from scratch


What it does

loads dataset vectors

validates shapes and types

normalizes vectors for cosine similarity

creates FAISS index

writes index file to disk

writes metadata / mapping file to disk


Expected result

You should get saved files such as:

FAISS index file

metadata / mapping file

optional config / dimension info file


The exact names depend on how the code was written.


---

2) search_query.py

Use this when you want to search for similar images.

It should be run:

after the FAISS index already exists

after metadata already exists

after the query image has already been converted to a feature vector


What it does

loads the saved FAISS index

loads metadata

accepts a query feature vector

normalizes the query vector

searches top-k nearest matches

returns results with similarity scores and mapped image info



---

Correct order of use

The integration should follow this order:

Step 1: extract dataset features

Another part of the project must generate embeddings for all dataset images.

Step 2: build the FAISS index

Run builddatasetindex.py once on the dataset vectors.

Step 3: save artifacts

Keep:

index file

metadata file

any saved config needed by search


Step 4: extract query feature

At search time, another part of the project must generate the query image embedding.

Step 5: run search

Pass that query vector into search_query.py.

Step 6: display or use results

Use returned image IDs / paths / scores in the API or UI.


---

What the integrator must connect

The integration person must wire these parts:

Input into indexing

Connect:

dataset loader

feature extractor output

metadata source


into the indexing script

Input into searching

Connect:

query image upload or source

same feature extractor used for dataset images

search script input


into the search script

Output from search

Connect returned results into:

backend API response

frontend results page

ranking display

result image loading



---

Rules the integrator must not break

1) Use the same embedding model everywhere

The model used for:

dataset images

query image


must be the same model and same preprocessing.

If not, search quality will break.


---

2) Use the same vector dimension everywhere

If dataset vectors are dimension d, then query vectors must also be dimension d.

Otherwise FAISS search will fail.


---

3) Normalize exactly the same way

Because this code uses cosine similarity, vectors must be normalized the same way in both:

indexing

querying


Usually this means L2 normalization before storing and before searching.


---

4) Keep metadata aligned with index positions

The order of metadata entries must match the order of vectors inserted into FAISS.

If this mapping breaks, search results will point to the wrong images.


---

5) Rebuild index when dataset changes

If images are added, removed, or embeddings are regenerated, rebuild the index unless incremental logic was explicitly implemented.


---

What files/artifacts must be preserved

The integrator must keep these generated outputs together:

FAISS index file

metadata mapping file

any saved config such as dimension, model name, normalization setting


These files should be loaded together during search.


---

How to run it in practice

Build phase

Run the index builder after dataset embeddings are ready.

Example flow:

1. dataset images


2. feature extractor


3. vectors + metadata


4. builddatasetindex.py


5. saved FAISS index + metadata




---

Search phase

Run the search script after query embedding is ready.

Example flow:

1. query image


2. same feature extractor


3. query vector


4. search_query.py


5. top-k similar results




---

What success looks like

The integration is correct if:

index builds without dimension/type errors

search loads index successfully

query returns top-k results

returned results map to the correct image files

similar images rank above unrelated images

same query gives stable results across runs



---

How the integrator can verify quickly

Smoke test

Use a query image that is:

already in the dataset

or very close to one in the dataset


Expected behavior:

the exact image or a very similar one appears at the top results


If not, check:

wrong embedding model

wrong normalization

wrong vector dimension

broken metadata mapping

wrong file loaded for FAISS index



---

Very important boundary

My code is not the full retrieval system.

It only gives the integrator:

indexed searchable vector storage

nearest-neighbor similarity search


The integrator still has to connect:

image ingestion

feature extraction

request handling

API/backend

frontend display

storage paths

logging and production error handling



---

One-line summary for the integrator

Use builddatasetindex.py once to turn dataset embeddings into a saved FAISS index, then use search_query.py to load that index and search it with a query embedding produced by the same feature extractor.


---

If you want, i can turn this into a cleaner handoff note written directly to the integrator in copy-paste form.
