import os
import json

import gradio as gr
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document


os.environ["OPENAI_API_KEY"] = "voc-8162499801266773377505669655d3c05508.40840521"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"


def load_data_into_vector_store(filename, vector_store, model_name):
    documents = []
    uuids = []
    with open(filename, "r") as file:
        i = 1
        for line in file:
            advert = json.loads(line)

            metadata = {}
            for k in advert.keys():
                if k not in ["property_description", "neighborhood_description"]:
                    metadata[k] = advert[k]
            metadata["source"] = model_name
            metadata["id"] = i
            i += 1
            page_content = advert["property_description"] + advert["neighborhood_description"]
        
            documents.append(
                Document(page_content=page_content, metadata=metadata)
            )
            uuids.append(str(uuid4()))
    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"Added {len(documents)} documents to the vector store.")


if __name__ == "__main__":
    model_name="gpt-3.5-turbo-instruct"
    vector_store_directory = "./chroma_langchain_db"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = Chroma(
        collection_name="real_estate",
        embedding_function=embeddings,
        persist_directory=vector_store_directory,  # Where to save data locally, remove if not necessary
    )
    
    count_documents = len(vector_store.get(where={"source": f"{model_name}"}))
    if count_documents == 0:
        load_data_into_vector_store(
            filename="./generated_adverts_b.jsonl",
            vector_store=vector_store,
            model_name=model_name
        )
    else:
        print(f"Already {count_documents} documents in the vector store.")

    def search(request):
        results = vector_store.similarity_search(
            request,
            k=2,
            filter={"source": model_name},
        )

        ##output = "\n\n".join([f"> {res.page_content} [{res.metadata}]" for res in results])
        output = "\n\n".join([f"> {res.page_content}" for res in results])
        print("output :", output)
        output = "{} advert(s) found:\n\n".format(len(results)) + output
        return output

    demo = gr.Interface(
        fn=search,
        inputs=[gr.Textbox(
            label="You request",
            lines=3,
            placeholder="What are you looking for?",
            value="2 bedroom apartment in New York",
            autofocus=True,
            submit_btn="Search...",
            max_length=300
            )],
        outputs=[gr.Textbox(label="Search result", lines=10)],
    )

    demo.launch()