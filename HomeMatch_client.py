import json
import argparse
from uuid import uuid4
from time import sleep
import random

import gradio as gr
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


from dotenv import load_dotenv
if not load_dotenv(): # keys are loaded from .env file
    print("Warning: .env file not found. Make sure to set environment variables manually.")
    exit(1)


def load_data_into_vector_store(
        filename: str, vector_store: Chroma, model_name: str
        ) -> None:
    """
    Load data from a JSONL file into the vector store.
    Args:
        filename (str): Path to the JSONL file.
        vector_store: The vector store instance.
        model_name (str): Name of the model used for embeddings.
    """
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


class PropertySearchRequestClass(BaseModel):
    """
    Class to define the schema of the property search request.
    """
    location: str = Field(
        description = "location in USA including the name the neighborhood"
    )
    style: str = Field(
        description = "style of construction"
    )
    rooms: int = Field(
        description = "number of rooms"
    )
    bedrooms: int = Field(
        description = "number of bedrooms"
    )
    bathrooms: int = Field(
        description = "number of bathrooms"
    )
    floors: int = Field(
        description = "number of floors"
    )
    house_size: int = Field(
        description = "surface area in square feet"
    )
    price: int = Field(
        description = "price in dollars"
    )


def get_analyze_query(user_search_request: str) -> str:
    """
    Generate a query to analyze the user's search request.
    Args:
        user_search_request (str): The user's search request.
    Returns:
        str: The query to analyze the search request.
    """
    return f"""
find all available information as defined in the following output schema from this request :"{user_search_request}", in order to generate a corresponding filled json data structure. set -1 for any missing information.
"""


def summarize_search_request(parsed_response: PropertySearchRequestClass) -> str:
    """
    Summarize the parsed search request.
    Args:
        parsed_response (PropertySearchRequestClass): The parsed search request.
    Returns:
        str: The summary of the search request.
    """
    summary = []
    for k, v in vars(parsed_response).items():
        if v != -1:
            if k == "location":
                summary.append(f"located in {v}")
            elif k == "style":
                summary.append(f"{v}-style")
            elif k == "rooms":
                summary.append(f"{v} room(s)")
            elif k == "bedrooms":
                summary.append(f"{v} bedroom(s)")
            elif k == "bathrooms":
                summary.append(f"{v} bathroom(s)")
            elif k == "floors":
                summary.append(f"{v} floors")
            elif k == "house_size":
                summary.append(f"{v} square feet size")
            elif k == "price":
                summary.append(f"priced ${v}")

    summary = ", ".join(summary)
    return "Your are looking a house like this : " + summary


def get_response_augmentation_request(
        advertisement: object, search_request: str
        ) -> str:
    """
    Generate a request to augment the advertisement description.
    Args:
        advertisement (object): The advertisement to augment.
        search_request (str): The user's search request.
    Returns:
        str: The request to augment the advertisement description.
    """
    start_with_list = ["Welcome to", "This wonderful", "This fantastic", "This awesome", "This incredible"]
    start_with = random.choice(start_with_list)
    return f"""create a new description from this real estate advertisement {advertisement} in order to show that it matches the following search request: {search_request}, with respect to the information included in the advertisement. Add arguments in the new description to make it more attractive. it must be in a natural language format, and must not be a json object. The advertisement should be at least 100 words long, in only one paragraph. All the text is in English. Start with '{start_with}!'"""


parser = PydanticOutputParser(pydantic_object=PropertySearchRequestClass)
analyzing_prompt_generator = PromptTemplate(
    template="{query}\n{format_instructions}",
    input_variables=["query", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions},
)
def search_in_vector_db(
        user_search_request: str,
        n_answers: int,
        llm: OpenAI,
        vector_store: Chroma,
        verbose: bool = False):
    """
    Search in the vector database and return the results.
    Args:
        user_search_request (str): The user's search request.
        n_answers (int): The number of answers to return.
        llm (OpenAI): The language model instance.
        vector_store (Chroma): The vector store instance.
        verbose (bool): Whether to print verbose output.
    Returns:
        str: The search results.
    """
    # Analyze and summazyze the user search request
    prompt = analyzing_prompt_generator.format(query=get_analyze_query(user_search_request))
    analyzed_request = llm.invoke(prompt)
    print("Analyzed request: ", analyzed_request)
    parsed_response = parser.parse(analyzed_request)
    print("Parsed response: ", parsed_response)

    summary = summarize_search_request(parsed_response)

    # search in the vector store
    results = vector_store.similarity_search(
        user_search_request,
        k=n_answers,
        filter={"source": model_name},
    )

    if verbose:
        verbose_output = "\n\n".join([f"> {res.page_content}" for res in results])
        print("output :", verbose_output)

    count = "{} advert(s) found:\n\n".format(len(results))

    # Augment the advertisement descriptions
    augmented_adverts = []
    for i, result in enumerate(results, start=1):
        metadata = result.metadata
        metadata.pop('source', None)
        metadata.pop('id', None)
        advertisement = f"{result.page_content} [{metadata}]"
        augmented_advert = llm.invoke(
            get_response_augmentation_request(advertisement, user_search_request)
            )
        augmented_adverts.append(f"#{i}: " + augmented_advert[2:])
        if verbose:
            print("> augmented :", augmented_adverts[-1], end="\n\n")
        sleep(2)

    return summary + "\n\n" + count + "\n\n".join(augmented_adverts)


def parsing_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HomeMatch Gradio app.")

    parser.add_argument(
        "--filename",
        type=str,
        default="./listings.jsonl",
        help="Path to the JSONL file containing the data",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="gpt-3.5-turbo-instruct",
        help="Name of the model to use for embeddings",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.01,
        help="Temperature for the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=3500,
        help="Maximum number of tokens for the model",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the test mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parsing_args()

    filename = args.filename
    model_name = args.model_name
    temperature = args.temperature
    max_tokens = args.max_tokens
    
    vector_store_directory = "./chroma_langchain_db"  # Directory to store the vector store data

    # Instantiate LLM and embeddings
    llm = OpenAI(
        model_name=model_name, temperature=temperature, max_tokens=max_tokens
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Initialize the vector store
    vector_store = Chroma(
        collection_name="real_estate",
        embedding_function=embeddings,
        persist_directory=vector_store_directory
    )
    
    # Check if the vector store is already populated
    count_documents = len(vector_store.get(where={"source": f"{model_name}"})["ids"])
    if count_documents == 0:
        load_data_into_vector_store(
            filename=filename,
            vector_store=vector_store,
            model_name=model_name
        )
    else:
        print(f"Already {count_documents} documents in the vector store.")

    sample_user_search_request = "I'm looking for a modern-style house with a sea view for a large family. The house should be located in a quiet neighborhood in San Francisco, California. It should have at least 2 bathrooms and a large garden. The total area of the house should be at least 2500 square feet, and the price should not exceed $500,000."

    # Define the Gradio interface
    search = lambda user_search_request, n_answers: search_in_vector_db(
        user_search_request,
        n_answers,
        llm=llm,
        vector_store=vector_store,
        verbose=args.verbose
    )
    demo = gr.Interface(
        fn=search,
        inputs=[
            gr.Textbox(
                label="Your request:",
                lines=7,
                placeholder="What are you looking for?",
                value=sample_user_search_request if args.test else "",
                autofocus=True,
                #submit_btn="Search...",
                max_length=300
            ),
            gr.Slider(1, 5, 2, label="Number of results", step=1),
            ],
        outputs=[gr.Textbox(label="Search result", lines=20)],
    )

    demo.launch()