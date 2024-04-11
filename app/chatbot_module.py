import os
import torch
from langchain import PromptTemplate, HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

class Chatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_template = """
            I'm your friendly NLP chatbot named AIT-GPT, here to assist students with any questions they have about AIT.
            Just let me know what you're wondering about, and I'll do my best to guide you through it!
            {context}
            Question: {question}
            Answer:
        """.strip()
        self.PROMPT = PromptTemplate.from_template(template=self.prompt_template)
        
        # Load embedding model
        model_name = 'hkunlp/instructor-base'
        self.embedding_model = HuggingFaceInstructEmbeddings(model_name=model_name, model_kwargs={"device": self.device})
        
        # Load vector database
        vector_path = '../vector-storage'
        db_file_name = 'nlp_stanford'
        self.vectordb = FAISS.load_local(folder_path=os.path.join(vector_path, db_file_name), embeddings=self.embedding_model, index_name='nlp')
        self.retriever = self.vectordb.as_retriever()
        
        # Load language model with CPU offloading
        model_id = '../models/fastchat-t5-3b-v1.0/'
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        bitsandbyte_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            load_in_8bit_fp32_cpu_offload=True  # Enable CPU offloading
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, 
            # quantization_config=bitsandbyte_config, 
            # device_map='auto',  # Automatically distribute model parts across available devices
            # load_in_8bit=True
        )
        self.pipe = pipeline(
            task="text2text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=256, 
            model_kwargs={"temperature": 0, "repetition_penalty": 1.5}
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        # Setup chains and memory
        self.setup_chains_and_memory()

    def setup_chains_and_memory(self):
        self.question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
        self.doc_chain = load_qa_chain(llm=self.llm, chain_type='stuff', prompt=self.PROMPT, verbose=True)
        self.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True, output_key='answer')
        self.chain = ConversationalRetrievalChain(
            retriever=self.retriever, 
            question_generator=self.question_generator, 
            combine_docs_chain=self.doc_chain, 
            return_source_documents=True, 
            memory=self.memory, 
            verbose=True, 
            get_chat_history=lambda h: h
        )

    def ask(self, question):
        answer = self.chain({"question": question})
        processed_answer = answer['answer'].replace('<pad>', '').strip()  # Remove <pad> tokens and strip whitespace
        return processed_answer
