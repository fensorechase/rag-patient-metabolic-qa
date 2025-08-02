from torch import cuda, bfloat16
import transformers
import pandas as pd
import logging
import csv
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting script...")

model_id = 'meta-llama/Llama-3.1-8B-Instruct' # Meta-Llama-3-8B-Instruct

model_name = 'Meta_Llama_3_1_8B'
logger.info(f"Model ID set to {model_id}")

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
logger.info(f"Device set to {device}")

# Set quantization configuration to load large model with less GPU memory
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
logger.info("Quantization configuration set.")

# Begin initializing HF items, need auth token for these
hf_auth = '' # '<your_huggingface_token>'
logger.info("HF auth token set.")

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
logger.info("Model configuration loaded.")

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
logger.info(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
logger.info("Tokenizer loaded.")

# Load the questions CSV file
#questions_df = pd.read_csv('/local/scratch/ydiekma/final_outputs/rephrased_questions/FINAL_questions_rephrased_temp_0.6.csv')
# -- TODO: mkdir medquad_inputs/ (& put csv in there w/ 1. full prompt 2. raw question/headers 3. ground truth answers)
questions_df = pd.read_csv('/local/scratch/cfensor/rag-llm/medquad_inputs/medquad_NIDDK_qa_dataset.csv')

logger.info("CSV file loaded.")

# TODO -- NOTE: good follow-up to Yella paper could be comparing her LLM rephrasing vs. simple NLP robustness testing via replacement on medical QA datasets]
#-- ... With reddit QA: (i) use raw Q's & get gold & silver answers (ii) use profiles to do simulated "answers" from patients (based on the reddit data & interviews -- like this paper: https://arxiv.org/pdf/2405.06061

# Define the temperatures to iterate over
# -- TODO: only do 1 temperature
temperatures = [0.1] # [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Iterate over the different temperatures
for temp in temperatures:
    logger.info(f"Starting text generation with temperature: {temp}")
    
    # Create a text generation pipeline with the current temperature
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,
        task='text-generation',
        temperature=temp,
        max_new_tokens=512,
        do_sample=True
    )
    logger.info(f"Text generation pipeline created with temperature {temp}.")

    # Define output CSV file
    output_file = f'/local/scratch/cfensor/rag-llm/medquad_outputs/{model_name}_temp_{temp}.csv'
    
    # Prepare the output CSV file
    output_headers = [
        f'qid',
        f'qtype',
        f'unique_question_id',
        f'question_text',
        f'{model_name}_answer_vanilla', # No RAG yet. Make more columns which use RAG.
        f'answer_text', # Ground truth answer
        f'document_id',
        f'document_source',
        f'document_url',
        f'document_focus'
    ]
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_headers)

    # System role for generating answers to patient questions
    doctor_system_prompt = (
        "You are a helpful doctor answering patient questions. Your responses should be informative, concise, and clear."
    )

    # Iterate over each question in the CSV file
    logger.info("Starting iteration over questions...")
    for _, row in questions_df.iterrows():
        original_question = row['question_text']
        logger.info(f"Processing question: {original_question}")
        
        # Prepare the message structure
        messages = [
            {"role": "system", "content": doctor_system_prompt},
            {"role": "user", "content": original_question}
        ]
        
        # Generate response
        response = generate_text(messages)
        model_answer = response[0]["generated_text"].strip()

        # Create a row with all the required values
        output_row = [
            row['qid'],                    # qid
            row['qtype'],                  # qtype
            row['unique_question_id'],     # unique_question_id
            row['question_text'],          # question_text
            model_answer,                  # model_name_answer_vanilla
            row['answer_text'],            # Ground-truth: answer_text
            row['document_id'],            # document_id
            row['document_source'],        # document_source
            row['document_url'],           # document_url
            row['document_focus']         # document_focus
        ]


        # Save the results to the output CSV
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(output_row)

    logger.info(f"Generated answers saved to {output_file}.")

