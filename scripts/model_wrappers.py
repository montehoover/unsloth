import asyncio
import time
import json
import uuid

import litellm
import openai
import datasets
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from constants import COT_OPENING, LABEL_OPENING

import logging
logger = logging.getLogger(__name__)

# Suppress annoying warning messages about new Gemini models it isn't aware of
litellm._logging._disable_debugging()
# litellm.set_verbose=True

class ComplianceProjectError(ValueError):
    pass

class ModelWrapper:
    def get_message_template(self, system_content=None, user_content=None, assistant_content=None):
        assistant_content = assistant_content or LABEL_OPENING
        if system_content is None:
            return [{'role': 'user', 'content': user_content}]
        else:
            return [
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': assistant_content},
            ]

    def get_message_template_cot(self, system_content, user_content, assistant_content=None):
        assistant_content = assistant_content or COT_OPENING
        return [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_content},
        ]

    def apply_chat_template(self, system_content=None, user_content=None, assistant_content=None):
        return self.get_message_template(system_content, user_content, assistant_content)
    
    def apply_chat_template_cot(self, system_content, user_content, assistant_content=None):
        return self.get_message_template_cot(system_content, user_content, assistant_content)


class LocalModelWrapper(ModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=300, max_new_tokens=1000):
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def apply_chat_template(self, system_content, user_content, assistant_content=None):
        message = super().apply_chat_template(system_content, user_content, assistant_content)
        try:
            # prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
        except Exception as e:
            print(message)
            raise
        return prompt
    
    def apply_chat_template_cot(self, system_content, user_content, assistant_content=None):
        message = super().apply_chat_template_cot(system_content, user_content, assistant_content)
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
        return prompt



class HfModelWrapper(LocalModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=300, max_new_tokens=1000):
        super().__init__(model_name, temperature, top_k, max_new_tokens)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        )

    def get_response(self, message):
        inputs = self.tokenizer(message, return_tensors="pt").to(self.model.device)
        output_content = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            temperature=self.temperature,
            top_k=self.top_k,
            pad_token_id=self.tokenizer.pad_token_id
        )
        output_text = self.tokenizer.decode(output_content[0], skip_special_tokens=True)
        return output_text

    def get_responses(self, messages):
        outputs = [self.get_response(message) for message in tqdm(messages)]
        return outputs


class VllmModelWrapper(LocalModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=300, max_new_tokens=1000, max_model_len=8192):
        from vllm import LLM, SamplingParams

        super().__init__(model_name, temperature, top_k, max_new_tokens)
        self.model = LLM(model_name, max_model_len=max_model_len, gpu_memory_utilization=0.95)

    def get_responses(self, messages):
        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            seed=time.time_ns(),
        )
        # responses -> List[obj(prompt, outputs -> List[obj(text, ???)])]
        responses = self.model.generate(messages, sampling_params=sampling_params)
        outputs = [response.outputs[0].text for response in responses]
        return outputs


class ApiModelWrapper(ModelWrapper):
    def __init__(self, model_name, temperature=0.6, api_delay=None, retries=3, max_batch_size=100, max_new_tokens=1000, log_interval=100):
        self.model_name = f"gemini/{model_name}" if "gemini" in model_name else model_name
        self.temperature = temperature
        self.delay = api_delay if api_delay is not None else (4 if "gemini" in model_name else 0)
        self.retries = retries
        self.max_new_tokens = max_new_tokens
        self.last_call_time = None
        self.log_interval = log_interval
        self.max_batch_size = max_batch_size
        self.completion_count = 0
        self.lock = asyncio.Lock()

    def get_response(self, message):
        success = False
        for _ in range(self.retries):
            try:
                current_time = time.time()
                if self.last_call_time:
                    elapsed_time = current_time - self.last_call_time
                    if elapsed_time < self.delay:
                        time.sleep(self.delay - elapsed_time)

                self.last_call_time = time.time()
                response = litellm.completion(
                    model=self.model_name,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    messages=message
                )
                if response['choices'][0]['message']['content'] is None:
                    logger.warning(f"Empty completion due to 'finish reason={response['choices'][0]['finish_reason']}'")
                    continue
                else:
                    success = True
                    break
            except (litellm.ContentPolicyViolationError, openai.APIError) as e:
                if isinstance(e, openai.APIError) and not litellm._should_retry(e.status_code):
                    raise
                else:
                    logger.warning(f"API call error: {e}")
        if not success:
            raise ComplianceProjectError(f"Failed after {self.retries} retries.")
        return response['choices'][0]['message']['content']

    async def aget_response(self, message):
        success = False
        for _ in range(self.retries):
            try:
                response = await litellm.acompletion(
                    model=self.model_name,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    messages=message,
                )
                success = True
                break
            except (litellm.ContentPolicyViolationError, openai.APIError) as e:
                if isinstance(e, openai.APIError) and not litellm._should_retry(e.status_code):
                    raise
        if not success:
            raise ComplianceProjectError(f"Failed after {self.retries} retries.")
        
        async with self.lock:
            self.completion_count += 1
            if self.completion_count % self.log_interval == 0:
                logger.info(f"Completed {self.completion_count} API calls.")
        
        return response['choices'][0]['message']['content']

    async def gather_responses(self, messages):
        tasks = [self.aget_response(message) for message in tqdm(messages)]
        logger.info(f"Starting round of {len(tasks)} API calls...")
        results = await asyncio.gather(*tasks)
        logger.info(f"Completed round of {len(tasks)} API calls.")
        self.completion_count = 0
        return results
    
    def get_responses(self, messages):
        if self.delay is None or self.delay == 0:
            # batch_size = self.max_batch_size
            # completed = 0
            # outputs = []
            # while completed < len(messages):
            #     batch = messages[completed:completed+batch_size]
            #     outputs += asyncio.run(self.gather_responses(batch))
            #     completed += batch_size
            outputs = asyncio.run(self.gather_responses(messages))
        else:
            outputs = [self.get_response(message) for message in tqdm(messages)]
        return outputs

class BatchApiModelWrapper(ModelWrapper):
    def __init__(self, model_name, temperature=0.6, max_batch_size=100, query_interval=60, log_interval=30):
        self.model_name = model_name
        self.temperature = temperature
        self.client = openai.Client()
        self.query_interval = query_interval
        self.log_interval = log_interval
        self.max_batch_size = max_batch_size
        unique_id = str(uuid.uuid4())
        self.input_filename = f"batch/batch_input_{unique_id}.jsonl"
        self.output_filename = f"batch/batch_output_{unique_id}.jsonl"

    def create_jsonl_file(self, messages):
        requests = []
        for i, message in enumerate(messages):
            request = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "messages": message,
                }
            }
            requests.append(request)
        dataset = datasets.Dataset.from_list(requests)
        dataset.to_json(self.input_filename)

    def save_jsonl_file(self, jsonl_result):
        with open(self.output_filename, "wb") as f:
            f.write(jsonl_result)

    def monitor_status_until_completion(self, batch_job_id):
        logger.info(f"Querying status for 24 hours, checking every {self.query_interval} seconds...")
        hr, min, sec = 24, 60, 60
        for i in tqdm(range(hr * min * sec // self.query_interval)):
            batch_job = self.client.batches.retrieve(batch_job_id)
            if batch_job.status in ["completed", "failed", "expired", "cancelled"]:
                break
            elif batch_job.status in ["validating", "finalizing", "cancelling"]:
                logger.info(f"Batch job status: {batch_job.status}")
            elif batch_job.status == "in_progress":
                pass
            else:
                raise ComplianceProjectError(f"Unexpected batch job status: {batch_job.status}")
            if i % self.log_interval == 0:
                logger.info(f"Batch job status: {batch_job.status}")
            time.sleep(self.query_interval)
        return batch_job
    
    def get_responses_from_finished_job(self, batch_job_id):
        batch_job = self.client.batches.retrieve(batch_job_id)
        jsonl_result = self.client.files.content(batch_job.output_file_id).content
        self.save_jsonl_file(jsonl_result)
        results = datasets.load_dataset("json", data_files={"placeholder": self.output_filename})["placeholder"]
        # Results object keys: {'custom_id' -> str, 'response' -> str, 'error' -> ?, 'id' -> ?}
        # The output json is not guaranteed to be in the same order as the input, so we check the index of each response and insert it into the correct spot in outputs
        outputs = [""] * batch_job.request_counts.total
        for result in results:
            index = int(result["custom_id"])
            outputs[index] = result["response"]["body"]["choices"][0]["message"]["content"]
        return outputs

    def get_responses(self, messages):
        batch_size = self.max_batch_size
        completed = 0
        outputs = []
        while completed < len(messages):
            batch = messages[completed:completed+batch_size]
        
            self.create_jsonl_file(batch)
            openai_file_obj = self.client.files.create(
                file=open(self.input_filename, "rb"), 
                purpose="batch"
            )
            logger.warning(f"OpenAI file object created. File ID: {openai_file_obj.id}")
            batch_job = self.client.batches.create(
                input_file_id=openai_file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            logger.warning(f"Batch job created. Batch ID: {batch_job.id}")
            
            completed_batch_job = self.monitor_status_until_completion(batch_job.id)
            if completed_batch_job.status != "completed":
                raise ComplianceProjectError(f"Batch job failed with status {completed_batch_job.status}: \n{(json.dumps(completed_batch_job.dict(), indent=4))}")

            # After successful completion
            outputs += self.get_responses_from_finished_job(batch_job.id)
            completed += batch_size
        return outputs