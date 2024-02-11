# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

pip install similarities PyPDF2 -U
"""
import argparse
import hashlib
import os
import re
from threading import Thread
from typing import Union, List

import jieba
import torch
from loguru import logger
from peft import PeftModel
from similarities import (
    EnsembleSimilarity,
    BertSimilarity,
    BM25Similarity,
)
from similarities.similarity import SimilarityABC
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
    AutoConfig,
)

jieba.setLogLevel("ERROR")

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

PROMPT_TEMPLATE = """已知内容:
{context_str}
请根据以下内容做出回答:
{query_str}
"""


class SentenceSplitter:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if self._is_has_chinese(text):
            return self._split_chinese_text(text)
        else:
            return self._split_english_text(text)

    def _split_chinese_text(self, text: str) -> List[str]:
        sentence_endings = {'\n', '。', '！', '？', '；', '…'}  # 句末标点符号
        chunks, current_chunk = [], ''
        for word in jieba.cut(text):
            if len(current_chunk) + len(word) > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += word
            if word[-1] in sentence_endings and len(current_chunk) > self.chunk_size - self.chunk_overlap:
                chunks.append(current_chunk.strip())
                current_chunk = ''
        if current_chunk:
            chunks.append(current_chunk.strip())
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)
        return chunks

    def _split_english_text(self, text: str) -> List[str]:
        # 使用正则表达式按句子分割英文文本
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        chunks, current_chunk = [], ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size or not current_chunk:
                current_chunk += (' ' if current_chunk else '') + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:  # Add the last chunk
            chunks.append(current_chunk)

        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)

        return chunks

    def _is_has_chinese(self, text: str) -> bool:
        # check if contains chinese characters
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return True
        else:
            return False

    def _handle_overlap(self, chunks: List[str]) -> List[str]:
        # 处理块间重叠
        overlapped_chunks = []
        for i in range(len(chunks) - 1):
            chunk = chunks[i] + ' ' + chunks[i + 1][:self.chunk_overlap]
            overlapped_chunks.append(chunk.strip())
        overlapped_chunks.append(chunks[-1])
        return overlapped_chunks


class ChatPDF:
    def __init__(
            self,
            similarity_model: SimilarityABC = None,
            generate_model_type: str = "llama",
            generate_model_name_or_path: str = "/home/user/DISK/checkpoints/lmoe/ziya13bmerge",
            lora_model_name_or_path: str = None,
            corpus_files: Union[str, List[str]] = None,
            save_corpus_emb_dir: str = "./corpus_embs/",
            device: str = None,
            int8: bool = False,
            int4: bool = True,
            chunk_size: int = 250,
            chunk_overlap: int = 30,
    ):
        """
        Init RAG model.
        :param similarity_model: similarity model, default None, if set, will use it instead of EnsembleSimilarity
        :param generate_model_type: generate model type
        :param generate_model_name_or_path: generate model name or path
        :param lora_model_name_or_path: lora model name or path
        :param corpus_files: corpus files
        :param save_corpus_emb_dir: save corpus embeddings dir, default ./corpus_embs/
        :param device: device, default None, auto select gpu or cpu
        :param int8: use int8 quantization, default False
        :param int4: use int4 quantization, default False
        :param chunk_size: chunk size, default 250
        :param chunk_overlap: chunk overlap, default 50
        """
        if torch.cuda.is_available():
            default_device = torch.device(0)
        elif torch.backends.mps.is_available():
            default_device = 'mps'
        else:
            default_device = torch.device('cpu')
        self.device = device or default_device
        self.text_splitter = SentenceSplitter(chunk_size, chunk_overlap)
        if similarity_model is not None:
            self.sim_model = similarity_model
        else:
            m1 = BertSimilarity(model_name_or_path="/home/user/DISK/checkpoints/lmoe/text2vec", device=self.device)
            m2 = BM25Similarity()
            default_sim_model = EnsembleSimilarity(similarities=[m1, m2], weights=[0.5, 0.5], c=2)
            self.sim_model = default_sim_model
        self.gen_model, self.tokenizer = self._init_gen_model(
            generate_model_type,
            generate_model_name_or_path,
            peft_name=lora_model_name_or_path,
            int8=int8,
            int4=True,
        )
        self.history = []
        self.corpus_files = corpus_files
        if corpus_files:
            self.add_corpus(corpus_files)
        self.save_corpus_emb_dir = save_corpus_emb_dir

    def __str__(self):
        return f"Similarity model: {self.sim_model}, Generate model: {self.gen_model}"

    def _init_gen_model(
            self,
            gen_model_type: str,
            gen_model_name_or_path: str,
            peft_name: str = None,
            int8: bool = False,
            int4: bool = True,
    ):
        """Init generate model."""
        if int8 or int4:
            device_map = None
        else:
            device_map = "auto"
        model_class, tokenizer_class = MODEL_CLASSES[gen_model_type]
        tokenizer = tokenizer_class.from_pretrained(gen_model_name_or_path, trust_remote_code=True)
        config_kwargs = {
            "trust_remote_code": None,
            "cache_dir": None,
            # "revision": 'main',
            "use_auth_token": None,
            "output_hidden_states": True
        }
        config = AutoConfig.from_pretrained(gen_model_name_or_path, **config_kwargs)
        model = model_class.from_pretrained(
            gen_model_name_or_path,
            load_in_8bit=int8 if gen_model_type not in ['baichuan', 'chatglm'] else False,
            load_in_4bit=int4 if gen_model_type not in ['baichuan', 'chatglm'] else False,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
            config=config,
        )
        if self.device == torch.device('cpu'):
            model.float()
        if gen_model_type in ['baichuan', 'chatglm']:
            if int4:
                model = model.quantize(4).cuda()
            elif int8:
                model = model.quantize(8).cuda()
        # try:
        #     model.generation_config = GenerationConfig.from_pretrained(gen_model_name_or_path, trust_remote_code=True)
        # except Exception as e:
        #     logger.warning(f"Failed to load generation config from {gen_model_name_or_path}, {e}")
        if peft_name:
            model = PeftModel.from_pretrained(
                model,
                peft_name,
                torch_dtype="auto",
            )
            logger.info(f"Loaded peft model from {peft_name}")
        model.eval()
        return model, tokenizer

    def create_prompt(self, new_user_input, max_length):
        """
        构建包含历史会话的 prompt，并考虑输入长度限制。

        :param history: 一个包含历史消息的列表，每个元素是一个 (speaker, message) 元组。
        :param new_user_input: 新的用户输入。
        :param max_length: 允许的最大 token 数量。
        :return: 构建好的 prompt 字符串。
        """
        prompt = ""
        total_length = 0
        messages = []
        # 预估新输入的长度
        estimated_input_length = len(new_user_input) + len("用户：\n助手：")

        # 从最近的历史开始添加，确保最新的对话被包含
        for speaker, message in reversed(self.history):
            message_length = len(message) + len(f"{speaker}：") + 1  # 加上额外字符的长度

            if total_length + message_length + estimated_input_length <= max_length:
                prompt = f"{speaker}：{message}\n" + prompt
                total_length += message_length
            else:
                break  # 停止添加历史消息以避免超过长度限制

        # 添加新的用户输入
        prompt += f"用户：{new_user_input}\n助手："

        return prompt
    def _get_chat_input(self):
        messages = []
        for conv in self.history:
            if conv and len(conv) > 0 and conv[0]:
                messages.append({'role': 'user', 'content': conv[0]})
            if conv and len(conv) > 1 and conv[1]:
                messages.append({'role': 'assistant', 'content': conv[1]})
            # 历史缓冲区内容不大于4条数据，否则删除第一条数据
            if len(self.history) >= 3:
                self.history.pop(0)
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            max_length=1024,
            return_tensors='pt'
        )
        print(input_ids.shape)
        return input_ids.to(self.gen_model.device)

    def _get_chat_input_id(self):
        messages = []
        for conv in self.history:
            if conv and len(conv) > 0 and conv[0]:
                messages.append({'role': 'user', 'content': conv[0]})
            if conv and len(conv) > 1 and conv[1]:
                messages.append({'role': 'assistant', 'content': conv[1]})
            # 历史缓冲区内容不大于4条数据，否则删除第一条数据
            if len(self.history) >= 3:
                self.history.pop(0)
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            max_length=128,
            return_tensors='pt'
        )
        return input_ids.to(self.gen_model.device)



    def add_corpus(self, files: Union[str, List[str]]):
        """Load document files."""
        if isinstance(files, str):
            # files = [files]
            files = files
        print(files)
        for doc_file in os.listdir(files[0]):

            doc_file = os.path.join(files[0], doc_file)
            print(doc_file)
            if doc_file.endswith('.pdf'):
                corpus = self.extract_text_from_pdf(doc_file)
            elif doc_file.endswith('.docx'):
                corpus = self.extract_text_from_docx(doc_file)
            elif doc_file.endswith('.md'):
                corpus = self.extract_text_from_markdown(doc_file)
            else:
                corpus = self.extract_text_from_txt(doc_file)
                # corpus = self.extract_text_from_pdf(doc_file)
            full_text = '\n'.join(corpus)
            chunks = self.text_splitter.split_text(full_text)
            self.sim_model.add_corpus(chunks)
        self.corpus_files = files
        logger.debug(f"files: {files}, corpus size: {len(self.sim_model.corpus)}, top3: "
                     f"{list(self.sim_model.corpus.values())[:3]}")

    @staticmethod
    def get_file_hash(fpaths):
        hasher = hashlib.md5()
        target_file_data = bytes()
        if isinstance(fpaths, str):
            fpaths = [fpaths]
        for fpath in fpaths:
            with open(fpath, 'rb') as file:
                chunk = file.read(1024 * 1024)  # read only first 1MB
                hasher.update(chunk)
                target_file_data += chunk

        hash_name = hasher.hexdigest()[:32]
        return hash_name

    @staticmethod
    def extract_text_from_pdf(file_path: str):
        """Extract text content from a PDF file."""
        import PyPDF2
        contents = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text().strip()
                raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
                new_text = ''
                for text in raw_text:
                    new_text += text
                    if text[-1] in ['.', '!', '?', '。', '\\n','！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
                                    '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                        contents.append(new_text)
                        new_text = ''
                if new_text:
                    contents.append(new_text)
        return contents

    @staticmethod
    def extract_text_from_txt(file_path: str):
        """Extract text content from a TXT file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = [text.strip() for text in f.readlines() if text.strip()]
        return contents

    @staticmethod
    def extract_text_from_docx(file_path: str):
        """Extract text content from a DOCX file."""
        import docx
        document = docx.Document(file_path)
        contents = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return contents

    @staticmethod
    def extract_text_from_markdown(file_path: str):
        """Extract text content from a Markdown file."""
        import markdown
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        html = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html, 'html.parser')
        contents = [text.strip() for text in soup.get_text().splitlines() if text.strip()]
        return contents

    @staticmethod
    def _add_source_numbers(lst):
        """Add source numbers to a list of strings."""
        return [f'[{idx + 1}]\t "{item}"' for idx, item in enumerate(lst)]

    def predict(
            self,
            querys,
            topn: int = 5,
            max_length: int = 128,
            context_len: int = 2048,
            temperature: float = 0.7,
            do_print: bool = False,

    ):
        """Query from corpus."""
        reference_results = []
        self.history = []
        for query in querys:
            if self.sim_model.corpus:
                sim_contents = self.sim_model.most_similar(query, topn=1)

                # Get reference results
                for query_id, id_score_dict in sim_contents.items():
                    for corpus_id, s in id_score_dict.items():
                        reference_results.append(self.sim_model.corpus[corpus_id])

                if not reference_results:
                    return '没有提供足够的相关信息', reference_results
                context_str = '\n'.join(reference_results)[:(context_len - len(PROMPT_TEMPLATE))]
                prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=query)
                # logger.debug(f"prompt: {prompt}")
            else:
                prompt = query
            self.history.append(('user', prompt))
            print(prompt)
            prompt = self.create_prompt( prompt, max_length=1024)
            # input_ids = self._get_chat_input()
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            generate_ids = self.gen_model.generate(
                input_ids,
                max_new_tokens=2048,
                do_sample=True,
                top_p=0.85,
                temperature=temperature,
                repetition_penalty=1.0
            )
            response = self.tokenizer.batch_decode(generate_ids)[0]
            # ,skip_special_tokens=True
            self.history.append(('assistant', response))
            # self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            response = response.strip()
            # self.history[-1][1] = response

            if len(self.history) == 4:
                embedding = self.gen_model(input_ids, output_hidden_states=True)
                hidden_state = embedding.hidden_states
                hidden_state = torch.mean(hidden_state[0].squeeze(0) + hidden_state[-1].squeeze(0), axis=0)
                hidden_state = hidden_state.unsqueeze(0)
                del(self.history[0])
                return response, reference_results, hidden_state

        return response, reference_results, _

    def save_corpus_emb(self):
        dir_name = self.get_file_hash(self.corpus_files)
        save_dir = os.path.join(self.save_corpus_emb_dir, dir_name)
        if hasattr(self.sim_model, 'save_corpus_embeddings'):
            self.sim_model.save_corpus_embeddings(save_dir)
            logger.debug(f"Saving corpus embeddings to {save_dir}")
        return save_dir

    def load_corpus_emb(self, emb_dir: str):
        if hasattr(self.sim_model, 'load_corpus_embeddings'):
            logger.debug(f"Loading corpus embeddings from {emb_dir}")
            self.sim_model.load_corpus_embeddings(emb_dir)

#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--sim_model", type=str, default="/home/user/DISK/checkpoints/lmoe/text2vec")
#     parser.add_argument("--gen_model_type", type=str, default="llama")
#     parser.add_argument("--gen_model", type=str, default="/home/user/DISK/checkpoints/lmoe/ziya13bmerge")
#     parser.add_argument("--lora_model", type=str, default=None)
#     parser.add_argument("--corpus_files", type=str, default="/home/user/git_code/LMOE/knowledge_floder")
#     parser.add_argument("--device", type=str, default=None)
#     parser.add_argument("--int4", action='store_false', help="use int4 quantization")
#     parser.add_argument("--int8", action='store_true', help="use int8 quantization")
#     parser.add_argument("--chunk_size", type=int, default=100)
#     parser.add_argument("--chunk_overlap", type=int, default=5)
#     args = parser.parse_args()
#     print(args)
#     sim_model = BertSimilarity(model_name_or_path=args.sim_model, device=args.device)
#     m = ChatPDF(
#         similarity_model=sim_model,
#         generate_model_type=args.gen_model_type,
#         generate_model_name_or_path=args.gen_model,
#         lora_model_name_or_path=args.lora_model,
#         device=args.device,
#         int4=args.int4,
#         int8=args.int8,
#         chunk_size=args.chunk_size,
#         chunk_overlap=args.chunk_overlap,
#         corpus_files=args.corpus_files.split(','),
#     )
#     query = [
#         "病理诊断是否有肺癌?",
#         " CT检测结果是什么?",
#         "EGFR Exon19是否基因突变？",
#         "EGFR Exon21是否基因突变？",
#         "请给根据以上诊断结果，说明该肺癌由哪个基因突变引起？"
#     ]
#     response, reference_results,hidden_state = m.predict(query)
#     print(response,hidden_state.shape)
#
