import os
import sys
import json
import time
import logging
import pymupdf

from models import JSONResume
from llm_utils import initialize_llm_provider, extract_json_from_response
from pymupdf_rag import to_markdown
from typing import Optional, Dict
from prompt import (
    DEFAULT_MODEL,
    MODEL_PARAMETERS,
)
from prompts.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class PDFHandler:
    def __init__(self):
        self.template_manager = TemplateManager()
        self._initialize_llm_provider()

    def _initialize_llm_provider(self):
        """Initialize the appropriate LLM provider based on the model."""
        self.provider = initialize_llm_provider(DEFAULT_MODEL)

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            with pymupdf.open(pdf_path) as doc:
                pages = range(doc.page_count)
                resume_text = to_markdown(
                    doc,
                    pages=pages,
                )
                logger.debug(
                    f"Extracted text from PDF: {len(resume_text) if resume_text else 0} characters"
                )
                return resume_text
        except Exception as e:
            logger.error(f"An error occurred while reading the PDF: {e}")
            return None

    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """Single LLM call that returns the whole resume as a JSON dict."""
        try:
            start_time = time.time()

            model_params = MODEL_PARAMETERS.get(
                DEFAULT_MODEL, {"temperature": 0.1, "top_p": 0.9}
            )

            system_message = self.template_manager.render_template(
                "full_system_message"
            )
            if not system_message:
                logger.error("❌ Failed to render full system message template")
                return None

            chat_params = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "stream": False,
                    "temperature": model_params["temperature"],
                    "top_p": model_params["top_p"],
                },
            }

            response = self.provider.chat(
                **chat_params, format=JSONResume.model_json_schema()
            )

            response_text = response["message"]["content"]
            response_text = extract_json_from_response(response_text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}")
            if json_start != -1 and json_end != -1:
                response_text = response_text[json_start : json_end + 1]
            parsed_data = json.loads(response_text)

            elapsed = time.time() - start_time
            print(f"   ✅ 简历板块提取完成（耗时 {elapsed:.0f} 秒）")
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error calling LLM: {e}")
            return None

    def extract_all_sections(self, resume_text: str) -> Optional[JSONResume]:
        """Extract the entire resume in a single LLM call."""
        prompt = self.template_manager.render_template(
            "all_sections", text_content=resume_text
        )
        if not prompt:
            logger.error("❌ Failed to render all_sections template")
            return None

        parsed_data = self._call_llm(prompt)
        if parsed_data is None:
            logger.warning("🔁 提取失败，重试一次...")
            parsed_data = self._call_llm(prompt)
        if parsed_data is None:
            logger.error("❌ 简历提取失败，无法继续")
            return None

        try:
            return JSONResume(**parsed_data)
        except Exception as e:
            logger.error(f"❌ 简历数据校验失败: {e}")
            return None

    def extract_json_from_text(self, resume_text: str) -> Optional[JSONResume]:
        return self.extract_all_sections(resume_text)

    def extract_json_from_pdf(self, pdf_path: str) -> Optional[JSONResume]:
        try:
            text_content = self.extract_text_from_pdf(pdf_path)
            if not text_content:
                logger.error("❌ Failed to extract text from PDF")
                return None

            print(
                f"   ✅ 已读取 PDF 文本（{len(text_content)} 字符），开始提取简历板块..."
            )
            return self.extract_all_sections(text_content)
        except Exception as e:
            logger.error(f"❌ Error during PDF to JSON extraction: {e}")
            return None
