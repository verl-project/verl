import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    comedians = [
        "Norm Macdonald",
        "John Mulaney",
        "Hasan Minhaj",
        "Dave Chappelle",
        "Ali Wong",
        "Chris Rock",
    ]
    formats = [
        "haiku",
        "one-liner",
        "q/a over sms",
    ]

    output_file = "humor_dataset.jsonl"
    model_name = "gpt-4o-mini"
    logger.info(f"Generating humor dataset to {output_file} using model {model_name}")

    with open(output_file, "w", encoding="utf-8") as fout:
        for comedian in comedians:
            for fmt in formats:
                question = (
                    f"What’s the best local LLM model to generate {fmt} jokes "
                    f"in the style of {comedian}? Please explain your reasoning step by step, "
                    f"and generate 3 example jokes."
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": question}],
                )
                answer = response.choices[0].message.content.strip()
                record = {
                    "comedian": comedian,
                    "format": fmt,
                    "question": question,
                    "response": answer,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(f"Wrote record: comedian={comedian}, format={fmt}")

    # Verify dataset count
    count = sum(1 for _ in open(output_file, encoding="utf-8"))
    logger.info(f"Dataset {output_file} contains {count} records")


if __name__ == "__main__":
    main()
