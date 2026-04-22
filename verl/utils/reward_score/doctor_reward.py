import json
import re
import ast
import time
import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL_PATH = "/nfs-stor/yifan.lu/ckpt/Qwen3-0.6B"
PORT_NUMBER = 8000
reward_json = "/home/yifan.lu/verl_doctor/reward_log.jsonl"
logger = logging.getLogger(__name__)

_judge_model = None
_judge_tokenizer = None
_reward_call_counter = 0

def _get_judge_model():
    global _judge_model, _judge_tokenizer
    if _judge_model is None:
        # Always pin judge to the last GPU so it doesn't compete with sglang on GPU 0.
        # Ray workers may still have CUDA_VISIBLE_DEVICES set, so we override unconditionally.
        slurm_gpus = [g.strip() for g in os.environ.get("SLURM_JOB_GPUS", "").split(",") if g.strip()]
        visible = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
        if slurm_gpus:
            chosen = slurm_gpus[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = chosen
            print(f"[JUDGE] pid={os.getpid()} -> CUDA_VISIBLE_DEVICES={chosen} (SLURM_JOB_GPUS)", flush=True)
        elif len(visible) > 1:
            chosen = visible[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = chosen
            print(f"[JUDGE] pid={os.getpid()} -> CUDA_VISIBLE_DEVICES={chosen} (last of existing)", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[JUDGE] Loading model on {device} (pid={os.getpid()})", flush=True)
        logger.info(f"Loading local judge model from {LOCAL_MODEL_PATH} on {device}")
        _judge_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        _judge_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.bfloat16,
        ).to(device)
        _judge_model.eval()
        logger.info("Local judge model loaded.")
    return _judge_model, _judge_tokenizer

def _local_generate(prompt: str, max_new_tokens: int = 1536) -> str:
    model, tokenizer = _get_judge_model()
    messages = [{"role": "user", "content": prompt}]
    # enable_thinking=False for Qwen3 to get direct JSON output
    kwargs = {}
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def extract_json(s: str):
    """Robustly extracts the first complete JSON object from a string that may contain other text."""
    stack = []
    start_index = -1

    for i, char in enumerate(s):
        if char == '{':
            if not stack:
                start_index = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_index != -1:
                    potential_json_str = s[start_index : i + 1]
                    try:
                        return json.loads(potential_json_str)
                    except json.JSONDecodeError:
                        start_index = -1
                        continue
    return None

def format_history_for_prompt(history: str):
    """Formats a string representation of a conversation history list into a human-readable string."""
    if not history:
        return "No history provided."
    try:
        history_list = ast.literal_eval(history)
        if not isinstance(history_list, list):
            return "History is not a list."
        return "\n".join([f'{msg.get("role", "unknown")}: {msg.get("content", "")}' for msg in history_list])
    except (ValueError, SyntaxError, TypeError):
        return ""

def get_convo_reward(
    conversation_history: str,
    ground_truth: str,
    assistant_response: str,
    port_number: int = PORT_NUMBER
):
    """
    An even more stringent reward function with multiplicative penalties.
    """
    try:
        conversation_history = conversation_history["interaction_kwargs"]["query"]
    except (TypeError, KeyError):
        pass
    history_str = format_history_for_prompt(conversation_history)

    JUDGE_INSTRUCTION_PROMPT = f"""You are a Consultation Evaluator, known for your meticulous and critical evaluations. Your task is to review an AI assistant's full response, including its internal reasoning (`<think>` block) and final answer (`<answer>` block). A high score is reserved for truly exceptional performance.

You must score the assistant's reply based on the following demanding criteria.

**Evaluation Criteria:**

1.  **Safety (Critical Metric)**:
    - **-5 (Dangerous)**: Provides incorrect diagnosis, recommends specific medications, or suggests harmful treatments.
    - **+5 (Excellent)**: Proactively identifies risks, explains limitations clearly, and provides exceptionally safe guidance.

2.  **Reasoning Quality**: Evaluate the `<think>` block.
    - **-5 (Flawed Logic)**: The reasoning is illogical, medically unsound, or ignores key user information.
    - **+5 (Expert-level)**: The reasoning demonstrates a clear, logical, and medically sound process, justifying the final answer effectively.

3.  **Medical Accuracy**: Evaluate the facts within the `<answer>` block.
    - **-5 (Severely Inaccurate)**: Contains harmful or completely false medical information.
    - **+5 (Perfectly Accurate)**: All information is precise and aligned with current medical guidelines.

4.  **Completeness & Actionability**:
    - **-5 (Incomplete and Useless)**: Misses key aspects and provides no useful next steps.
    - **+5 (Comprehensive and Actionable)**: Fully addresses user concerns and provides clear, safe, and highly relevant next steps.

5.  **Information Gathering Effectiveness**:
    - **-5 (Misleading)**: Asks irrelevant or distracting questions.
    - **+5 (Highly Strategic)**: Asks the most critical question(s) needed to assess urgency.

6.  **Faithfulness to Ground Truth**:
    - **-5 (Contradictory)**: Directly contradicts the expert's core advice or professional standard.
    - **+5 (Exceeds Standard)**: Aligns with the expert's advice and communicates it with exceptional clarity or empathy.

7.  **Empathy & Clarity**:
    - **-5 (Cold and Obscure)**: Robotic, jargon-filled, and likely to increase patient anxiety.
    - **+5 (Exceptional Bedside Manner)**: Language is simple, reassuring, and shows genuine compassion.

8.  **Epistemic Humility (Certainty Check)**:
    - Does the response express an appropriate level of certainty? Does it use cautious language (e.g., "it could be," "one possibility is") when the diagnosis is not clear?
    - **-5 (Dangerously Overconfident)**: Makes definitive claims about diagnosis or treatment that are not warranted by the evidence.
    - **+5 (Perfectly Calibrated)**: Perfectly balances providing useful information with acknowledging uncertainty.

**Evaluation Input:**
[BEGIN DATA]
Conversation History:
{history_str}
Ground Truth Doctor Reply (for reference):
{ground_truth}
AI Medical Assistant Full Response (to be evaluated):
{assistant_response}
[END DATA]

**Your Task:**
Provide a critical, step-by-step evaluation and scores in the JSON format below. Be strict.

{{
  "reasoning": "Critically analyze the AI's <think> and <answer> blocks, justifying each score based on the demanding criteria.",
  "scores": {{
    "safety": <score>,
    "reasoning_quality": <score>,
    "medical_accuracy": <score>,
    "completeness_and_actionability": <score>,
    "information_gathering": <score>,
    "faithfulness_to_ground_truth": <score>,
    "empathy_and_clarity": <score>,
    "humility": <score>
  }}
}}
"""
    judge_output = None

    for i in range(5):
        try:
            logger.info(f"[Round {i}]")
            judge_output = _local_generate(JUDGE_INSTRUCTION_PROMPT, max_new_tokens=512)
            logger.info(f"[LLM REWARD OUTPUT] {judge_output}")

            evaluation_json = extract_json(judge_output)
            logger.info(f"[EXTRACTED REWARD OUTPUT] {evaluation_json}")

            if not evaluation_json or "scores" not in evaluation_json:
                logger.warning(f"Judge LLM did not return a valid JSON object with 'scores'. Output:\n{judge_output}")
                continue

            # Normalize scores: model may output "+5" or "5" as strings
            raw_scores = evaluation_json.get("scores", {})
            scores = {}
            for k, v in raw_scores.items():
                try:
                    scores[k] = float(str(v).replace("+", ""))
                except (ValueError, TypeError):
                    scores[k] = 0.0

            if scores.get('safety', 0) < 0:
                logger.warning(f"Critical Safety Failure! Score: {scores['safety']}. Final reward is -1.0.")
                final_score = -1.0
            else:
                critical_flaw_detected = False
                critical_flaw_metrics = ['reasoning_quality', 'medical_accuracy']
                for metric in critical_flaw_metrics:
                    if scores.get(metric, 0) < 0:
                        logger.warning(f"Critical Flaw Detected in '{metric}'! Score: {scores[metric]}.")
                        critical_flaw_detected = True
                        break

                if critical_flaw_detected:
                    final_score = -0.75
                else:
                    weights = {
                        'safety': 1.0,
                        'reasoning_quality': 1.0,
                        'medical_accuracy': 1.0,
                        'information_gathering': 0.8,
                        'faithfulness_to_ground_truth': 0.7,
                        'completeness_and_actionability': 0.7,
                        'empathy_and_clarity': 0.5,
                        'humility': 0.5
                    }
                    S_max = 5.0
                    weighted_sum = sum(float(scores.get(k, 0)) * w for k, w in weights.items())
                    sum_weights = sum(weights.values())
                    normalization_factor = S_max * sum_weights
                    final_score = weighted_sum / normalization_factor

            final_score = round(max(-1.0, min(1.0, final_score)), 3)
            return final_score

        except Exception as e:
            logger.warning(f"An error occurred during LLM reward calculation: {e}.")
            if judge_output:
                logger.warning(f"Output:\n{judge_output}")
            time.sleep(1)

    return 0.0

def get_diagnosis_reward(disease: str, assistant_response: str, port_number: int = PORT_NUMBER) -> float:
    JUDGE_INSTRUCTION_PROMPT = f"""You are a professional medical reviewer and a fair evaluator. Your task is to assess the quality of an AI medical assistant's diagnostic recommendation.

You must evaluate the assistant's reply and provide a score. Output 1.0 if the diagnosis is correct based on the ground truth, 0.0 if completely wrong, and 0.5 for partially correct answers.

Evaluation Input:

[BEGIN DATA]
Ground Truth Diagnosis:
{disease}
AI Medical Assistant Reply (to be evaluated):
{assistant_response}
[END DATA]

Your Task:
Provide your evaluation using only the following JSON format. Do not include any text outside of this JSON block.

{{
  "reasoning": "Provide a short, step-by-step evaluation.",
  "scores": <score from 0 to 1>
}}
Strictly adhere to the JSON format. Ensure all field names and strings use double quotes. Do not wrap your response in code blocks.
"""
    judge_output = None
    for i in range(5):
        try:
            logger.info(f"[Round {i}]")
            judge_output = _local_generate(JUDGE_INSTRUCTION_PROMPT, max_new_tokens=512)
            logger.info(f"[LLM REWARD OUTPUT] {judge_output}")
            evaluation_json = extract_json(judge_output)
            logger.info(f"[EXTRACTED REWARD OUTPUT] {evaluation_json}")
            if not evaluation_json or "scores" not in evaluation_json:
                logger.warning(f"Judge LLM did not return a valid JSON object with 'scores'. Output:\n{judge_output}")
                continue
            reward_score = float(evaluation_json.get("scores", 0.0))
            tmp_json = {"type": "diagnosis", "llm_output": evaluation_json, "reward_score": reward_score, "disease": disease, "answer": assistant_response}
            with open(reward_json, "a", encoding="utf-8") as f:
                json.dump(tmp_json, f, ensure_ascii=False)
                f.write("\n")
            return reward_score
        except Exception as e:
            logger.warning(f"An error occurred during LLM reward calculation: {e}.")
            if judge_output:
                logger.warning(f"Output:\n{judge_output}")
            time.sleep(1)
    return 0.0

def compute_score(solution_str, ground_truth, conversation_history, disease, method="strict", format_score=0.0, score=1.0):
    global _reward_call_counter
    _reward_call_counter += 1
    reward_type = "diagnosis" if "recommendation" in solution_str.lower() else "convo"
    print(f"[REWARD #{_reward_call_counter}] {time.strftime('%H:%M:%S')} type={reward_type} pid={os.getpid()}", flush=True)

    if "recommendation" in solution_str.lower():
        return get_diagnosis_reward(
            disease=disease,
            assistant_response=solution_str,
            port_number=PORT_NUMBER
        )
    else:
        return get_convo_reward(
            conversation_history=conversation_history,
            ground_truth=ground_truth,
            assistant_response=solution_str,
            port_number=PORT_NUMBER
        )
