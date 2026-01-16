from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# --------------------------------------------------
# Define output structure (UNCHANGED)
# --------------------------------------------------
class EvaluationScore(BaseModel):
    relevance_score: int = Field(description="Score 1-5 for relevance")
    accuracy_score: int = Field(description="Score 1-5 for accuracy based on context")
    reasoning: str = Field(description="Brief explanation of the score")

# --------------------------------------------------
# Judge LLM (UNCHANGED)
# --------------------------------------------------
def get_judge_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0.0  # 🔥 deterministic & strict
    )

# --------------------------------------------------
# Evaluation Logic (STRICTER PROMPT)
# --------------------------------------------------
def evaluate_response(query, context_text, generated_answer, judge_llm):
    
    parser = JsonOutputParser(pydantic_object=EvaluationScore)
    
    # 🔥 IMPROVED PROMPT: Added Rubrics and Hallucination Checks
    eval_template = """
You are a strict academic evaluator for Retrieval-Augmented Generation (RAG) systems. 

You are given:
- Retrieved Context
- User Question
- System Answer

Evaluate the System Answer using ONLY the retrieved context. 

SCORING GUIDELINES:

Accuracy (1–5):
- 5: Fully correct, complete, no missing key ideas, equations, or reasoning steps.
- 4: Mostly correct, minor omissions.
- 3: Partially correct, important details missing.
- 2: Mostly incorrect or superficial.
- 1: Incorrect or unsupported by the context.

Relevance (1–5):
- 5: Directly answers the question with focus.
- 3: Partially answers or slightly unfocused.
- 1: Off-topic or vague.

IMPORTANT RULES:
- If key concepts or equations present in the context are missing → Accuracy ≤ 3
- If the answer adds information NOT present in the context → Penalize accuracy (Hallucination check)
- Do NOT reward verbosity.
- Be conservative with 5/5 scores.

Context:
{context}

User Question:
{question}

System Answer:
{answer}

Return ONLY valid JSON with:
- relevance_score (int 1–5)
- accuracy_score (int 1–5)
- reasoning (short explanation)
"""
    
    prompt = ChatPromptTemplate.from_template(eval_template)
    chain = prompt | judge_llm | parser
    
    try:
        result = chain.invoke({
            "context": context_text, 
            "question": query, 
            "answer": generated_answer
        })
        
        # Return same structure your app already expects
        return {
            "relevance_score": result.relevance_score,
            "accuracy_score": result.accuracy_score,
            "reasoning": result.reasoning
        }
        
    except Exception as e:
        # Added print for debugging so you know why it failed
        print(f"Evaluation Error: {e}")
        return {
            "relevance_score": 0, 
            "accuracy_score": 0, 
            "reasoning": "Evaluation failed due to parsing or model error."
        }