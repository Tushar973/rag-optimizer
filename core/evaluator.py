from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Define output structure for the judge
class EvaluationScore(BaseModel):
    relevance_score: int = Field(description="Score 1-5 for relevance")
    accuracy_score: int = Field(description="Score 1-5 for accuracy based on context")
    reasoning: str = Field(description="Brief explanation of the score")

def get_judge_llm(api_key):
    return ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0)

def evaluate_response(query, context_text, generated_answer, judge_llm):
    
    parser = JsonOutputParser(pydantic_object=EvaluationScore)
    
    eval_template = """You are an expert RAG evaluator.
    
    Context provided to the system:
    {context}
    
    User Question:
    {question}
    
    System Answer:
    {answer}
    
    Evaluate the System Answer based on:
    1. Relevance (Does it answer the user's specific question?)
    2. Accuracy (Is it supported by the Context provided?)
    
    Return a JSON with 'relevance_score' (1-5), 'accuracy_score' (1-5), and 'reasoning'.
    """
    
    prompt = ChatPromptTemplate.from_template(eval_template)
    chain = prompt | judge_llm | parser
    
    try:
        result = chain.invoke({
            "context": context_text, 
            "question": query, 
            "answer": generated_answer
        })
        return result
    except Exception as e:
        return {"relevance_score": 0, "accuracy_score": 0, "reasoning": "Error in evaluation"}