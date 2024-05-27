from langchain_core.prompts import PromptTemplate

template = """Answer the question in your own words from the 
    context given to you.
    If questions are asked where there is no relevant context available, please answer I can only answer diabetes related 
    questions. Give concise factual answer. NO ADDITIONAL DETAILS

    Context: {context}

    Human: {question}
    Assistant:"""


diabetes_prompt_template = PromptTemplate(
    input_variables=["context",  "question"], template=template)
