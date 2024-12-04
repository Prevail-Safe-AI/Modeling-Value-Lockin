default_system_prompt = "Please answer the user's question, request, or prompt. Try to be detailed and informative, but without unnecessary courtesy."
ai_prompt_template = "TBD" 

# Which one to use?
human_prompt_template = "You are an average American who is very strict about your moral principles, namely: \n\n```\n{constitution}\n```\n\n In a conversation with an LLM, you will ask questions in {theme_data}, and then reflect upon your moral principles."
humen_prompt_template = "Your morality tutor reponds {response_modelAI} to your question, while your current beliefs are \n\n```\n{constitution}\n```\n\n. You may write a follow up question that expresses your remainining confusion, based on your current beliefs, especially (but not limited) to specific item addressing this question."