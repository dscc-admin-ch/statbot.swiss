

import re

def extractSQLpart(llm_output):

    #pattern = r'select(.*?)(?:;|```|$|Explanation)'
    first_pattern=r'```sql(.*?)```'
    match = re.search(first_pattern, llm_output, re.DOTALL | re.IGNORECASE)

    second_pattern = r'(?:.*?)(SQL query|the query|way to do it|Here\'s|SQL statement|query|```|[SQL query]:|[SQL]|[SQL]:|\s*)(.*?)\s*SELECT(.*?)(?:;|```|$|Explanation|Here\'s|This query|The result is|;|\n\n)'

    # Use re.search to find the match
    match_2 = re.search(second_pattern, llm_output, re.DOTALL | re.IGNORECASE)

    pattern_ = r'```sql(.*?)```'

    # # Use re.search to find the match
    # match_ = re.search(pattern_, llm_output, re.DOTALL | re.IGNORECASE)

    # Check if a match is found
    if match:
        # Extract the matched string
        selected_string = match.group(1).strip()
    else:
        if match_2:
            # Extract the matched string
            selected_string = match_2.group(3).strip()
        else:
            selected_string=""
            print(f"Not Found: {llm_output}")
    return selected_string

   


def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)

def get_input_token_length(tokenizer, message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
    prompt = get_prompt(message, chat_history, system_prompt)
    input_ids = tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']
    return input_ids.shape[-1]


