from openai import OpenAI
import os
import pandas as pd
import json
import pandas as pd

openai_key = os.getenv("OPEN_API_KEY")

client = OpenAI(api_key=openai_key)

def send_message_to_gpt(player, comment):

    json_format = {
        "sentiment_score": 0,
        "sentiment": "NEUTRAL",
        "explanation": "example",
        "translation": "example"
    }

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a linguistics expert and have been assigned to rate social media comments from soccer fans. In the following you will be given a name of a soccer player and a comment from a social media channel. Please rate it according to sentiment towards the Player. The score should be in a range from -1 to 1, where -1 is negative, 0 is neutral and 1 is positive. Please also consider whether the comment really refers to the specified player and which part of the comment refers to the sentiment against the player. Also note the emojis. please give me a sentiment score (-1 to 1), a sentiment (NEGATIVE, NEUTRAL OR POSITIVE) and an explanation. Please translate the comment into English. Please give the answer back in JSON format exact like this: {json.dumps(json_format)}"},
            {
                "role": "user",
                "content": f"Player: {player}, Comment: {comment}"
            }
        ],
        response_format={"type": "json_object"}
    )

    return completion

def analyze_sentiment(player, comment):
    completion = send_message_to_gpt(player=player, comment=comment)
    try:
        content = json.loads(completion.choices[0].message.content)
        return content["sentiment_score"], content["sentiment"], content["explanation"], content["translation"]
    except:
        print(completion.choices[0].message.content)
        return None, None, None, None
    
if __name__ == '__main__':
    print(analyze_sentiment("Alexander Nübel", "Nübel der Gott"))